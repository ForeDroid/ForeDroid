#!/usr/bin/env python3
"""Replay frozen ForeDroid inference and verify the saved benchmark."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from anomaly_detection_pipeline.mean_ocsvm import predict_scores, train
from show_results import render

KEYS = ['dataset_year', 'apk_name', 'dataset_class']
CASES = {
    'original': ('original', False),
    'fdroid': ('fdroid', False),
    'gpmalware': ('original', True),
}


def measure(rows):
    scored = rows[rows.predicted_label.notna()]
    malware = scored.dataset_class.eq('malware')
    flagged = scored.predicted_label.eq(1)
    tp, fp, fn, tn = (int(mask.sum()) for mask in
                      (malware & flagged, ~malware & flagged, malware & ~flagged, ~malware & ~flagged))
    return dict(total_apks=len(rows), scorable_apks=len(scored), unscorable_apks=len(rows)-len(scored),
                TP=tp, FP=fp, FN=fn, TN=tn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                malware_only_f1=2*tp/(2*tp+fn) if tp+fn else None,
                benign_fpr=fp/(tn+fp) if tn+fp else None,
                coverage=len(scored)/len(rows) if len(rows) else None,
                auc=float(roc_auc_score(malware, scored.anomaly_score)) if malware.nunique()==2 else None)


def replay(root, name, input_kind, refit):
    model_name, malware_only = CASES[name]
    source = root / 'inputs' / name
    model = joblib.load(root / 'models' / f'{model_name}.pkl')
    types = {c: str for c in KEYS + ['cluster', 'dataset_name']}
    if refit:
        training = root / 'inputs' / model_name / 'train_behavior_scores.csv.gz'
        training_scores = pd.read_csv(training, dtype=types)
        model = train(training_scores, model['clusters'],
                      training_dataset='F-Droid' if model_name == 'fdroid' else None)
    filename = 'test_behavior_scores.csv.gz' if input_kind == 'behaviors' else 'scene_means.csv.gz'
    precision = 'round_trip' if input_kind == 'means' else None
    scores = pd.read_csv(source / filename, dtype=types, float_precision=precision)
    inventory = pd.read_csv(source / 'inventory.csv', dtype=str)
    expected = pd.read_csv(root / 'expected' / name / 'predictions.csv', sep=';',
                           dtype={c:str for c in KEYS})
    predicted = predict_scores(scores, model)
    actual = inventory.merge(predicted, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if actual['_merge'].eq('right_only').any():
        raise ValueError(f'{name}: score identity outside test inventory')
    actual = actual.drop(columns='_merge')
    paired = actual.merge(expected[KEYS+['predicted_label','anomaly_score']], on=KEYS,
                          how='outer', validate='one_to_one', suffixes=('', '_expected'), indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError(f'{name}: APK inventory mismatch')
    differences = {}
    for column in ('predicted_label', 'anomaly_score'):
        left = paired[column].to_numpy(dtype=float, na_value=np.nan)
        right = paired[f'{column}_expected'].to_numpy(dtype=float, na_value=np.nan)
        rtol, atol = (1e-8, 1e-12) if column == 'anomaly_score' else (0, 0)
        if refit:
            differences[column] = int((~np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)).sum())
        else:
            np.testing.assert_allclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
    overall = measure(actual)
    years = {} if malware_only else {year: measure(rows) for year, rows in actual.groupby('dataset_year')}
    reference = json.loads((root / 'expected' / name / 'metrics.json').read_text())
    comparisons = [(overall, reference['scorable_only'] if malware_only else reference['overall'])]
    comparisons.extend((metrics, reference['years'][year]) for year, metrics in years.items())
    for metrics, saved in comparisons:
        for key in ('TP','FP','FN','TN','f1','malware_only_f1','benign_fpr'):
            if not refit and key in saved and saved[key] is not None:
                np.testing.assert_allclose(metrics[key], saved[key], rtol=1e-12)
    if malware_only:
        overall['scorable_only'] = {key: overall[key] for key in ('TP','FP','FN','TN','precision','recall','f1')}
    results = dict(overall=overall, years=years, verified_apks=len(actual))
    if refit:
        results['differences_from_archived'] = differences
    out = root / 'results' / ('refit' if refit else '') / name
    out.mkdir(parents=True, exist_ok=True)
    actual.to_csv(out / 'predictions.csv', sep=';', index=False)
    (out / 'metrics.json').write_text(json.dumps(results, indent=2)+'\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['all', *CASES], default='all')
    parser.add_argument('--input', choices=['means', 'behaviors'], default='means')
    parser.add_argument('--refit', action='store_true', help='Refit OCSVM from bundled training behavior scores')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    names = CASES if args.experiment=='all' else [args.experiment]
    results = {name: replay(root, name, args.input, args.refit) for name in names}
    gp = results.get('gpmalware', {}).get('overall')
    if 'original' in results:
        metrics = results['original']
        print(render(metrics, sorted(metrics['years'])))
    if gp and 'original' not in results:
        print(f"GPMalware | {gp['scorable_apks']}/{gp['total_apks']} scored | F1 score {gp['f1']:.4f}")
    print('\nCompared predictions with archived results:')
    for name, result in results.items():
        compared = result['verified_apks']
        if args.refit:
            differences = result['differences_from_archived']
            print(f"  {name}: {compared:,} APKs, {differences['predicted_label']:,} label differences, "
                  f"{differences['anomaly_score']:,} score differences")
        else:
            print(f'  {name}: {compared:,} APKs verified')
    print(f"\nOutput: {root / 'results'}")


if __name__ == '__main__':
    main()
