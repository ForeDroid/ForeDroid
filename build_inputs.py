"""Prepare portable replay inputs from completed server runs."""
import argparse
import shutil
from pathlib import Path

import joblib
import pandas as pd

KEYS = ['dataset_year', 'apk_name', 'dataset_class']
COLUMNS = KEYS + ['cluster', 'anomaly_score']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--original-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data, args.output
    f = data / 'F-Droid'
    gp = data / 'RQ2/GPMalware'
    original = data / '2018-2022'
    models = {'original': args.original_model,
              'fdroid': f / 'training/anomaly_detection_pipeline/models/mean_all_model.pkl'}
    (output / 'models').mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        shutil.copyfile(model, output / 'models' / f'{name}.pkl')
    cases = [
        ('original', original / 'runs', original / 'anomaly_detection_pipeline/results', 'original'),
        ('fdroid', f / 'testing/2018-2022', f / 'testing/2018-2022/results', 'fdroid'),
        ('gpmalware', gp, gp / 'detection', 'original'),
    ]
    for name, run, results, model in cases:
        expected = output / 'expected' / name
        inputs = output / 'inputs' / name
        expected.mkdir(parents=True, exist_ok=True)
        inputs.mkdir(parents=True, exist_ok=True)
        for filename in ('metrics.json', 'predictions.csv'):
            shutil.copyfile(results / filename, expected / filename)
        inventory = pd.read_csv(expected / 'predictions.csv', sep=';', usecols=KEYS, dtype=str)
        inventory.to_csv(inputs / 'inventory.csv', index=False)
        vocabulary = set(map(str, joblib.load(models[model])['clusters']))
        scores = pd.read_csv(run / 'test/anomaly_detection_pipeline/test_top50_anomaly_scores.csv',
                             sep=';', usecols=COLUMNS, dtype={c: str for c in COLUMNS[:-1]})
        observed = scores[scores.cluster.isin(vocabulary) & scores.anomaly_score.notna()]
        means = observed.groupby(KEYS + ['cluster'], as_index=False, sort=False).anomaly_score.mean()
        means.to_csv(inputs / 'scene_means.csv.gz', index=False)
        print(f'{name}: {len(inventory):,} APKs, {len(means):,} observed scene means', flush=True)


if __name__ == '__main__':
    main()
