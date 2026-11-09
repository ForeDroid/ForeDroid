"""Export the score columns required for refitting and replay."""
import argparse
import gzip
import shutil
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--package', type=Path, required=True)
args = parser.parse_args()
data, package = args.data, args.package
original = data / '2014-2017'
fdroid = data / 'F-Droid'
runs = {
    'original': (original, data / '2018-2022/runs'),
    'fdroid': (fdroid / 'training', fdroid / 'testing/2018-2022'),
    'gpmalware': (None, data / 'RQ2/GPMalware'),
}
for name, (training, testing) in runs.items():
    target = package / 'inputs' / name
    for split, root in [('train', training), ('test', testing)]:
        if root is None:
            continue
        columns = ['dataset_year', 'apk_name', 'dataset_class', 'cluster', 'anomaly_score']
        if name == 'fdroid' and split == 'train':
            columns.append('dataset_name')
        source = root / split / 'anomaly_detection_pipeline' / f'{split}_top50_anomaly_scores.csv'
        destination = target / f'{split}_behavior_scores.csv.gz'
        with gzip.open(destination, 'wt', encoding='utf-8', newline='', compresslevel=3) as output:
            chunks = pd.read_csv(source, sep=';', usecols=columns, dtype=str,
                                 keep_default_na=False, chunksize=50000)
            count = 0
            for chunk in chunks:
                chunk[columns].to_csv(output, index=False, header=count == 0)
                count += len(chunk)
        print(f'{name} {split}: {count:,} behavior rows', flush=True)
    if training is None:
        continue
    assets = package / 'scene_assets' / name
    assets.mkdir(parents=True, exist_ok=True)
    scenes = training / 'train/cluster_pipeline'
    for branch in ('gui', 'bg'):
        folder = scenes / f'{branch}_centroids'
        for suffix in ('centroids.npy', 'cluster_ids.json', 'scenario_labels.json', 'cluster_summary.csv'):
            path = folder / f'{branch}_{suffix}'
            shutil.copyfile(path, assets / path.name)
    for path in (scenes / 'bg_centroids/bg_pca_model.pkl',
                 scenes / 'output/cluster_names.csv', scenes / 'output/cluster_id_mapping.csv'):
        shutil.copyfile(path, assets / path.name)
    manifest = training / 'anomaly_detection_pipeline/precomputed_vectors/manifest.json'
    shutil.copyfile(manifest, assets / 'reference_manifest.json')
