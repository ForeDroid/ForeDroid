# ForeDroid detection reproduction

Extract the package and run from Linux or macOS with Python 3.10:

```sh
bash run.sh
```

The first run creates a virtual environment and installs the pinned dependencies from PyPI. Subsequent runs reuse the environment. Use `bash run.sh --experiment original`, `fdroid`, or `gpmalware` to select one experiment.

The default runner recomputes predictions from the supplied per-APK, per-scene mean anomaly scores with the supplied frozen models. It verifies each APK's label and score against the archived predictions, checks confusion counts and F1, prints the yearly results and writes fresh `predictions.csv` and `metrics.json` under `results/`. A mismatch makes the command exit with an error.

Run from the individual behavior scores or refit the model before evaluation:

```sh
bash run.sh --input behaviors
bash run.sh --input behaviors --refit
```

The second command refits the OCSVM and its normalization parameters using the bundled benign training scores and the frozen scene vocabulary. GPMalware uses the refitted original model. The frozen-model replay strictly verifies archived predictions. Refitting compares labels and scores against the archive, reports differences, and saves its own metrics under `results/refit/`.

| Directory | Contents |
| --- | --- |
| `models/` | Original 2014–2017 benign model and independent F-Droid model |
| `inputs/` | Full labeled test inventories, sparse scene means, and compressed per-behavior train/test score tables |
| `scene_assets/` | Original and F-Droid centroids, PCA, scene labels and reference library manifests |
| `expected/` | Archived per-APK predictions and evaluation metrics |
| `anomaly_detection_pipeline/` | Original model inference and fitting implementation |
| `method_source/` | Python and shell source for upstream stages, for inspection |

The original and F-Droid models each use the 2018–2022 benign and malware test cohort. GPMalware uses the original model and contains malware only. APKs without a usable scene remain in the inventory and are excluded from detection metrics. The malware-only F1 does not account for benign false alarms.

See `FORMATS.md` for the pipeline map and file schemas. End-to-end APK processing and reference-library reconstruction require the upstream inputs and tools listed in `method_source/REPRODUCIBILITY_CHECKLIST.md`. The bundled scene assets include reference manifests, while the reference vector shards are held upstream. GPMalware replay covers APK-level detection. The previously reported representation was explored on 2018–2022 and GPMalware. Model fitting used benign training inputs.

Model files are Python joblib objects. Run them only from a trusted copy of this package.
