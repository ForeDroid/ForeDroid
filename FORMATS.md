# ForeDroid pipeline and file formats

| Stage | Output | Format and contents |
| --- | --- | --- |
| GUI extraction | `appSerialized.txt`, `outputBenignUi_train.txt`, `api_train.txt` | XML GUI model per APK and semicolon-delimited GUI/API export tables. |
| ICCBot | `rule_analysis_report.json` | JSON call-chain report per APK. |
| GUI merge | `merge_ui_api_with_relation.txt` | Semicolon-delimited table with APK, widget ID, callback, API and relation strength. |
| Behavior fusion | `final_training_data.csv` | Comma CSV. One row per behavior with APK identity, entry method, sensitive API, call chain and available GUI context. |
| Description | `final_training_data_with_desc.csv` | Comma CSV with additional `generated_description` and `status` columns. |
| Scene assignment | `all_apks_with_clusters.csv` | Semicolon CSV with scene IDs. Scene ID `-1` denotes an unassigned behavior. Training produces NumPy centroids, JSON labels and a pickled background PCA transform. |
| Embedding and reference library | `embeddings.npy`, `cluster_<id>.npy`, `cluster_<id>_apk.json`, `manifest.json` | NumPy vector matrices and JSON metadata. Within each scene the APK list aligns with vector row order. |
| Behavior scoring | `*_top50_anomaly_scores.csv` | Semicolon CSV with average cosine similarity, anomaly score, reference count, neighbors used and scoring status. The score is one minus the mean similarity to up to 50 references in the same scene, excluding references from the same APK. |
| APK model and evaluation | Model `.pkl`, `predictions.csv`, `metrics.json` | Joblib model, semicolon-delimited APK predictions and JSON evaluation metrics. The APK anomaly score is the negated OCSVM decision function. |

The bundled `*_behavior_scores.csv.gz` files retain every behavior row from the upstream score tables, including missing scores, and project them to the fields used by training and prediction. They are gzip-compressed **comma CSV**, with columns `dataset_year`, `apk_name`, `dataset_class`, `cluster`, and `anomaly_score`. F-Droid training adds `dataset_name` and uses year `0` as a dataset marker. Duplicate rows contribute individually to the scene mean. An empty score is missing. A numeric zero is an observed score.

The bundled `scene_means.csv.gz` files use the same first five fields and contain one row per observed test APK and scene. Test identity is `dataset_year` plus `apk_name`. The original and F-Droid experiments carry independent models and scene vocabularies.
