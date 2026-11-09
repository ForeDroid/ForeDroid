# ForeDroid 复现包上传清单

主实验使用 2014–2017 年 benign 训练，并在 2018–2022 年 benign 和 malware 上评测。复现包从已保存的中间产物开始运行。上传范围为代码、训练好的模型、中间文件和用于对照的结果。

## 主实验必传

| 内容 | 文件 |
| --- | --- |
| 代码 | `pipeline/run_inference.py`、`anomaly_detection_pipeline/mean_ocsvm.py`、`show_results.py`、`output_layout.py`。如发布完整方法源码，同时保留各阶段的源代码、脚本和静态配置。 |
| 正式 OCSVM | `anomaly_detection_pipeline/models/mean_all_model.pkl`，保留正式路径。 |
| 测试行为分数 | `ForeDroid_output/2018-2022/runs/test/anomaly_detection_pipeline/test_top50_anomaly_scores.csv`。 |
| 全量测试清单和预期结果 | `ForeDroid_output/2018-2022/anomaly_detection_pipeline/results/{predictions.csv,metrics.json}`。重算时只从预测表读取 `dataset_year`、`apk_name`、`dataset_class` 三列作为清单，其余预测字段用于对照。清单保留全部 20,000 个样本。 |

从复现包根目录运行以下命令。它使用模型和中间分数重新判定，输出写入 `replayed_results/`。

```bash
python pipeline/run_inference.py \
  --root ForeDroid_output/2018-2022/runs \
  --inventory ForeDroid_output/2018-2022/anomaly_detection_pipeline/results/predictions.csv \
  --replay-output replayed_results
```

`replayed_results/metrics.json` 包含整体及逐年结果。用它与上传的 `metrics.json` 对照。建议在说明中写明保存模型时使用的 scikit-learn 版本。

## 需要重跑更早阶段时再加

- 从测试描述重算场景和行为分数：`ForeDroid_output/2018-2022/runs/test/final_training_data_with_desc.csv`，以及 `ForeDroid_output/2014-2017/train/cluster_pipeline/` 的 GUI、后台场景资产和 `ForeDroid_output/2014-2017/anomaly_detection_pipeline/precomputed_vectors/`（包含 manifest 和向量分片）。
- 重建训练参考库：再加 `ForeDroid_output/2014-2017/train/final_training_data_with_desc.csv`。若只要从分数重训 OCSVM，再加 `ForeDroid_output/2014-2017/train/anomaly_detection_pipeline/train_top50_anomaly_scores.csv`，用已上传的正式模型提供冻结场景词表。
- F-Droid 独立实验：在 `ForeDroid_output/F-Droid/` 下保留 `training_manifest.json`、`training/results/coverage.json`、`training/anomaly_detection_pipeline/models/mean_all_model.pkl`，以及 `testing/2018-2022/inventory.json`、`testing/2018-2022/test/anomaly_detection_pipeline/test_top50_anomaly_scores.csv` 和 `testing/2018-2022/results/{metrics.json,predictions.csv}`。若需从描述重跑，附 F-Droid 自己的场景资产与参考向量，测试描述可复用主实验文件。
- GPMalware 若纳入发布，单独保留 `ForeDroid_output/RQ2/GPMalware/detection/{metrics.json,predictions.csv}` 以及 `ForeDroid_output/RQ2/GPMalware/test/anomaly_detection_pipeline/test_top50_anomaly_scores.csv`。RQ2 技术标签评测另附对应代码、标注、样本映射与报告。

上传时排除原始 APK、环境目录、实际 `api.json`、凭据、临时文件、缓存、日志和历史试验。若打包符号链接指向的中间文件，应收录实际内容并在包内使用相对路径。
