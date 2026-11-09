# 输出目录

服务器根目录：`/home/lb/data/Dataset/ForeDroid_output`。

```text
2014-2017/                    仅2014–2017 benign训练
  2014/ … 2017/              逐年提取、merge和融合输出
  train/                     训练融合、描述、场景、行为分数
  anomaly_detection_pipeline/ 训练嵌入、参考库、模型及断点
2018-2022/                    2018–2022 benign和malware测试
  2018/ … 2022/              逐年提取、merge和融合输出
  runs/test/                 全部测试数据、描述、场景与行为分数
  runs/anomaly_detection_pipeline/ 统一测试嵌入、评分断点和结果
  runs/state.sqlite          全部测试提取状态
  runs/descriptions.sqlite   统一描述缓存
  anomaly_detection_pipeline/results/
    predictions.csv          全部20000个原始测试APK的对齐预测
    metrics.json             总体及逐年指标，包含覆盖率
    2018/ … 2022/           逐年预测及指标
```

所有测试计算文件已经统一，不再按原运行批次分目录。记录ID已统一编号，年份保留在dataset_year字段中。逐年提取数据直接读取外层年份目录。历史实验目录已按要求删除。

## 检测与监控

在服务器 `/home/lb/tool/ForeDroid` 中运行：

```bash
/home/lb/venvs/foredroid/bin/python pipeline/run_inference.py
/home/lb/venvs/foredroid/bin/python monitor_inference.py
```

默认测试2018–2022，使用统一断点；训练资产读取2014-2017。输入未变且阶段已完成时复用结果，不重新调用LLM或训练。当前OCSVM为正式项目内的 `anomaly_detection_pipeline/models/mean_all_model.pkl`，使用场景均值、训练均值填缺失、场景覆盖归一化、训练集RMS尺度校准和原生0边界。训练与测试使用相同变换；训练按APK哈希去重，测试按年份+APK识别。表示方案曾在GPMalware及2018–2022查看效果，结果属于探索性评估；模型与尺度仅用2014–2017 benign拟合。仅模型更新时，推理入口复用已有行为分数并重算最终检测，不重新调用LLM。

## 从提取阶段执行

```bash
# 训练提取：默认仅2014–2017 benign
/home/lb/venvs/foredroid/bin/python batch_extract.py
# 测试提取：恢复统一状态库
/home/lb/venvs/foredroid/bin/python batch_extract.py --output /home/lb/data/Dataset/ForeDroid_output/2018-2022 --years 2018 2019 2020 2021 2022
# 测试融合，GUI merge仍需手动执行原merge_ui_api.py
/home/lb/venvs/foredroid/bin/python pipeline/pipeline.py --output-root /home/lb/data/Dataset/ForeDroid_output/2018-2022 --years 2018 2019 2020 2021 2022
# 测试描述
/home/lb/venvs/foredroid/bin/python pipeline/generate_description.py --output-root /home/lb/data/Dataset/ForeDroid_output/2018-2022 --splits test --workers 32
```

训练的融合、描述、聚类、参考库入口默认使用2014-2017；测试根目录到runs的路由位于 `output_layout.py`。Shell默认路径、监控和描述复用缓存也已更新。可用OUTPUT_ROOT或Python对应参数覆盖。原始APK目录apks_period不变。

旧训练/测试合并入口保留用于复现旧模型。当前均值版检测应使用pipeline/run_inference.py；重新训练均值版使用anomaly_detection_pipeline/mean_ocsvm.py。

迁移保留数据、模型、描述缓存、嵌入、正常参考向量、行为分数和断点。统一记录ID时同步更新了缓存中的ID，未重新运行模型；已验证全部20000个APK的预测及连续分数与合并前一致。

已在原始批次目录不可访问的条件下跑通统一检测流程，验证全部1721522条描述均有成功缓存，20000个APK预测保持一致；根据用户授权，已删除不再依赖的原始批次目录。
