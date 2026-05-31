# 两类相似搜寻算法研究综述

## 1. 最新进展是什么

- Matrix Profile / MASS：适合 motif、join、shapelet、segmentation 等相似片段任务，优点是参数少、可扩展，适合作为“历史时序相似”的候选预筛层。当前库 v1 先把 STUMPY/MASS 作为可选 benchmark，不替换默认搜索。
- 弹性距离族：DTW、ShapeDTW、WDDTW、LCSS、ERP、TWE 等能缓解局部速度错位，适合验证“肉眼形态一致但日 K 不完全对齐”的样本。当前库先接 `dtw_optional`，未安装依赖时显式不可用。
- shapelet / 字典类方法：更适合有标注任务，例如识别某类走势模板。当前库还没有人工标签库，暂不进入主链路。
- Time-series foundation models：TimesFM、MOMENT、Chronos 等近期进展主要集中在预测或通用表征。它们适合后续做 embedding 实验，但 v1 不作为主验收算法，避免引入重依赖和不可解释排序。

## 2. 著名专家和团队

- Eamonn Keogh / UCR：Matrix Profile、时间序列相似搜索与 UCR archive 的核心团队，适合作为 motif/search 方向的第一研究入口。
- Abdullah Mueen：Matrix Profile / MASS 相关核心作者之一，适合跟踪可扩展 subsequence search。
- Anthony Bagnall / aeon-sktime：时间序列分类、距离族和 benchmark 工具链，适合参考工程化评测口径。
- tslearn 社区：Python 下 DTW、soft-DTW、shapelets 等常用实现，适合作为可选依赖。
- CMU MOMENT、Google TimesFM、Amazon Chronos：2024 后 foundation model 方向代表团队，适合进入后续表征实验观察清单。

## 3. 技术社区和工具

- UCR Matrix Profile：理论、论文和任务边界参考。
- STUMPY：Python Matrix Profile 工具，可用于历史时序相似的预筛 benchmark。
- aeon / sktime：时间序列算法生态，距离族、分类与 benchmark 组件完整。
- tslearn：DTW、soft-DTW、shapelet 等实现更直接，适合轻量验证。
- HuggingFace time-series models：TimesFM、MOMENT、Chronos 等模型生态，暂列研究观察。

## 4. 重要文章和工具取舍

| 方向 | 工具/文章 | 当前用途 | 是否纳入 v1 |
| --- | --- | --- | --- |
| Matrix Profile / MASS | UCR Matrix Profile、STUMPY | 历史时序候选预筛与研究 benchmark | 可选 benchmark |
| 价格路径 baseline | 当前库归一化价格路径 + 手工特征 | 保持默认可回退 | 默认启用 |
| 收益形态 | 对数收益路径 z-normalize | 验证形态变化相似度 | 启用 |
| Hybrid v2 | 价格路径 + 收益路径 + 手工特征 | 主推候选算法 | 启用 |
| DTW / ShapeDTW | aeon / tslearn | 弹性距离核验 | 可选启用 |
| Foundation models | TimesFM / MOMENT / Chronos | 后续 embedding 实验 | 暂不纳入 |

## 当前库落地口径

- 默认算法保持 `baseline_price_feature`，不改变历史时序和横截面旧结果口径。
- 新增 `return_shape` 和 `hybrid_shape_v2`，用于核验“肉眼形态一致性”。
- 新增 `dtw_optional` 和 `mass_optional_history` 状态注册；依赖缺失时显式显示不可用，不影响主功能。
- 新增 benchmark 命令和 HTML 图集，用固定样本比较算法 Top 结果。

## 参考入口

- UCR Matrix Profile: https://www.cs.ucr.edu/~eamonn/MatrixProfile.html
- STUMPY docs: https://stumpy.readthedocs.io/en/latest/Tutorial_The_Matrix_Profile.html
- aeon distances: https://aeon-tsml.readthedocs.io/en/latest/api_reference/distances.html
- tslearn metrics: https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.metrics.html
- TimesFM: https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/
- MOMENT: https://github.com/moment-timeseries-foundation-model/moment
