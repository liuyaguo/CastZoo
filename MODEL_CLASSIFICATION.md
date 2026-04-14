# CastZoo Models 分类

## 统计范围

- 统计目录: `CastZoo/models`
- `.py` 文件总数: 51
- 实际模型文件数: 50
- 非模型文件: `__init__.py`

## 分类说明

- 本文档只统计 `CastZoo/models` 目录中的文件。
- 分类维度分为 5 类: 统计模型、机器学习模型、深度学习模型、FoundationModel、其他。
- 每一类下面继续按“模型家族”归位。
- 当前目录中没有的家族先保留为空，便于后续补充。

## 分类总览

| 大类 | 数量 |
| --- | ---: |
| 统计模型 | 4 |
| 机器学习模型 | 4 |
| 深度学习模型 | 35 |
| FoundationModel | 7 |
| 其他 | 1 |

## 1. 统计模型

| 模型家族 | 对应模型 | 数量 |
| --- | --- | ---: |
| ARIMA / SARIMA 家族 | `ARIMA` | 1 |
| 指数平滑 / ETS 家族 | `ETS` | 1 |
| Prophet 家族 | `Prophet` | 1 |
| Theta 家族 | `Theta` | 1 |
| 状态空间 / Kalman 家族 |  |  |
| 分解类统计模型家族 |  |  |
| 经典频域 / 概率统计家族 |  |  |

## 2. 机器学习模型

| 模型家族 | 对应模型 | 数量 |
| --- | --- | ---: |
| 线性回归 / 广义线性模型家族 |  |  |
| 树模型家族 | `RandomForest` | 1 |
| Boosting 家族 | `XGBoost`, `LightGBM`, `CatBoost` | 3 |
| 核方法 / SVM 家族 |  |  |
| 邻近法 / 聚类法家族 |  |  |

## 3. 深度学习模型

| 模型家族 | 对应模型 | 数量 |
| --- | --- | ---: |
| Transformer 家族 | `Autoformer`, `Crossformer`, `ETSformer`, `FEDformer`, `Informer`, `MultiPatchFormer`, `Nonstationary_Transformer`, `PAttn`, `PatchTST`, `Pyraformer`, `Reformer`, `TemporalFusionTransformer`, `Transformer`, `TimeXer`, `iTransformer` | 15 |
| 卷积 / CNN 家族 | `ConvTimeNet`, `KANAD`, `MICN`, `SCINet`, `TimesNet` | 5 |
| 线性家族 | `DLinear` | 1 |
| MLP / Mixer 家族 | `LightTS`, `TSMixer`, `TiDE`, `TimeMixer` | 4 |
| RNN 家族 | `SegRNN` | 1 |
| Mamba / 状态空间家族 | `Mamba`, `MambaSimple`, `MambaSingleLayer` | 3 |
| 图神经网络家族 | `MSGNet` | 1 |
| 频域 / 谱方法家族 | `FiLM`, `FreTS`, `TimeFilter` | 3 |
| 小波家族 | `WPMixer` | 1 |
| Koopman / 动力系统家族 | `Koopa` | 1 |

## 4. FoundationModel

| 模型家族 | 对应模型 | 数量 |
| --- | --- | ---: |
| Chronos 家族 | `Chronos`, `Chronos2` | 2 |
| Moirai 家族 | `Moirai` | 1 |
| TimesFM 家族 | `TimesFM` | 1 |
| TiRex 家族 | `TiRex` | 1 |
| Causal LM 时序家族 | `Sundial`, `TimeMoE` | 2 |

## 5. 其他

| 模型家族 | 对应模型 | 数量 |
| --- | --- | ---: |
| 包初始化文件 | `__init__.py` | 1 |

## 备注

- 这里的“家族”是按模型主干结构或使用方式划分，不是按任务划分。
- `FoundationModel` 从技术上看也属于深度学习，但为了和普通可训练模型区分，这里单独列出。
- 统计模型和机器学习模型已按 `PLAN.md` 补齐；仍未覆盖的家族继续保留为空位。
