# dl-models：机器学习与深度学习实验记录

研究生阶段积累的模型学习、算法实现与实验脚本，涵盖传统机器学习、深度学习、特征选择和不平衡数据分类。部分实验面向生物信息学数据，包括基因表达数据和肽序列相关任务。

这里保留了不同时期的学习示例、模型实现、交叉验证脚本、数据处理工具和实验结果，便于回顾研究过程与查找代码。各实验的参数、数据格式和依赖有所不同，使用时以对应脚本为准。

## 内容导航

- [模型索引](#模型索引)
- [特征选择与实验分析](#特征选择与实验分析)
- [目录说明](#目录说明)
- [使用说明](#使用说明)
- [参考资料](#参考资料)

## 模型索引

表中列出主要实现与代表性入口；同一模型的其他实验变体可在根目录及对应子目录中查找。

### 神经网络与生成模型

| 模型 | 主要内容 | 实现与实验入口 |
| --- | --- | --- |
| FNN | 前馈神经网络，以及 Batch Normalization、Layer Normalization 变体 | [nn/](nn/)、[基础版](fnn-kfold-run.py)、[BN 版](fnn-bn-kfold-run.py)、[LN 版](fnn-ln-kfold-run.py) |
| CNN | 卷积神经网络、二分类与肽序列特征实验 | [cnn/](cnn/)、[交叉验证](cnn-k-fold-run.py)、[特征实验](cnn-features-run.py) |
| RNN / BiRNN | 循环神经网络、双向循环网络与预测实验 | [rnn/](rnn/)、[RNN 交叉验证](rnn-kfold-run.py)、[BiRNN 交叉验证](birnn-kfold-run.py)、[预测入口](rnn-pred-run.py) |
| SNN | 自归一化神经网络，使用 SELU 与 Alpha Dropout | [snn/](snn/)、[交叉验证](snn-k-fold-run.py)、[新版实验](snn-new-kfold-run.py) |
| CapsNet | 胶囊网络与肽数据实验 | [capsNet/](capsNet/)、[肽数据入口](capsNet/peptide_capsnet.py) |
| GAN | 生成对抗网络学习示例 | [gan/gan.py](gan/gan.py) |
| VAE | 变分自编码器学习示例 | [vae/vae.py](vae/vae.py) |

### 树模型与集成学习

| 模型 | 主要内容 | 实现与实验入口 |
| --- | --- | --- |
| Decision Tree | 决策树分类、树结构与规则导出 | [dt/](dt/)、[交叉验证](dt/sk-dt-kfold-run.py)、[规则导出](dt/rules-export.py) |
| Random Forest | 随机森林分类与 SMOTE 实验 | [rf/](rf/)、[scikit-learn 入口](sk-rf-kfold-run.py)、[SMOTE 入口](sk-rf-kfold-smote-run.py) |
| Extra Trees / GBDT | 极端随机树与梯度提升树分类 | [Extra Trees](sk-ert-kfold-smote-run.py)、[GBDT](sk-gbt-kfold-smote-run.py) |
| XGBoost | 梯度提升、并行与不平衡分类实验 | [基础入口](xgb-kfold-run.py)、[并行入口](xgb-kfold-run-parallel.py)、[不平衡分类](imb-xgb-kfold-run.py) |
| LightGBM | 梯度提升与 SMOTE 实验 | [并行入口](lgb-kfold-run-parallel.py)、[SMOTE 入口](lgb-smote-kfold-run.py) |
| CatBoost / ThunderGBM | 其他梯度提升框架实验 | [CatBoost](cgb-kfold-smote-run.py)、[ThunderGBM](thundergbm-kfold-smote.py) |
| Deep Forest | 多粒度级联森林及其他深度森林实验 | [deepForest/](deepForest/)、[交叉验证](deep-forest-kfold-run.py)、[SMOTE 入口](sk-deep-forest-kfold-smote-run.py) |
| RuleFit / Stacking | 规则模型、规则输出与模型融合 | [RuleFit](rulefit-k-fold-run.py)、[规则输出](rulefit-rules-output.py)、[Stacking](sci-fs/stacking_fs.py) |

### 其他机器学习方法

| 方法 | 主要内容 | 实验入口 |
| --- | --- | --- |
| SVM / SVR / ThunderSVM | 支持向量分类、回归与 ThunderSVM 实验 | [SVM](sk-svm-kfold-smote-run.py)、[SVR](sk-svr-kfold-run.py)、[ThunderSVM](tsvm-k-fold-run.py) |
| KNN / Nearest Centroid | 近邻与最近质心分类 | [KNN](sk-knn-kfold-run.py)、[Nearest Centroid](sk-ncentroid-kfold-smote-run.py) |
| Naive Bayes / Gaussian Process | 朴素贝叶斯与高斯过程分类 | [Naive Bayes](sk-bayes-kfold-smote-run.py)、[Gaussian Process](sk-gpc-kfold-smote-run.py) |
| Linear / SGD / LDA / QDA | 线性模型、随机梯度下降与判别分析 | [Linear](sk-linear-kfold-smote-run.py)、[SGD](sk-sgd-kfold-smote-run.py)、[LDA](sk-lda-kfold-smote-run.py)、[QDA](sk-qda-kfold-smote-run.py) |
| Multi-label Classification | 基于 Binary Relevance 与 SVC 的多标签分类 | [multi-label/sk-mutli-label.py](multi-label/sk-mutli-label.py) |
| Symbolic Regression | 基于遗传编程的符号回归 | [gp-symbolic-regressor.py](gp-symbolic-regressor.py) |

## 特征选择与实验分析

| 方向 | 内容与入口 |
| --- | --- |
| 特征筛选与排序 | [Boruta](boruta-features.py)、[SVM-RFE](svm/svm-rfe.py)、[特征选择 Notebook](notebook/features-selection-learn.ipynb) |
| 增量特征选择与模型比较 | [sci-fs/](sci-fs/)，包括特征排序、特征子集评估、多分类器比较与 Stacking 实验 |
| 不平衡学习 | 根目录下的 `*-smote-*.py`，以及 [SMOTE 数据导出](smote-output.py)、[欠采样数据导出](undersampling-output.py)、[多类别 SMOTE Notebook](notebook/smoteMultiClass.ipynb) |
| 模型评估 | [utils/model_evaluation.py](utils/model_evaluation.py)，包含 ACC、MCC、混淆矩阵及分类报告 |
| 降维与可视化 | [PCA](utils/d_pca.py)、[t-SNE](tsne/t_sne.py)、[绘图脚本](plt/)，包括 IFS 曲线与 AUC 曲线 |
| 数据预处理与转换 | [files-ops/](files-ops/)、[utils/](utils/)，包括 CSV 操作、肽序列转换、标准化及数据导出 |

## 目录说明

模型目录见上方索引，其他辅助目录如下：

| 路径 | 说明 |
| --- | --- |
| 根目录 `*.py` | 模型训练、交叉验证、预测与结果导出入口 |
| [sci-fs/](sci-fs/) | 特征选择、模型比较与组合实验 |
| [notebook/](notebook/) | 数据探索、预处理、特征选择与工具学习笔记 |
| [files-ops/](files-ops/) | 数据集整理与文件转换 |
| [utils/](utils/) | 公共评估函数与数据处理工具 |
| [plt/](plt/)、[tsne/](tsne/) | 实验曲线、降维与可视化 |
| [scripts/](scripts/) | 批量训练、定时任务及日志提取的 Shell / AWK 脚本 |
| [tf-test/](tf-test/)、[ml-test/](ml-test/)、[numpy-test/](numpy-test/) | 框架、算法与基础工具的学习示例 |
| [multi-label/](multi-label/)、[ib-learn/](ib-learn/) | 多标签分类与不平衡学习实验 |
| [go/](go/)、[bio-chen/](bio-chen/) | 生物信息学相关下载脚本与数据来源说明 |
| [logs/](logs/) | 部分历史实验的 ACC、MCC 等结果记录 |
| [build/](build/) | 历史 ThunderSVM 编译产物 |

常见文件名中的 `kfold` / `k-fold` 表示交叉验证，`smote` 表示包含过采样步骤，`pred` 表示预测相关实验，`fs` 表示特征选择。具体行为仍需查看脚本实现。

## 使用说明

### 历史环境与依赖

原始 README 记录的环境为 **Python 3.5 / 3.6、TensorFlow 1.2+、PyTorch 0.4+**。这些版本反映早期实验背景；后续脚本也有补充，不能将其视为整个仓库的统一兼容范围。

仓库目前没有统一的依赖清单或锁定环境，需根据选用的脚本准备依赖：

- 基础数据处理与机器学习：NumPy、pandas、SciPy、scikit-learn。
- 神经网络：按实现选择 TensorFlow 或 PyTorch。
- 专项实验：imbalanced-learn、XGBoost、LightGBM、CatBoost、ThunderSVM、ThunderGBM、RuleFit、Boruta、scikit-multilearn、gplearn 等。
- Notebook 与绘图：Jupyter、Matplotlib、Seaborn 等。

部分代码保留了 `np.int`、`fit_sample` 和 TensorFlow 1.x 接口，复用时可能需要适配依赖。ThunderSVM 还需对应平台的动态库；仓库中的 `build/lib/libthundersvm.so` 是历史 Linux 编译产物。

### 数据约定

多数表格分类脚本通过 `--datapath` 读取带表头的 CSV：**第一列为标签，其余列为数值特征**。例如：

```csv
label,feature_1,feature_2,feature_3
1,0.12,0.35,0.80
2,0.67,0.21,0.44
```

上面仅展示列结构，不是可直接用于交叉验证的完整数据集。[sk-rf-kfold-run.py](sk-rf-kfold-run.py) 等脚本会对标签执行 `y - 1`，而其他脚本可能保留原标签，运行前需核对标签编码。

多标签入口使用 ARFF，序列与胶囊网络实验也有各自的数据处理约定。部分脚本引用本地路径或未随仓库提供的数据，需自行准备并调整路径。

### 查看参数与运行入口

在仓库根目录执行以下命令，可查看两个代表性入口的参数：

```bash
python sk-rf-kfold-run.py --help
python fnn-kfold-run.py --help
```

准备好兼容环境与数据后，随机森林的交叉验证命令示例如下，`path/to/dataset.csv` 需替换为实际文件路径：

```bash
python sk-rf-kfold-run.py --datapath path/to/dataset.csv --ntrees 100 --kfolds 5 --randomseed 0
```

不同入口的参数不完全一致，部分学习示例直接在源码中设置路径和超参数。历史脚本中的采样与评估流程也需结合实验目的检查；例如 [sk-rf-kfold-smote-run.py](sk-rf-kfold-smote-run.py) 在交叉验证前对全量数据执行 SMOTE，若用于新的泛化性能评估，应将采样限制在各训练折内，避免数据泄漏。

## 参考资料

以下保留原始记录中的教程、开源实现与相关方法链接。部分代码参考或改编自这些项目，具体来源也可查看文件内注释；列出的资料不一定都在本仓库中有对应实现。

### 框架与学习资源

- [TensorFlow Models](https://github.com/tensorflow/models)
- [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples)
- [TensorFlow Machine Learning Cookbook](https://github.com/nfmcclure/tensorflow_cookbook)
- [Data Science Python Notebooks](https://github.com/donnemartin/data-science-ipython-notebooks)
- [PyTorch Tutorial for Deep Learning Researchers](https://github.com/yunjey/pytorch-tutorial)

### 神经网络与图表示学习

- [Awesome RNN](https://github.com/kjw0612/awesome-rnn)
- [TensorFlow Generative Model Collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections)
- [Self-Normalizing Neural Networks](https://github.com/bioinf-jku/SNNs/blob/master/README.md)
- [OpenNE：Network Embedding](https://github.com/thunlp/OpenNE)
- [GEM：Graph Embedding Methods（原记录中的 SDNE 参考）](https://github.com/palash1992/GEM)
- [Graph Convolutional Networks](https://github.com/tkipf/gcn)
- [graph2vec](https://github.com/benedekrozemberczki/graph2vec)

### 分类、规则与不平衡学习

- [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)
- [RuleFit](https://github.com/christophM/rulefit)
- [ThunderSVM](https://github.com/Xtra-Computing/thundersvm)
- [XGBoost](https://github.com/dmlc/xgboost)
- [LightGBM](https://github.com/Microsoft/lightgbm/)
- [Skope-rules](https://github.com/scikit-learn-contrib/skope-rules)
