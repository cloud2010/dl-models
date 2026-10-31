# 旧研究档案检视与代码整理

这次整理从原有的 172 个 Python 文件、19 个 Notebook 和 16 份指标日志入手，检查目录、语法、重复函数、数据读取、交叉验证与评估方式。完整静态清单见 [archive-review.json](archive-review.json)，可通过 `python scripts/review_archive.py` 重新生成。

主要重构对象是个人实验中反复出现的基础流程。第三方模型实现与不同研究任务的实验差异继续保留；静态检查通过不等于所有历史脚本都能在现代环境中运行，也不代表原始研究结果已重新复现。

## 当年的工作脉络

| 时间线 | 仓库中的证据 | 形成的工作内容 |
| --- | --- | --- |
| 2017 | 首次提交 `133b9df`（2017-08-04）；`tf-test/`、`ml-test/`、`numpy-test/`、`snn/`、`cnn/` | 从 Python、NumPy、SVM、TensorFlow 学习走向分类实验、SELU、自归一化网络和卷积网络 |
| 2018 | `capsNet/peptide_capsnet.py` 的首次提交 `9e715e7`（2018-04-22）；`rnn/` | 序列表示、肽相关任务与胶囊网络实验 |
| 2019 | `nn/fnn.py` 的首次提交 `f85f973`（2019-04-26）；BN/LN 支持提交 `11e59bf`（2019-10-15） | PyTorch 实现、不同归一化方式、分类器比较、不平衡和多标签学习 |
| 2020 | `sci-fs/clf_fs.py` 的首次提交 `47e833e`（2020-05-29） | 特征排序、增量特征选择、多模型比较和组合实验 |
| 后续延伸 | `gp-symbolic-regressor.py` 的首次提交 `fd78e9e`（2023-06-04） | 符号回归、SVR 等补充探索 |

日期是所列文件在当前历史中的首次提交，不一定等于方法开始研究的时间，也不据此推断入学或毕业时间。

这些材料反映出的核心积累是：从具体生物数据出发，比较不同表示和分类方法，再通过特征选择、采样和多种指标分析结果。代码包含教程移植和开源方法的实验应用，不把引用的方法或图表解释为原创算法、论文成果或新的性能结论。

## 已修正的问题

| 问题 | 旧行为 | 本次调整 |
| --- | --- | --- |
| 分类流程重复 | 16 个 scikit-learn 入口重复读取 CSV、采样、交叉验证和评估 | 统一到 `experiments/classifiers.py`；旧文件保留说明、日期及入口调用 |
| 采样与验证混在一起 | 多个入口在完整数据上先执行 SMOTE，再划分交叉验证；部分还把合成样本计入评估 | 将 SMOTE 放入训练折的 Pipeline；测试折只包含原始样本 |
| 类别与划分假设 | 标签统一减一、部分参数中的随机种子未用于折划分 | sklearn 入口保留原始标签；显式分层划分与随机种子；检查最小类别样本数 |
| 重复拟合与多分类 AUC | ROC 入口分别训练预测模型和概率模型；只取第二列计算 AUC | 每折拟合一次，概率与预测来自同一模型；二分类与多分类分别计算 AUC |
| 评估函数复制 | 多份 MCC、混淆矩阵及分类报告代码；零分母风险；正负类名字与真实标签未绑定 | 统一到 `experiments/metrics.py`；退化矩阵返回 MCC=0，使用实际标签，保留旧导入路径 |
| FNN 训练重复 | 普通 FNN、BN、LN 重复整个训练循环 | 共享架构与 `nn/training.py`，保留原来的层名、拓扑与 `run()` 签名 |
| 神经网络跨折状态 | 模型或 Adam 在循环外创建；只重置部分权重，未重置优化器与归一化状态 | FNN / BN / LN / PyTorch SNN / BiRNN 在每折重建模型与优化器 |
| 验证模式 | 测试阶段未显式 `eval()`；FNN 的 `squeeze()` 会移除单样本批次维 | 测试使用 `eval()` 与 `no_grad()`；输出维持 `[样本数, 类别数]` |
| BiRNN 序列维度 | 用全部特征数作为每个时间步维度，序列长度大于 1 时会改变样本数 | 按每行样本重排为 `[样本数, 序列长度, 每步特征数]`；检查可整除；隐藏状态跟随输入设备 |
| 回归目标处理 | SVR 把连续目标减一，并输出无意义的分类计数 | 保留连续目标原值，使用有种子的 KFold，输出无索引 CSV 与回归指标 |
| CNN 数据处理重复 | 多份目录读取、矩阵补零和截断代码；使用旧 NumPy 别名 | 统一到 `cnn/data.py`，显式校验矩阵宽度、按文件名稳定排序；CNN 网络结构继续保留 |
| 特征选择的评估顺序 | 历史 `sci-fs` 中部分实验先在全量数据上排序，再交叉验证 | 新入口 `python -m experiments.feature_selection` 在各训练折内拟合 SelectKBest；保留历史的其他排序与模型组合探索 |
| 测试没有断言 | 原 `test_utils.py` 仅打印“所有测试通过” | 改为真实的 unittest 回归测试入口 |

折内采样与预处理的方式参考 [imbalanced-learn 官方说明](https://imbalanced-learn.org/stable/common_pitfalls.html) 和 [scikit-learn 官方说明](https://scikit-learn.org/stable/common_pitfalls.html)。

由于修正了采样、划分和训练状态，新结果可能与旧日志不同。历史日志原样保存；不能直接用旧曲线给新流程作性能背书。特征数量的比较仍是探索性实验，如需报告“选择最佳特征数之后”的性能，应另外做外层验证或使用独立测试集。

## 重复内容如何处理

- 原来的分类脚本仍然可执行，代码主体移到共享模块，不再复制维护。`--ntrees`、`--mdepth`、KNN 的 `-n` 等原有含义保留；RF 的 `--epochs` 保留为兼容参数，不参与训练。
- `utils/` 和各模型目录的评估模块保留兼容导出，避免破坏旧脚本和导入路径。XGBoost 工具中独有的辅助函数签名也保留。
- FNN / SNN / BiRNN 共用训练循环，但不同模型的层结构和初始化差异继续保留。BN/LN 没有扩展到新的层，避免把架构改变混入代码清理。
- 不把 CNN、RNN、CapsNet 或特征选择的不同研究设定当作重复文件直接删除。旧示例中剩余的同源网络代码可在静态报告中查到。
- Lasso 是回归器，旧分类入口的 `--classifier lasso` 现在明确报错；不会悄悄把连续输出当作类别。该入口可继续选择 `ridge`。

## 保留的第三方实现与历史边界

ThunderSVM 的 ctypes 绑定与编译库、Deep Forest 的移植实现、CapsNet 的参考网络与层、GAN/VAE 教程及 TensorFlow 1.x 示例，没有进行框架重写或第三方算法内部改造。它们的来源继续保留在文件注释和 README 中。

以下内容完成了静态检查，但没有承诺现代环境可直接执行：

| 范围 | 状态与后续使用方式 |
| --- | --- |
| `tf-test/`、旧 TensorFlow CNN / RNN / RF / SNN | 保留 TensorFlow 1.x 的 Session、contrib、旧数据接口；本次环境没有 TensorFlow，未训练验证 |
| XGBoost / LightGBM / CatBoost / ThunderGBM / RuleFit / 第三方 Deep Forest 入口 | 不升级其模型 API 或本地构建；个别历史采样流程与参数仍需按对应版本检查 |
| `sci-fs/` 的旧特征排序与模型组合脚本 | 保留研究变体；新训练优先使用统一的特征选择入口，避免沿用全数据排序后的分数解释 |
| Notebook | 原始输出不重跑、不擦除；图表保留历史语境 |
| 缺失的原始数据与本地路径 | 仓库没有完整实验数据，无法据此确认历史 ACC/MCC 的独立复现结果 |

静态报告中的 `legacy_patterns` 给出遗留 NumPy 别名、`fit_sample`、`tf.contrib` 和全数据拟合候选位置。它们是需要人工结合上下文检查的线索，不是逐一确证的缺陷数量。

## 运行和验证

核心环境在 Python 3.13 上验证：NumPy 2.2.4、pandas 2.2.3、SciPy 1.15.2、scikit-learn 1.6.1、imbalanced-learn 0.13.0；神经网络验证使用 PyTorch 2.7.0 CPU。固定版本见根目录两份 requirements 文件，仅覆盖本次维护入口。

```bash
python -m pip install -r requirements.txt
# 需要运行 PyTorch 模型时：
python -m pip install -r requirements-torch.txt
python -m unittest discover -s tests -v
python scripts/review_archive.py
```

回归测试覆盖：SMOTE 实际接收的训练行、折内特征选择、OOF 行顺序和原始标签、概率与预测一致性、最小类别与邻居数校验、MCC 退化情况、兼容导入、SVR 目标保真、单样本网络输出、每折独立模型/优化器/归一化状态，以及 SNN/BiRNN 的合成数据训练和 CNN 矩阵读取。

数据检查与合成小样本训练验证的是运行正确性，不是研究数据集上的模型质量。GPU 路径未实测。

## 纪念相册

打开根目录 [index.html](../index.html) 即可浏览，无需联网、安装前端依赖或启动服务。包含：

- 按 Git 提交证据整理的五段时间线。
- 六个研究方向，可按类别筛选。
- 四张原始 Notebook 图像，可放大、翻页与用 Escape 关闭。
- 8 组历史配置、ACC/MCC 切换、941 步指标读取与当前曲线 SVG 导出。
- 桌面、手机布局，减少动态效果设置与打印样式。

图片直接提取自 Notebook，未用新生成的照片冒充研究经历。`assets/album/sources.json` 保存原图 SHA-256、单元格编号、日志路径和档案提交版本；`python scripts/build_album.py` 可重新生成材料。

浏览器检查使用本机 Chrome，通过 DevTools Protocol 访问本地文件，验证图片、交互和实际日志数值；在桌面、768px、390px、340px 宽度检查布局。脚本为 `scripts/check_album.mjs`，需要先启动本地调试端口 9229；截图和报告保存于被 Git 忽略的 `.preview/`。
