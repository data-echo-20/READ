# READ

**R**elaxed **E**ntropy-regularized **A**lignment for **D**ating

融合画面与题跋信息的传统书画年代智能辨识方法。

书画断代长期依赖专家对风格、款识与著录的综合判断。画面的时代风格是连续的软证据，只能给出一个
模糊的年代区间；题跋中的年号、干支、生卒与印鉴则是离散的硬锚点，指向具体的年份。两类证据的量纲
与可靠性都不一致，直接按固定权重加权或按最近邻匹配，都会在证据冲突时给出误导性的结果。

本仓库给出一种把二者放进同一个优化问题的方法：将题跋的年代主张表示为带位置不确定度的高斯锚点，
与画面的软证据分布做**松弛熵正则对齐**。不加边际约束使该问题具有闭式解，全部运算可微，可以直接
端到端反向传播，不需要 Sinkhorn 迭代。对齐过程同时导出三个可直接使用的量：年代后验分布、度量
画面与题跋一致程度的**自洽度**，以及逐锚点的**冲突分数**——后者可用于定位题跋中哪一处纪年与画面相悖。

## 方法概览

数据流向为：画面经 ViT-Base 编码为年代轴上的高斯软证据 $\nu(t)=\mathcal{N}(t;\mu_\nu,\sigma_\nu^2)$；
题跋经 SikuBERT 编码后由锚点提取头输出 $K$ 组 $(\mu_i,\sigma_i,w_i)$，即式 (4) 的高斯锚点
$q_i(y)=\mathcal{N}(y;\mu_i,\sigma_i^2)$；二者在年代网格 $\{s_l\}$ 上做松弛熵正则对齐。

对核函数取 $q_i$ 下的期望得到**边际化核**，核宽由 $\epsilon$ 放宽为

$$W_i=\epsilon+2\sigma_i^2,\qquad \chi_i=\left(1+\frac{2\sigma_i^2}{\epsilon}\right)^{-1/2}$$

$\sigma_i=0$ 时 $W_i=\epsilon$、$\chi_i=1$，退化为点锚点形式。这意味着锚点的可靠度是从其位置不确定度
**推导**出来的：$\sigma_i$ 越大，该锚点在核宽上被自动降权，无需额外乘一个可靠度系数。印鉴款识这类
纪年模糊的实体因此不会与明确的年号等权参与融合。

由于目标函数不含边际约束、逐点取极小，对齐矩阵有闭式解，整个融合模块没有迭代求解过程。

## 仓库结构

```
config.py                          全部超参数，与论文第 5 章表 4 一一对应
data.py                            数据集加载、题跋分词与增广
encoders.py                        画面编码器、题跋编码器、锚点提取头
optimal_transport.py               松弛熵正则对齐（闭式解、对数域实现）
model.py                           完整模型的前向传播
losses.py                          式 (14)(15)(16) 的损失项
calibration.py                     概率校准、检测指标与冲突定位指标
train.py                           训练入口
evaluate.py                        评测入口
smoke_test.py                      端到端冒烟测试
verify_generalized_theorems.py     定理 1–4 的独立数值复核
docs/hyperparameters.md            论文全部超参数，含 config.py 符号对照
docs/appendix-A-dataset.md         论文附录 A：数据集构建
docs/appendix-B-reproducibility.md 论文附录 B：复现性清单
```

## 环境

```
Python >= 3.9
torch >= 2.0
torchvision >= 0.15
transformers >= 4.30
numpy
Pillow
```

```bash
pip install -r requirements.txt
```

题跋编码器默认从 HuggingFace 加载 [`SIKU-BERT/sikubert`](https://huggingface.co/SIKU-BERT/sikubert)。
网络不通时可用镜像，或先把权重下到本地再传 `--text-model /path/to/sikubert`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 数据准备

仓库不附带数据集。数据根目录下放置图像目录与一份 JSON 清单：

```
<data-root>/
  images/
    0001.jpg
    ...
  inscriptions.json
```

`inscriptions.json` 为对象数组，每项字段如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 作品编号，用于复现划分与报错定位 |
| `image` | str | 相对 `<data-root>` 的图像路径 |
| `year` | float | 作品年代，单位「年」，须落在 $[618,\ 1949]$ 内 |
| `inscription` | str | 著录题跋全文；无题跋的作品置为空串 |
| `split` | str | `train` / `val` / `test` 之一，缺省视为 `train` |

```json
[
  {"id": "sh-0001", "image": "images/0001.jpg", "year": 1086.0,
   "inscription": "元祐元年春正月題", "split": "train"},
  {"id": "sh-0002", "image": "images/0002.jpg", "year": 1620.0,
   "inscription": "", "split": "test"}
]
```

年代标注协议与划分方式见 [附录 A](docs/appendix-A-dataset.md)。清单的构建方式不限，只要字段与
划分齐全即可。

## 训练与评测

```bash
python train.py --data /path/to/data --out best_model.pt

python evaluate.py --data /path/to/data --checkpoint best_model.pt --localization
```

`train.py` 的常用开关：

| 开关 | 默认 | 说明 |
|------|------|------|
| `--data` | 必填 | 数据根目录 |
| `--out` | `best_model.pt` | 最优权重的保存路径，按验证集年代误差择优 |
| `--epochs` | 100 | 最大训练轮数 |
| `--batch-size` | 32 | 批量大小 |
| `--workers` | 4 | DataLoader 工作进程数 |
| `--text-model` | `SIKU-BERT/sikubert` | SikuBERT 的 HF id 或本地权重目录 |
| `--freeze-text` | 关 | 冻结题跋编码器，只训练下游头 |

`evaluate.py` 报告连续年代 MAE、朝代准确率、宏平均 F1、自洽度 AUC，以及 NLL、CRPS、预测区间覆盖率、
ECE、Brier 等概率质量指标；加 `--localization` 另报锚点级冲突定位的准确率与 P/R/F1。

训练集默认开启 5.3 节的两项题跋增广：15% 概率丢弃题跋（对应零条锚点与恒等似然），10% 概率替换为
**另一朝代作品的真实题跋**并标记为冲突样本。两项增广都只作用在训练集上。伪题刻意不使用「对真实
年代加常数偏移」的构造方式，理由见 [超参数文档第 4 节](docs/hyperparameters.md)。

## 自检

```bash
python verify_generalized_theorems.py   # 定理 1–4 的数值复核，输出全为 ASCII
python smoke_test.py                    # 前向/反向有限性、题跋缺失退化
```

`verify_generalized_theorems.py` 逐条核验四条闭式结论，并检查 $\sigma_i=0$ 时严格复现点锚点形式的
取值（$\chi=0.8165$、$\alpha=2/3$、$\tau=1/3$）。`smoke_test.py` 检查混合批次（含有题跋与无题跋样本）
的前向传播、损失与梯度有限性，以及无题跋样本是否满足 $p=\tilde{\nu}$、$\mathrm{SC}=1$。

## 超参数

论文的全部超参数见 [`docs/hyperparameters.md`](docs/hyperparameters.md)，其中每一行都对应 `config.py` 中的一个符号。
配置只有这一份，代码中不再有第二处副本。改超参数只改 `config.py`。

## 论文

方法、理论分析与实验的完整论述见论文《融合画面与题跋信息的传统书画年代智能辨识方法》：

- [附录 A：CIP-Dating 数据集构建流程](docs/appendix-A-dataset.md)
- [附录 B：复现性清单](docs/appendix-B-reproducibility.md)

## 许可

[MIT](LICENSE)
