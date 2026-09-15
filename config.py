# -*- coding: utf-8 -*-
"""全局配置：与论文第 5 章表 4 超参数一一对应。"""

# ---- 时间网格（式 6、7 的 s_l）----
T_MIN, T_MAX = 618, 1949          # 年代范围 [t_min, t_max]，覆盖唐至近现代
L = 200                            # 网格点数，网格步长 (T_MAX-T_MIN)/(L-1) ≈ 6.7 年
GRID = None                        # 运行期由 torch.linspace(T_MIN, T_MAX, L) 填充
GRID_STEP = (T_MAX - T_MIN) / (L - 1)   # 网格步长 Δs ≈ 6.69 年；同时是 sigma_nu
                                        # 的下界：σ_ν 小于 Δs 时密度在网格上退化为
                                        # 单点冲激，式 7 的归一化将除以零。

# ---- 松弛熵正则对齐（式 5、6、8）----
SIGMA_BASE = 30.0                  # 画面标称不确定度 sigma_base
EPSILON = 4.0 * SIGMA_BASE ** 2    # 温度系数 epsilon = 4 * sigma_base^2 = 3600
                                   # 由论文 4.6 节命题 1 的准则确定：此时信任系数
                                   # tau = 1/3，且权重 >= 0.5 的跨代伪锚点可检出。
                                   # 注：式 (6) 的 W_i = epsilon + 2 sigma_i^2 中的
                                   # 长度单位为「年」，故 epsilon 亦以年^2 计。

# ---- 锚点宽度回归头（式 4 的 sigma_i）----
SIGMA_MIN = 1.0                    # sigma_i 的正下界，单位 年。见 encoders.py 的
                                   # 说明：sigma_i 精确为 0 会使核宽退回 epsilon，
                                   # 冲突分数 d_i 的分辨率随之下降。
ANCHOR_PRIOR_DIM = 0               # 规则先验特征 u_i 的维度（实体类型、识别置信度、
                                   # 纪年歧义度、上下文完整性四项）。0 表示当前
                                   # 实现不注入先验特征，仅用题跋表示回归 sigma_i。

# ---- 自洽度正则（式 14）----
CONS_MARGIN = 0.25                 # 间隔超参数 m，冲突样本目标 SC <= 1 - m = 0.75
                                   # 注意：SC 为式 8 的无量纲自洽度，取值 (0, 1]，
                                   # 与年代尺度无关，故 m 无须随年代跨度调整。

# ---- 朝代区间（式 11，朝代分类概率的积分区间）----
DYNASTY_INTERVALS = [
    ("唐", 618, 907),
    ("五代", 907, 960),
    ("宋", 960, 1279),
    ("元", 1279, 1368),
    ("明", 1368, 1644),
    ("清", 1644, 1912),
    ("近现代", 1912, 1949),
]
NUM_CLASSES = len(DYNASTY_INTERVALS)

# ---- 视觉编码器（ViT-Base）----
IMG_SIZE = 224
PATCH_SIZE = 16
VIT_DIM = 768
VIT_HEADS = 12
VIT_DEPTH = 12

# ---- 题跋编码器（SikuBERT）----
TEXT_MODEL = 'SIKU-BERT/sikubert'   # HuggingFace 权重 id（王东波等，2022）；
                                    # 也可填本地权重目录以离线加载，分词器与
                                    # 编码器必须指向同一路径。
TEXT_DIM = 768                      # SikuBERT 隐层维度，与 VIT_DIM 对齐以便融合
MAX_TEXT_LEN = 128                  # 题跋截断长度 N

# ---- 训练（式 16 的权重 + 优化器）----
LAMBDA_REG = 1.0
LAMBDA_CLS = 1.0
LAMBDA_CONS = 0.1
LAMBDA_CAL = 0.1       # 式 15 的校准损失权重。取值与 LAMBDA_CONS 同量级：
                       # 该校准项的作用是让后验标准差 sigma_p 成为可用的
                       # 不确定度输出（支撑 5.11 节的覆盖率与 CRPS），而非
                       # 提升点估计精度，权重过大会以 MAE 为代价换取过窄区间。

LR = 1e-4
WEIGHT_DECAY = 5e-2
ADAM_BETAS = (0.9, 0.999)   # AdamW 的动量系数（表 4）
BATCH_SIZE = 32
EPOCHS = 100
WARMUP_EPOCHS = 5
EARLY_STOP_PATIENCE = 10
SEED = 42

# ---- 数据增广 ----
DROP_TEXT_PROB = 0.15    # 15% 概率丢弃题跋模态（K = 0，恒等似然）
PSEUDO_PROB = 0.10       # 10% 概率以同数据集内跨朝代采样的伪题替换真题
                         # 注：不使用“加常数偏移”，以免模型通过分布偏移而非
                         # 时间不一致来识别伪题（论文 5.3 节）。

# ---- 数据集划分 ----
SPLIT = (0.7, 0.15, 0.15)   # 训练/验证/测试 = 7:1.5:1.5

# ---- 锚点级冲突定位的评测协议（式 9、5.12 节）----
CONFLICT_TOL = 60.0          # 锚点位置与样本真实年代相差超过该值即判为冲突锚点，
                             # 单位 年。取值与定理 4 在 sigma_i = sigma_base 时的
                             # 最小可检偏移 delta_min ≈ 70.6 年同量级，略宽以避免
                             # 把恰好落在下界附近的锚点标为冲突。
