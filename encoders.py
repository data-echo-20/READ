# -*- coding: utf-8 -*-
"""编码器：画面风格时间分布估计（第 3.4 节）与题跋硬时间锚点提取（第 3.5 节）。"""
import torch
import torch.nn as nn

import config


class VisualEncoder(nn.Module):
    """ViT-Base 视觉编码器，输出风格时间分布的均值与标准差（式 3、4）。

    默认使用 torchvision 的 vit_b_16；也可替换为任意输出 768 维特征的骨干网络。
    """

    def __init__(self, vit_dim: int = config.VIT_DIM, pretrained: bool = False):
        super().__init__()
        from torchvision import models
        weights = (models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = models.vit_b_16(weights=weights)
        self.backbone.heads.head = nn.Identity()     # 取 768 维 [CLS] 特征
        self.mu_head = nn.Linear(vit_dim, 1)         # 式 3
        self.sigma_head = nn.Linear(vit_dim, 1)      # 式 4

    def forward(self, image: torch.Tensor):
        h_v = self.backbone(image)                   # [B, 768] 风格特征
        # 式 3：经 sigmoid 映射到年代区间 [t_min, t_max]，与锚点提取头的年代映射
        # 保持一致。若不加此约束，mu_nu 可落在年代网格之外，nu(t) 在网格上整体
        # 下溢为 0，式 7 的归一化会得到 0/0，前向传播在训练初期即产生 NaN。
        mu_raw = torch.sigmoid(self.mu_head(h_v))    # [B, 1]
        mu_nu = config.T_MIN + (config.T_MAX - config.T_MIN) * mu_raw
        # 式 4：softplus 保证标准差为正，下界取网格步长 Δs。ν(t) 是以 Δs 离散
        # 表示的，σ_ν < Δs 时密度在网格上退化为单点冲激，同样会使式 7 除以零。
        sigma_nu = config.GRID_STEP + torch.nn.functional.softplus(self.sigma_head(h_v))
        return mu_nu.squeeze(-1), sigma_nu.squeeze(-1)


class TextEncoder(nn.Module):
    """题跋编码器：古汉语预训练语言模型（第 3.5 节）。

    默认加载 SikuBERT，即在《四库全书》语料上继续预训练的古汉语语言模型
    （王东波, 刘畅, 朱子赫, 等. SikuBERT 与 SikuRoBERTa：面向数字人文的
    《四库全书》预训练模型构建及应用研究[J]. 图书馆论坛, 2022, 42(6): 31-43），
    取最后一层隐状态做掩码平均池化，得到题跋整体表示。

    分词由 data.build_tokenizer() 提供，二者须指向同一权重目录，否则词表
    id 与嵌入矩阵错位。freeze=True 时冻结语言模型、只训练下游头。
    """

    def __init__(self, name_or_path: str = config.TEXT_MODEL,
                 freeze: bool = False):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        self.cfg = AutoConfig.from_pretrained(name_or_path)
        self.encoder = AutoModel.from_pretrained(name_or_path)
        self.text_dim = self.cfg.hidden_size
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # input_ids: [B, N]，attention_mask: [B, N]
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                       # [B, N, d_t]
        # 掩码平均池化得到整体题跋特征，掩码为 0 的位置不参与分母
        mask = attention_mask.unsqueeze(-1).float()     # [B, N, 1]
        h_t = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return h_t                                      # [B, d_t]


class AnchorExtractor(nn.Module):
    """时间锚点提取头（第 3.5 节，式 4）。

    将题跋表示映射为一组 (mu_i, sigma_i, w_i) 的带宽度时间锚点，即式 (4) 的
    高斯分布 q_i(y) = N(y; mu_i, sigma_i^2)。每个锚点对应题跋中的一处纪年
    实体——年号、干支、生卒年、避讳与印鉴款识——h_i 为该实体的表示，
    u_i 为随之给出的规则先验特征。

    位置均值 mu_i 由 sigmoid 映射到 [t_min, t_max]；位置不确定度 sigma_i 由
    sigma_min + softplus(w_s^T [h_i; u_i] + b_s) 给出，其中 u_i 为规则先验特征
    （实体类型、识别置信度、纪年歧义度、上下文完整性）。config.ANCHOR_PRIOR_DIM
    为 0 时 u_i 不参与拼接，该头仅由题跋表示回归 sigma_i。
    sigma_min > 0 是必须的：sigma_i = 0 会使核宽退化为 epsilon 本身并使 1 - rho_hat
    恒为 0，冲突分数失去分辨率。权重 w_i 由 softmax 归一化，满足 sum_i w_i = 1。
    """

    def __init__(self, text_dim: int = config.TEXT_DIM, max_anchors: int = 8,
                 prior_dim: int = 0):
        super().__init__()
        self.max_anchors = max_anchors
        self.prior_dim = prior_dim
        self.t_head = nn.Linear(text_dim + prior_dim, max_anchors)   # 锚点位置 mu_i
        self.s_head = nn.Linear(text_dim + prior_dim, max_anchors)   # 锚点宽度 logits
        self.w_head = nn.Linear(text_dim + prior_dim, max_anchors)   # 锚点权重 logits

    def forward(self, h_t: torch.Tensor, u: torch.Tensor = None):
        # h_t: [B, d_t] 题跋整体表示；u: [B, prior_dim] 规则先验向量，可缺省。
        # 三个头各自把上下文映射为 K 个逐锚点输出，即第 i 个锚点对应第 i 个
        # 线性泛函的输出。生产环境中 h_i 为第 i 个实体的表示，此时只需把下面
        # 的 h_t 换成逐实体表示 h_i 即可，公式形式不变。
        if self.prior_dim > 0 and u is not None:
            h_t = torch.cat([h_t, u], dim=-1)                   # [B, d_t + d_u]

        # 式 (4) 的 mu_i：将未归一化年代映射到 [t_min, t_max]，与画面分支的
        # mu_nu 用同一映射，保证锚点与软证据处在同一坐标域内。
        t_raw = torch.sigmoid(self.t_head(h_t))                 # [B, K]
        mu_i = config.T_MIN + (config.T_MAX - config.T_MIN) * t_raw

        # 式 (4) 的 sigma_i：softplus 保证正的增量，再加正下界 sigma_min。
        # 下界不可省略——sigma_i 精确为 0 会使边际化核退回点锚点形式，
        # rho_hat 的上界语义被破坏。
        sigma_i = config.SIGMA_MIN + nn.functional.softplus(self.s_head(h_t))

        # 锚点权重：softmax 归一化，sum_i w_i = 1，使 SC = sum_i w_i rho_hat_i
        # 成为可靠度加权平均（论文 3.7 节第 ⑤ 条）。
        w_i = torch.softmax(self.w_head(h_t), dim=-1)           # [B, K]

        anchors = torch.stack([mu_i, sigma_i, w_i], dim=-1)     # [B, K, 3]
        return anchors
