# -*- coding: utf-8 -*-
"""完整模型：画面软证据 → 题跋硬锚点 → 松弛熵正则对齐融合 → 双任务输出。

对应论文第 3.3 节总体框架与算法 1 的前向传播。
"""
import torch
import torch.nn as nn

import config
from encoders import VisualEncoder, TextEncoder, AnchorExtractor
from optimal_transport import (RelaxedAlignment, gaussian_log_evidence,
                               posterior_std, dual_task_head)


class CIPDatingModel(nn.Module):
    def __init__(self, grid: torch.Tensor, epsilon: float = config.EPSILON):
        super().__init__()
        self.register_buffer('grid', grid)          # [L]
        self.visual = VisualEncoder()
        self.text = TextEncoder()
        # 锚点头的输入维度跟随题跋编码器的实际隐层维度，避免换用其他
        # 预训练语言模型时维度错配
        self.anchor = AnchorExtractor(text_dim=self.text.text_dim,
                                      prior_dim=config.ANCHOR_PRIOR_DIM)
        self.fusion = RelaxedAlignment(epsilon)
        self.intervals = config.DYNASTY_INTERVALS

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, anchor_prior: torch.Tensor = None):
        B = image.shape[0]

        # 画面软证据（式 2、3）。以对数密度而非密度参与后续运算：密度在尾部
        # 于 float32 下下溢为 0，回取对数会把这些格点错误抬高，破坏式 6-9。
        mu_nu, sigma_nu = self.visual(image)        # [B], [B]
        log_nu = gaussian_log_evidence(mu_nu, sigma_nu, self.grid)   # [B, L]

        # 题跋证据（式 4）：一组带位置不确定度的锚点 (mu_i, sigma_i, w_i)
        has_text = attention_mask.sum(dim=-1) > 0   # [B]
        h_t = self.text(input_ids, attention_mask)  # [B, 768]
        anchors = self.anchor(h_t, anchor_prior)    # [B, K, 3]

        # 题跋缺失的样本把锚点权重整体置零，融合模块据权重之和判定有效锚点数
        # 为零，从而启用恒等似然 r_l ≡ 1（论文 3.7 节）。这里置零的是权重而非
        # 裁剪 K 维，因此同一批次中有无题跋的样本可以共用同一个张量。
        # 需要强调的是，题跋缺失对应的是“零条锚点”而非“一条位于中点的锚点”：
        # 任何有限的锚点都携带某个具体的年份倾向，无论权重多小都会使后验偏移，
        # 只有恒等似然才等价于题跋未提供时间信息。
        anchors = anchors.clone()
        anchors[..., 2] = anchors[..., 2] * has_text.to(anchors.dtype).unsqueeze(-1)

        # 跨模态松弛熵正则对齐融合（式 5-9）。第三个返回值为对齐矩阵 Gamma，
        # 第四、五个分别为锚点归一化保留率 rho_hat 与冲突分数 d，第六个为
        # 后验混合权重 pi。
        p, SC, Gamma, rho_hat, d, pi = self.fusion(
            log_nu, sigma_nu.unsqueeze(-1), anchors, self.grid)

        # 双任务输出（式 10、11）
        y_hat, p_class = dual_task_head(p, self.grid, self.intervals)
        sigma_p = posterior_std(p, self.grid, y_hat)     # 式 15 的分母

        return {'y_hat': y_hat, 'p_class': p_class, 'posterior': p,
                'consistency': SC,            # 式 8 的自洽度，取值 (0, 1]
                'sigma_p': sigma_p,           # 式 15 的后验标准差，单位 年
                'Gamma': Gamma,               # 对齐矩阵，供冲突定位可视化
                'retention': rho_hat,         # 式 8 的锚点归一化保留率
                'conflict': d,                # 式 9 的锚点冲突分数，逐锚点
                'pi': pi,                     # 式 21 的后验混合权重
                'anchors': anchors, 'log_nu': log_nu,     # 对数密度，尾部无下溢
                'nu': torch.exp(log_nu),                  # 密度，仅供可视化
                'mu_nu': mu_nu, 'sigma_nu': sigma_nu}
