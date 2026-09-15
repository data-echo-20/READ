# -*- coding: utf-8 -*-
"""跨模态松弛熵正则对齐模块（论文第 3.6、3.7 节，式 5-9）。

对齐矩阵不含边际约束，目标函数逐点取极小，因此具有闭式解，
全部运算为可微操作，可直接端到端反向传播，无 Sinkhorn 迭代。

注意：由于不含边际约束，Gamma 的行和与列和均不等于预先给定的 w 与
nu_tilde，故本文称其为“对齐矩阵”而非“传输计划/耦合矩阵”。

数学约定：锚点不是 Dirac 硬点，而是带位置不确定度的 q_i(y) = N(y; mu_i,
sigma_i^2)（式 4）。对核函数取该分布下的期望，得到边际化核：核宽由 epsilon
放宽为 W_i = epsilon + 2 sigma_i^2，并附带一个逐锚点常数
chi_i = (1 + 2 sigma_i^2 / epsilon)^(-1/2)（式 6）。
sigma_i = 0 时 W_i = epsilon、chi_i = 1，严格退化为点锚点形式。
"""
import math

import torch
import torch.nn as nn

import config


class RelaxedAlignment(nn.Module):
    """将题跋锚点分布 {q_i} 与画面软证据 nu 在年代轴上对齐。

    输入:
        log_nu   : [B, L]     画面软证据的对数密度（式 3 在网格上的对数采样）
        sigma_nu : [B, 1]     画面软证据标准差，用于式 8 的核宽归一化
        anchors  : [B, K, 3]  每行 (mu_i, sigma_i, w_i)，即式 4 的锚点位置均值、
                              位置标准差与锚点权重。题跋缺失有两种等价表示：
                              K=0，或该样本的 w_i 全为 0（后者用于同批次中混有
                              无题跋样本的情形），二者均触发恒等似然
        grid     : [L]        年代网格点 s_l

    输出:
        p        : [B, L]     融合年代后验（式 7）
        SC       : [B]        时间自洽度（式 8），取值 (0, 1]
        Gamma    : [B, K, L]  对齐矩阵（式 6）
        rho_hat  : [B, K]     锚点归一化保留率（式 8），K=0 时为空
        d        : [B, K]     锚点冲突分数（式 9），K=0 时为空
        pi       : [B, K]     后验混合权重（定理 2 的式 21），供理论核验使用。
                              数值上 pi_i = rho_i / sum_j rho_j，即行边缘的归一化
    """

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, log_nu: torch.Tensor, sigma_nu: torch.Tensor,
                anchors: torch.Tensor, grid: torch.Tensor):
        B, L = log_nu.shape
        tiny = torch.finfo(log_nu.dtype).tiny
        eps = self.epsilon

        # 软证据在网格上归一化（式 7 的分母）：nu_tilde_l = nu(s_l) / sum_j nu(s_j)。
        # 注意 nu 是概率密度而非 logits，此处须用直接归一化；若误用 softmax，
        # 密度值所在的窄区间会使分布退化为近似均匀，破坏式 8 的保留率与定理 3 的信任界。
        # 入参为对数密度，全程不再取指数，故尾部不丢失精度。
        log_nu_tilde = log_nu - torch.logsumexp(log_nu, dim=-1, keepdim=True)

        mu_i = anchors[..., 0]                                   # [B, K]
        sigma_i = anchors[..., 1].clamp_min(0.0)                 # [B, K]
        w_i = anchors[..., 2]                                    # [B, K]
        K = mu_i.shape[1]

        # 题跋缺失（K = 0）：似然恒为常数 r_l ≡ 1，融合后验严格退化为软证据本身
        # （论文 3.7 节约定）。这是“恒等似然”，而非“均匀锚点”。
        if K == 0:
            p = torch.exp(log_nu_tilde)
            SC = torch.ones(B, device=log_nu.device)
            Gamma = log_nu.new_zeros(B, 0, L)
            rho_hat = log_nu.new_zeros(B, 0)
            d = log_nu.new_zeros(B, 0)
            pi = log_nu.new_zeros(B, 0)
            return p, SC, Gamma, rho_hat, d, pi

        # 逐样本的有效锚点指示量。K > 0 的张量里仍可能有样本没有题跋（同批次中
        # 混有无题跋样本），这类样本的权重被上游整体置零，此处据此判定并按
        # 恒等似然处理，与上面 K = 0 的分支语义完全一致：
        #   r_l ≡ 1、SC ≡ 1，后验退化为 nu_tilde。
        # 只把 w_i 置零并不足以做到这一点——那样 r_l 会恒为 0 而非 1。
        active = w_i.sum(dim=-1) > 1e-6                          # [B]
        missing = (~active).to(log_nu.dtype)                     # [B]

        # 边际化核的逐锚点带宽与常数（式 6）：W_i = epsilon + 2 sigma_i^2，
        # chi_i = (1 + 2 sigma_i^2 / epsilon)^(-1/2)。sigma_i 越大，核越平缓、
        # 常数越小，该锚点的整体影响力下降，这正是“可靠度是推导出来的”的来源。
        W_i = eps + 2.0 * sigma_i * sigma_i                      # [B, K]
        log_chi = -0.5 * torch.log1p(2.0 * sigma_i * sigma_i / eps)   # [B, K]

        # 二次时间代价的指数因子（式 6 的指数项）：exp(-(mu_i - s_l)^2 / W_i)
        diff = mu_i.unsqueeze(-1) - grid.unsqueeze(0).unsqueeze(0)    # [B, K, L]
        log_kernel = -(diff * diff) / W_i.unsqueeze(-1)               # [B, K, L]

        # 以下四式原本都形如“nu 的尾部 × 核的尾部”，必须在对数域计算。锚点与画面
        # 严重冲突时（例如画面指向 1000 年、题跋指向 2000 年），nu_tilde_l 在锚点
        # 附近可取到 e^{-500} 量级、exp(-(mu-s)^2/W) 在画面附近同样是 e^{-277} 量级，
        # 二者在整个网格上同时下溢为 0：式 7 的 p 退化为 0/0 得到 NaN，式 8 的
        # 保留率与式 9 的冲突分数则退化为 0，恰好丢失本文最关心的冲突信息。
        # 对数域计算可完整保留这些量，且不改变任何一式的数学取值。
        # 权重取对数前做下界截断，是为了避免 -inf 进入反向传播；题跋缺失样本的
        # w_i 全为 0 会得到 log(tiny) ≈ -87 的虚假权重，但该样本的 log r、rho_hat、
        # d、Gamma 随后一律被 active 掩码覆写为约定值，故不受影响。
        log_w = torch.log(w_i.clamp_min(tiny))                   # [B, K]
        # 不含权重的对数核矩阵，式 8 的分子直接由此得到，与 w_i 无关
        log_base = log_nu_tilde.unsqueeze(1) + log_kernel        # [B, K, L]
        # 松弛熵正则对齐的闭式解（式 6）取对数：
        # log Gamma[b,i,l] = log w_i + log nu_tilde_l + log chi_i - (mu_i - s_l)^2 / W_i
        log_Gamma = log_w.unsqueeze(-1) + log_chi.unsqueeze(-1) + log_base

        # 题跋似然核密度（式 7 的 r_l）：
        # log r_l = logsumexp_i(log w_i + log chi_i - (mu_i - s_l)^2 / W_i)。
        # 无题跋样本的权重和被置零，其 r_l 补为常数 1，即恒等似然。
        log_r = torch.logsumexp(
            log_w.unsqueeze(-1) + log_chi.unsqueeze(-1) + log_kernel, dim=1)   # [B, L]
        log_r = torch.where(active.unsqueeze(-1), log_r, torch.zeros_like(log_r))

        # 融合年代后验（式 7）：p_l = nu_tilde_l * r_l / sum_j nu_tilde_j r_j。
        # 列边缘 c_l = sum_i Gamma_il = nu_tilde_l * r_l，故取 softmax 即得 p。
        p = torch.softmax(log_nu_tilde + log_r, dim=-1)          # [B, L]

        # 核宽归一化常数（式 8 的 kappa_i）：kappa_i = chi_i * chi_W^(i)，其中
        # chi_W^(i) = (1 + 2 sigma_nu^2 / W_i)^(-1/2) 是画面侧的同型常数。
        sn = sigma_nu.reshape(B).clamp(min=1e-3)                 # [B]
        log_chi_W = -0.5 * torch.log1p(
            2.0 * (sn * sn).unsqueeze(-1) / W_i)                 # [B, K]
        log_kappa = log_chi + log_chi_W                          # [B, K]

        # 锚点归一化保留率（式 8）：rho_hat_i = rho_i / (w_i kappa_i)，其中行边缘
        # rho_i = w_i * kappa_i * rho_hat_i。两处 chi 因子在比值中相消，故
        # rho_hat_i = (1 / chi_W^(i)) * sum_l nu_tilde_l exp(-(mu_i - s_l)^2 / W_i)。
        # 由对数域直接给出，不经过 Gamma，故锚点与画面严重冲突时依然精确，
        # 其值趋近 0 而非 NaN。sigma_i = 0 时 chi_W^(i) 退回原式的 chi。
        log_rho_hat = torch.logsumexp(log_base, dim=-1) - log_chi_W    # [B, K]
        rho_hat = torch.exp(log_rho_hat)                                # [B, K]
        rho_hat = torch.where(active.unsqueeze(-1), rho_hat,
                              torch.zeros_like(rho_hat))

        # 时间自洽度（式 8）：SC = sum_i w_i * rho_hat_i，取值 (0, 1]。
        # 无题跋样本不存在可被否证的锚点，自洽度按约定取 1。
        SC = (w_i * rho_hat).sum(dim=-1) + missing                   # [B]

        # 锚点冲突分数（式 9）：d_i = w_i (1 - rho_hat_i)，sum_i d_i = 1 - SC。
        # 该量逐锚点给出，既可作交叉熵式正则的软标签，也直接支撑锚点级冲突定位：
        # 冲突锚点即 d_i 最大（等价于 rho_hat_i 最小）者。相较直接用 rho_hat 排序，
        # d_i 额外按权重折算，避免把“权重极小但恰巧不吻合”的锚点误判为冲突源。
        d = w_i * (1.0 - rho_hat)                                    # [B, K]
        d = torch.where(active.unsqueeze(-1), d, torch.zeros_like(d))

        # 后验混合权重（式 21）：pi_i = w_i kappa_i rho_hat_i / sum_j (...)。
        # 行边缘 rho_i = w_i kappa_i rho_hat_i，故 pi_i = rho_i / sum_j rho_j，
        # 直接由行边缘归一化得到，无需显式展开 kappa_i。该量供定理 2、3 的
        # 数值核验与后验分解诊断使用，不参与损失。
        rho = w_i * torch.exp(log_kappa) * rho_hat                   # [B, K]
        rho = torch.where(active.unsqueeze(-1), rho, torch.zeros_like(rho))
        denom = rho.sum(dim=-1, keepdim=True)
        safe = denom > tiny
        pi = torch.where(safe, rho / denom.clamp_min(tiny),
                         torch.full_like(rho, 1.0 / max(K, 1)))      # [B, K]

        # 对齐矩阵（式 6）。返回前按样本内的最大元素归一：exp(log Gamma) 在严重
        # 冲突时会整体下溢为 0，逐样本缩放可保持其用于冲突定位的可用性（行内
        # argmin、行形状、行间相对大小均不受影响）。该缩放不改变 p、SC、rho_hat
        # 与 d，仅改变显示的绝对尺度。
        shift = torch.where(active, log_Gamma.amax(dim=(1, 2)),
                            torch.zeros_like(SC))
        Gamma = torch.exp(log_Gamma - shift[:, None, None])          # [B, K, L]
        Gamma = torch.where(active[:, None, None], Gamma, torch.zeros_like(Gamma))

        return p, SC, Gamma, rho_hat, d, pi


# 兼容旧接口名
OptimalTransportFusion = RelaxedAlignment


def gaussian_log_evidence(mu_nu: torch.Tensor, sigma_nu: torch.Tensor,
                          grid: torch.Tensor):
    """画面软证据在网格上的对数采样（式 3）：log nu(s_l) = log N(s_l; mu_nu, sigma_nu^2)。

    返回对数密度而非密度本身。融合模块内部需要的是对数密度，而从密度回取对数
    是不可逆的：密度在尾部会于 float32 下下溢为 0，log(0) 或 log(clamp(0)) 会把
    这些格点错误地抬到 log(tiny) ≈ -87 或 -inf，使式 7 在锚点与画面严重冲突时
    选中错误的峰。此处直接在对数域计算，全程不经过指数，故无下溢损失。
    """
    g = grid.unsqueeze(0)                                          # [1, L]
    mu = mu_nu.unsqueeze(-1)                                       # [B, 1]
    sigma = sigma_nu.unsqueeze(-1).clamp(min=1e-3)                 # [B, 1]
    return (-0.5 * ((g - mu) / sigma) ** 2
            - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi))    # [B, L]


def gaussian_soft_evidence(mu_nu: torch.Tensor, sigma_nu: torch.Tensor, grid: torch.Tensor):
    """画面软证据在网格上的采样（式 3）：nu(t) = N(t; mu_nu, sigma_nu^2)。

    仅在需要真实密度值（如可视化）时使用；参与式 6-9 运算的一律取对数形式，
    见 gaussian_log_evidence。
    """
    return torch.exp(gaussian_log_evidence(mu_nu, sigma_nu, grid))     # [B, L]


def posterior_std(p: torch.Tensor, grid: torch.Tensor, y_hat: torch.Tensor):
    """后验标准差 sigma_p（式 15）：sigma_p^2 = sum_l p_l (s_l - y_hat)^2。

    该量是式 15 校准损失的分母，也是 5.11 节覆盖率与 CRPS 评测所需的
    不确定度输出。注意它是后验分布的二阶矩而非某个定理的界，直接由 p 定义。
    """
    return torch.sqrt(((p * (grid.unsqueeze(0) - y_hat.unsqueeze(-1)) ** 2)
                       .sum(dim=-1)).clamp_min(1e-12))             # [B]


def dual_task_head(p: torch.Tensor, grid: torch.Tensor, intervals):
    """双任务输出头（式 10、11）：连续年代估计 + 朝代分类概率。"""
    y_hat = (p * grid.unsqueeze(0)).sum(dim=-1)                    # [B]，式 10

    probs = []                                                     # 式 11
    for _, a, b in intervals:
        mask = (grid >= a) & (grid < b)
        probs.append((p * mask.float().unsqueeze(0)).sum(dim=-1))  # [B]
    p_class = torch.stack(probs, dim=-1)                           # [B, C]
    p_class = p_class / p_class.sum(dim=-1, keepdim=True)
    return y_hat, p_class
