# -*- coding: utf-8 -*-
"""不确定度质量与概率校准评测（论文第 5.11、5.12 节）。

本文的输出不止一个点估计，还包括一个定义在年代网格上的后验分布 p、一个
标量自洽度 SC 与一组逐锚点的冲突分数 d。因此评测也必须覆盖这些量：
点估计的准确率无法说明后验分布是否可信，而"分布是否可信"恰恰是本文
相对纯点估计方法的主要主张之一。

本模块不依赖 sklearn，全部指标以 numpy 实现，便于随论文代码一起分发。
"""
import numpy as np


# --------------------------------------------------------------- 概率校准

def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               n_bins: int = 15):
    """期望校准误差 ECE 与可靠性图分箱。

    confidence: [N] 预测类别的最大概率；correct: [N] 该预测是否正确的 0/1 向量。
    返回 (ece, bin_conf, bin_acc, bin_cnt)，后三者用于绘制图 17 的可靠性图，
    空箱以 nan 填充，绘图时应跳过。

    ECE = sum_b (n_b / N) * |acc_b - conf_b|，即按样本数加权的各箱校准偏差。
    注意本文的后验分布是连续分布，其"置信度"取朝代分类头在若干朝代区间上
    积分所得的 p(c) 的最大值，与连续年代估计的区间覆盖率互为对照。
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    bin_cnt = np.zeros(n_bins, dtype=int)
    ece = 0.0
    n = len(confidence)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (confidence > lo) & (confidence <= hi) if b > 0 else (confidence <= hi)
        cnt = int(m.sum())
        bin_cnt[b] = cnt
        if cnt == 0:
            continue
        bin_conf[b] = confidence[m].mean()
        bin_acc[b] = correct[m].mean()
        ece += cnt / n * abs(bin_acc[b] - bin_conf[b])
    return ece, bin_conf, bin_acc, bin_cnt


def brier_score(probs: np.ndarray, target: np.ndarray, num_classes: int):
    """多类 Brier 分数：Brier = (1/N) sum_n sum_c (p_nc - y_nc)^2。

    该指标同时惩罚"分错类"与"过度自信"，取值范围 [0, 2]（多类情形），
    越低越好。与准确率相比，它对概率本身的质量更敏感。
    """
    probs = np.asarray(probs, dtype=float)
    n = probs.shape[0]
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), np.asarray(target, dtype=int)] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


# ----------------------------------------------------- 区间覆盖率与 CRPS

def _interval_from_grid(p: np.ndarray, grid: np.ndarray, level: float):
    """由网格后验的 CDF 给出中心预测区间 [a, b]。

    对离散分布，分位点须按阶梯 CDF 定义取 Q(q) = min{ s_l : F(s_l) >= q }，
    而不能在阶梯的跳变段上做线性插值：后者会把区间端点错误地拉到跳变段的
    内部，使一个本应宽度为零的尖峰后验得到一个正宽度且不含峰位的区间。
    这里用 searchsorted 直接取阶梯反函数，与年代网格的离散性一致。
    """
    cdf = np.cumsum(p, axis=1)
    cdf = cdf / cdf[:, -1:]
    lo_q, hi_q = 0.5 * (1.0 - level), 1.0 - 0.5 * (1.0 - level)
    last = len(grid) - 1
    lo = np.empty(p.shape[0])
    hi = np.empty(p.shape[0])
    for i in range(p.shape[0]):
        # 上界截断只是防御性的：cdf[:, -1] 已归一为 1，分位点不会越界
        lo[i] = grid[min(np.searchsorted(cdf[i], lo_q, side='left'), last)]
        hi[i] = grid[min(np.searchsorted(cdf[i], hi_q, side='left'), last)]
    return np.stack([lo, hi], axis=1)


def coverage(p: np.ndarray, grid: np.ndarray, year: np.ndarray,
             levels=(0.8, 0.9)):
    """名义水平的预测区间覆盖率（论文 5.11 节）。

    返回 dict，键为 'cov80'、'cov90'、以及平均区间宽度 'width80'、'width90'。
    覆盖率低于名义水平说明后验过窄、存在过度自信；显著高于名义水平说明
    后验过宽、不确定度被高估。两者都不利于把 sigma_p 当作可用的不确定度输出，
    这也是 3.9 节引入式 (15) 校准损失的直接动机。
    """
    out = {}
    for lv in levels:
        iv = _interval_from_grid(np.asarray(p, dtype=float),
                                 np.asarray(grid, dtype=float), lv)
        y = np.asarray(year, dtype=float)
        out['cov%d' % int(round(lv * 100))] = float(
            ((y >= iv[:, 0]) & (y <= iv[:, 1])).mean())
        out['width%d' % int(round(lv * 100))] = float((iv[:, 1] - iv[:, 0]).mean())
    return out


def crps(p: np.ndarray, grid: np.ndarray, year: np.ndarray):
    """连续排序概率分数 CRPS（离散预测分布的能量形式）。

    CRPS = E|S - y| - 0.5 E|S - S'|，其中 S、S' 独立同分布于网格后验 p。
    第一项是绝对误差的能量惩罚，第二项是预测分布自身的展宽奖励。CRPS 的单位
    与年代相同，可直接与 MAE 比较：CRPS 明显小于 MAE 说明分布的形状（而不只是
    其均值）携带了有效信息；两者接近则说明后验的宽度未能提供额外的分辨能力。
    """
    p = np.asarray(p, dtype=float)
    g = np.asarray(grid, dtype=float)
    y = np.asarray(year, dtype=float)
    term1 = (p * np.abs(g[None, :] - y[:, None])).sum(axis=1)
    M = np.abs(g[:, None] - g[None, :])
    term2 = 0.5 * np.einsum('nl,lm,nm->n', p, M, p)
    return float((term1 - term2).mean())


# ------------------------------------------- 检测与冲突定位（论文 5.12 节）

def _rank_curve(scores: np.ndarray, labels: np.ndarray):
    """按分数降序给出 ROC 与 PR 曲线的坐标。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-scores, kind='mergesort')
    s, y = scores[order], labels[order]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # 并列分数必须同进同出，否则曲线会被人为抬高
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[distinct].astype(float)
    fp = (distinct + 1 - tp)
    tpr = tp / n_pos
    fpr = fp / n_neg
    prec = tp / (tp + fp)
    rec = tpr
    return fpr, tpr, prec, rec, n_pos, n_neg


def auroc(scores: np.ndarray, labels: np.ndarray):
    """ROC 曲线下面积（Mann-Whitney 形式）。"""
    r = _rank_curve(scores, labels)
    if r is None:
        return float('nan')
    fpr, tpr = r[0], r[1]
    return float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))


def auprc(scores: np.ndarray, labels: np.ndarray):
    """PR 曲线下面积（平均精度，average precision）。

    在正负样本极不平衡时，AUPRC 比 AUROC 更能反映实际可用性，故本文两个
    都报告（表 13）。
    """
    r = _rank_curve(scores, labels)
    if r is None:
        return float('nan')
    prec, rec = r[2], r[3]
    mrec = np.r_[0.0, rec]
    mpre = np.r_[1.0, prec]
    return float((np.diff(mrec) * mpre[1:]).sum())


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, tpr_target: float = 0.95):
    """在召回率固定为 tpr_target 时的假正率 FPR@95TPR。

    该指标回答的是"若要求检出 95% 的伪题跋，会有多少真实题跋被误判"，
    比单看 AUROC 更贴近鉴定流程的实际代价。
    """
    r = _rank_curve(scores, labels)
    if r is None:
        return float('nan')
    fpr, tpr = r[0], r[1]
    if tpr.max() < tpr_target:
        return float('nan')          # 该阈值不可达，不做外推估计
    idx = int(np.searchsorted(tpr, tpr_target, side='left'))
    if idx == 0:
        return float(fpr[0])
    x0, x1 = tpr[idx - 1], tpr[idx]
    y0, y1 = fpr[idx - 1], fpr[idx]
    if x1 == x0:
        return float(y1)
    return float(y0 + (tpr_target - x0) * (y1 - y0) / (x1 - x0))


def detection_prf(scores: np.ndarray, labels: np.ndarray, threshold: float):
    """给定阈值下的精确率、召回率与 F1（用于冲突定位的锚点级评测）。"""
    pred = np.asarray(scores, dtype=float) >= threshold
    y = np.asarray(labels, dtype=int).astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


# ------------------------------------------------------------- 多重比较

def holm(pvals):
    """Holm 逐步向下校正，返回与输入同序的校正后 p 值。

    与 Bonferroni 相比，Holm 在保持族错误率控制的同时更不易过度保守，
    适合本文这样"4 个基线 × 3 个指标"量级的比较族（见 5.8 节与附录 B.4）。
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)      # 保证校正后 p 值单调不减
        adj[i] = running
    return adj


def summarize(y_hat, sigma_p, p, grid, year, p_class, dynasty, num_classes):
    """一次性汇总 5.11 节表 12 所需的全部概率质量与校准指标。"""
    p = np.asarray(p, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    year = np.asarray(year, dtype=float)
    sigma_p = np.asarray(sigma_p, dtype=float).clip(min=1.0)
    pred = np.asarray(p_class, dtype=float).argmax(axis=1)
    correct = (pred == np.asarray(dynasty, dtype=int)).astype(float)
    conf = np.asarray(p_class, dtype=float).max(axis=1)
    ece, _, _, _ = expected_calibration_error(conf, correct)
    out = {
        'nll': float((0.5 * np.log(2 * np.pi * sigma_p ** 2)
                      + (y_hat - year) ** 2 / (2 * sigma_p ** 2)).mean()),
        'crps': crps(p, grid, year),
        'mae': float(np.abs(y_hat - year).mean()),
        'ece': float(ece),
        'brier': brier_score(p_class, dynasty, num_classes),
    }
    out.update(coverage(p, grid, year))
    return out
