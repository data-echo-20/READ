# -*- coding: utf-8 -*-
"""训练目标（论文第 3.9 节，式 12-16）。"""
import math

import torch
import torch.nn.functional as F

import config


def regression_loss(posterior: torch.Tensor, grid: torch.Tensor, year: torch.Tensor,
                    delta: float = 10.0):
    """式 12：L_reg = sum_l p_l * Huber(s_l - y)。"""
    diff = grid.unsqueeze(0) - year.unsqueeze(-1)                # [B, L]
    huber = F.smooth_l1_loss(diff, torch.zeros_like(diff),
                             reduction='none', beta=delta)        # [B, L]
    return (posterior * huber).sum(dim=-1).mean()


def classification_loss(p_class: torch.Tensor, dynasty: torch.Tensor):
    """式 13：L_cls = -log p(c*)。"""
    logp = torch.log(p_class + 1e-8)
    return F.nll_loss(logp, dynasty)


def consistency_loss(SC: torch.Tensor, is_consistent: torch.Tensor):
    """式 14：L_cons = (1 - SC)（自洽）或 max(0, SC - (1 - m))（冲突）。

    SC 为式 8 的自洽度，取值 (0, 1]，与年代尺度无关，因此间隔 m 无须
    随年代跨度调整。默认 m = 0.25，即冲突样本的目标自洽度不高于 0.75。
    """
    # is_consistent: 1 = 真实题（自洽），0 = 伪题（冲突）
    margin = 1.0 - config.CONS_MARGIN
    loss = torch.where(is_consistent.bool(), 1.0 - SC,
                       torch.clamp(SC - margin, min=0.0))
    return loss.mean()


def calibration_loss(y_hat: torch.Tensor, sigma_p: torch.Tensor, year: torch.Tensor):
    """式 15：L_cal = -log N(y; y_hat, sigma_p^2)。

    该项使后验标准差 sigma_p 成为可用的不确定度输出，从而支撑 5.11 节的预测
    区间覆盖率、CRPS 与期望校准误差。它与 L_reg 的分工是：L_reg 只关心后验的
    均值位置，对后验的展宽完全不敏感；L_cal 则同时惩罚均值偏差与宽度失配——
    区间过窄且偏差大时惩罚显著增大，区间过宽时 log(sigma_p) 项增大。若只有
    L_reg，后验会在训练中逐步锐化，其名义不确定度将系统性小于真实误差。

    下界截断取 1 年：年代网格步长约 6.69 年，任何真实后验的展宽都不会低于
    该量级，截断只为防止前向早期出现 sigma_p -> 0 导致的数值发散。
    """
    sp = sigma_p.clamp_min(1.0)
    return (0.5 * torch.log(2.0 * math.pi * sp * sp)
            + (y_hat - year) ** 2 / (2.0 * sp * sp)).mean()


def total_loss(out: dict, year: torch.Tensor, dynasty: torch.Tensor,
               is_consistent: torch.Tensor, grid: torch.Tensor):
    """式 16：四项损失加权和。"""
    l_reg = regression_loss(out['posterior'], grid, year)
    l_cls = classification_loss(out['p_class'], dynasty)
    l_cons = consistency_loss(out['consistency'], is_consistent)
    l_cal = calibration_loss(out['y_hat'], out['sigma_p'], year)
    loss = (config.LAMBDA_REG * l_reg + config.LAMBDA_CLS * l_cls
            + config.LAMBDA_CONS * l_cons + config.LAMBDA_CAL * l_cal)
    return loss, {'l_reg': l_reg.item(), 'l_cls': l_cls.item(),
                  'l_cons': l_cons.item(), 'l_cal': l_cal.item()}
