# -*- coding: utf-8 -*-
"""端到端冒烟测试：前向/反向有限性、题跋缺失退化、与独立复核脚本的一致性。

运行：python smoke_test.py
对应论文 3.7 节的约定（K = 0 的恒等似然）与式 (6)-(9) 的实现。

所有输出为 ASCII，避免在 GBK 控制台出现乱码。
"""
import numpy as np
import torch

import config
from model import CIPDatingModel
from optimal_transport import RelaxedAlignment, gaussian_log_evidence
from losses import total_loss

torch.manual_seed(0)
np.random.seed(0)

GRID = torch.linspace(config.T_MIN, config.T_MAX, config.L)
EPS = config.EPSILON


def hdr(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# ------------------------------------------------------------ 1. 模块级核验
hdr("[1] RelaxedAlignment vs. the independent closed-form checker")

fusion = RelaxedAlignment(EPS)


def module_pass(mus, sigs, ws, mu_nu, sigma_nu):
    """Run the module for one sample and return the diagnostics of interest."""
    log_nu = gaussian_log_evidence(
        torch.tensor([mu_nu]), torch.tensor([sigma_nu]), GRID)
    # anchors: [1, K, 3] with columns (mu_i, sigma_i, w_i)
    arr = np.stack([np.atleast_1d(mus), np.atleast_1d(sigs),
                    np.atleast_1d(ws)], axis=-1)
    anchors = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    p, SC, Gamma, rho_hat, d, pi = fusion(log_nu, torch.tensor([[sigma_nu]]),
                                          anchors, GRID)
    return dict(p=p[0], SC=float(SC[0]), rho_hat=rho_hat[0], d=d[0], pi=pi[0])


SN = 30.0


def W_of(s):
    return EPS + 2.0 * s ** 2


def chi_of(s):
    return (1.0 + 2.0 * s ** 2 / EPS) ** -0.5


def chi_W(s):
    return (1.0 + 2.0 * SN ** 2 / W_of(s)) ** -0.5


# 闭式解（式 26）：rho_hat_i = exp(-(mu_i - mu_nu)^2 / (2 sigma_nu^2 + W_i))
print("  %-8s %-22s %-22s %-12s" % ("sigma", "rho_hat module", "rho_hat closed", "abs diff"))
for sig in [0.0, 15.0, 40.0]:
    R = module_pass([1200.0], [sig], [1.0], 1000.0, SN)
    closed = float(np.exp(-(1200.0 - 1000.0) ** 2 / (2 * SN ** 2 + W_of(sig))))
    got = float(R['rho_hat'][0])
    print("  %-8.1f %-22.15f %-22.15f %-12.2e" % (sig, got, closed, abs(got - closed)))
    assert abs(got - closed) < 1e-5, "rho_hat closed form mismatch at sigma=%s" % sig

# 单锚点方差比等于 alpha_i（定理 1 的推广，式 19）
print("\n  single anchor variance ratio (must equal alpha_i):")
print("  %-8s %-18s %-18s %-12s" % ("sigma", "var/sigma_nu^2", "alpha_i", "abs diff"))
for sig in [0.0, 15.0, 40.0]:
    R = module_pass([1200.0], [sig], [1.0], 1000.0, SN)
    g = GRID.numpy()
    p = R['p'].numpy()
    mu_p = float((p * g).sum())
    var = float((p * (g - mu_p) ** 2).sum())
    alpha = W_of(sig) / (W_of(sig) + 2 * SN ** 2)
    print("  %-8.1f %-18.12f %-18.12f %-12.2e"
          % (sig, var / SN ** 2, alpha, abs(var / SN ** 2 - alpha)))
    assert abs(var / SN ** 2 - alpha) < 1e-4

# pi_i = w_i kappa_i rho_hat_i，且等于行边缘的归一化（式 21）
print("\n  pi_i from the module vs. w_i * kappa_i * rho_hat_i (eq. 21):")
R = module_pass([850.0, 1150.0], [0.0, 40.0], [0.5, 0.5], 1000.0, SN)
kap = np.array([chi_of(0.0) * chi_W(0.0), chi_of(40.0) * chi_W(40.0)])
w = np.array([0.5, 0.5])
rh = R['rho_hat'].numpy()
ref = w * kap * rh
ref = ref / ref.sum()
print("    pi (module) = %s" % np.array2string(R['pi'].numpy(), precision=12))
print("    pi (eq. 21) = %s" % np.array2string(ref, precision=12))
print("    max abs diff = %.3e" % np.abs(R['pi'].numpy() - ref).max())
assert np.abs(R['pi'].numpy() - ref).max() < 1e-5

# d_i = w_i (1 - rho_hat_i)，求和等于 1 - SC（式 9）
print("\n  conflict score identity  sum_i d_i = 1 - SC (eq. 9):")
lhs = float(R['d'].sum())
rhs = 1.0 - R['SC']
print("    sum d_i = %.12f   1 - SC = %.12f   diff = %.2e" % (lhs, rhs, abs(lhs - rhs)))
assert abs(lhs - rhs) < 1e-6

# sigma_i = 0 必须严格退回点锚点形式（硬正确性判据）
# 退化量：alpha = 2/3、tau = 1/3、chi = 0.816496580928、Var_p / sigma_nu^2 = 2/3，
# 且 rho_hat 的闭式与行边缘实测在数值精度内一致。
print("\n  sigma_i = 0 degeneration (hard criterion):")
R = module_pass([1200.0], [0.0], [1.0], 1000.0, SN)
g0 = GRID.numpy(); p0 = R['p'].numpy()
mu0 = float((p0 * g0).sum()); var0 = float((p0 * (g0 - mu0) ** 2).sum())
legacy_chi = (1.0 + 2.0 * SN ** 2 / EPS) ** -0.5
closed0 = float(np.exp(-(1200.0 - 1000.0) ** 2 / (2 * SN ** 2 + EPS)))
print("    chi      = %.15f  (expect 0.816496580928)" % legacy_chi)
print("    alpha    = %.15f  (expect 0.666666666667)" % (W_of(0.0) / (W_of(0.0) + 2 * SN ** 2)))
print("    tau      = %.15f  (expect 0.333333333333)" % (1 - W_of(0.0) / (W_of(0.0) + 2 * SN ** 2)))
print("    Var_p / sigma_nu^2 = %.15f  (expect 0.666666666667)" % (var0 / SN ** 2))
print("    rho_hat module = %.15f   closed form = %.15f   diff = %.2e"
      % (float(R['rho_hat'][0]), closed0, abs(float(R['rho_hat'][0]) - closed0)))
assert abs(legacy_chi - 0.816496580928) < 1e-12
assert abs(var0 / SN ** 2 - 2.0 / 3.0) < 1e-5
assert abs(float(R['rho_hat'][0]) - closed0) < 1e-6


# ------------------------------------------------------ 2. 题跋缺失的退化
hdr("[2] missing inscription: p = nu_tilde exactly, SC = 1, d empty")


def empty_or_zeroed(zero_weights):
    n = 4
    log_nu = gaussian_log_evidence(torch.full((n,), 1000.0),
                                   torch.full((n,), SN), GRID)
    if zero_weights:
        anchors = torch.zeros(n, 8, 3)
        anchors[..., 0] = 1200.0
        anchors[..., 1] = 5.0
        anchors[..., 2] = 0.0            # weights all zero -> identity likelihood
    else:
        anchors = torch.zeros(n, 0, 3)   # K = 0
    return log_nu, anchors


for flag in [False, True]:
    log_nu, anchors = empty_or_zeroed(flag)
    p, SC, Gamma, rho_hat, d, pi = fusion(log_nu, torch.full((4, 1), SN),
                                          anchors, GRID)
    ref = torch.softmax(log_nu, dim=-1)
    tag = "weights zeroed" if flag else "K = 0"
    print("  %-16s |p - nu_tilde| = %.2e   SC = %s   Gamma.shape = %s   d.numel = %d"
          % (tag, float((p - ref).abs().max()), float(SC.min()),
             tuple(Gamma.shape), d.numel()))
    assert float((p - ref).abs().max()) < 1e-6
    assert abs(float(SC.min()) - 1.0) < 1e-6


# --------------------------------------------------- 3. 端到端前向/反向
hdr("[3] end-to-end forward/backward on a mixed batch")
model = CIPDatingModel(GRID)
B, N = 6, 32
image = torch.rand(B, 3, config.IMG_SIZE, config.IMG_SIZE)
input_ids = torch.randint(0, 1000, (B, N))
mask = torch.ones(B, N, dtype=torch.long)
mask[0, :] = 0                      # sample 0 has no inscription
mask[1, 20:] = 0                    # sample 1 has a short inscription
year = torch.tensor([900.0, 1100.0, 1300.0, 1500.0, 1700.0, 1850.0])
dynasty = torch.tensor([0, 2, 3, 4, 5, 6])
consistent = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])

out = model(image, input_ids, mask)
loss, stats = total_loss(out, year, dynasty, consistent, GRID)
loss.backward()
gnorm = float(torch.cat([p.grad.flatten() for p in model.parameters()
                         if p.grad is not None]).norm())
print("  loss = %.6f  %s" % (float(loss), stats))
print("  gradient norm = %.6f" % gnorm)
print("  finite: loss=%s  grad=%s  y_hat=%s  sigma_p=%s"
      % (np.isfinite(float(loss)), np.isfinite(gnorm),
         bool(torch.isfinite(out['y_hat']).all()),
         bool(torch.isfinite(out['sigma_p']).all())))
assert np.isfinite(float(loss)) and np.isfinite(gnorm)
assert torch.isfinite(out['posterior']).all()

# 题跋缺失的样本：p 应严格等于归一化后的 nu（即 nu_tilde），而非 nu 本身
nu_tilde = torch.softmax(out['log_nu'], dim=-1)
print("  sample 0 (no inscription): |p - nu_tilde| = %.2e   SC = %.6f"
      % (float((out['posterior'][0] - nu_tilde[0]).abs().max()),
         float(out['consistency'][0])))
assert float(out['consistency'][0]) == 1.0
assert float((out['posterior'][0] - nu_tilde[0]).abs().max()) < 1e-6

# 锚点位置与宽度落在设计区间内
a = out['anchors']
print("  mu_i  in [%.1f, %.1f]  (grid range [%.0f, %.0f])"
      % (float(a[..., 0].min()), float(a[..., 0].max()), config.T_MIN, config.T_MAX))
print("  sigma_i in [%.4f, %.4f]  (floor %.1f)"
      % (float(a[..., 1].min()), float(a[..., 1].max()), config.SIGMA_MIN))
assert float(a[..., 0].min()) >= config.T_MIN - 1e-3
assert float(a[..., 1].min()) >= config.SIGMA_MIN - 1e-6

# 自洽度恒在 (0, 1]
print("  SC range: [%.6f, %.6f]" % (float(out['consistency'].min()),
                                    float(out['consistency'].max())))
assert float(out['consistency'].min()) > 0.0
assert float(out['consistency'].max()) <= 1.0 + 1e-6


# ------------------------------------------------- 4. 校准与检测指标自检
hdr("[4] calibration / detection helpers self-check")
from calibration import (auroc, auprc, fpr_at_tpr, crps, coverage,
                         expected_calibration_error, holm, brier_score)

# 完全可分时 AUROC / AUPRC 应为 1
s = np.array([0.9, 0.8, 0.2, 0.1])
y = np.array([1, 1, 0, 0])
print("  separable: AUROC=%.6f  AUPRC=%.6f  FPR@95TPR=%.6f"
      % (auroc(s, y), auprc(s, y), fpr_at_tpr(s, y)))
assert abs(auroc(s, y) - 1.0) < 1e-9 and abs(auprc(s, y) - 1.0) < 1e-9

# 完全不可分（全同分）时 AUROC 应为 0.5
print("  tied scores: AUROC=%.6f" % auroc(np.ones(4), y))
assert abs(auroc(np.ones(4), y) - 0.5) < 1e-9

# 覆盖率：窄后验应覆盖，宽后验宽度应显著更大；尖峰后验的区间宽度应为 0
g = np.linspace(config.T_MIN, config.T_MAX, config.L)
narrow = np.exp(-0.5 * ((g - 1000.0) / 5.0) ** 2); narrow /= narrow.sum()
flat = np.ones((1, config.L)) / config.L
c_narrow = coverage(narrow[None, :], g, np.array([1000.0]))
c_flat = coverage(flat, g, np.array([1000.0]))
print("  narrow posterior (sigma=5): %s" % c_narrow)
print("  flat   posterior         : %s" % c_flat)
assert c_narrow['cov80'] == 1.0 and c_narrow['width80'] < 20.0
assert c_flat['width80'] > 500.0

point = np.zeros((1, config.L)); point[0, np.argmin(np.abs(g - 1000.0))] = 1.0
print("  point mass: width80 = %.4f (must be 0)"
      % coverage(point, g, np.array([1000.0]))['width80'])
assert coverage(point, g, np.array([1000.0]))['width80'] == 0.0

print("  CRPS narrow=%.3f  flat=%.3f  MAE of the narrow posterior=%.3f"
      % (crps(narrow[None, :], g, np.array([1000.0])),
         crps(flat, g, np.array([1000.0])),
         abs(1000.0 - float((narrow * g).sum()))))

# ECE：分组内准确率恰等于平均置信度时应为 0
cf = np.full(5, 0.2)            # 全部落在同一箱，平均置信度 0.2
ec = np.array([1.0, 0.0, 0.0, 0.0, 0.0])   # 5 个中 1 个正确，准确率 0.2
print("  ECE when binned accuracy == mean confidence = %.6f"
      % expected_calibration_error(cf, ec, n_bins=5)[0])
assert expected_calibration_error(cf, ec, n_bins=5)[0] < 1e-12

# Holm 校正：最小 p 值乘以 m，且单调不减
raw = np.array([0.001, 0.02, 0.04, 0.5])
print("  Holm: %s -> %s" % (raw.tolist(), np.round(holm(raw), 6).tolist()))
assert abs(holm(raw)[0] - 0.004) < 1e-12

print("  Brier (uniform 7-class, one-hot target) = %.6f"
      % brier_score(np.full((1, 7), 1 / 7), np.array([3]), 7))

print("\n" + "=" * 72)
print("SMOKE TEST PASSED")
print("=" * 72)
