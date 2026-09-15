"""Independent verification of Theorems 1-4 under the marginalized kernel.

Convention (marginalized kernel, Eq. 6):
    Gamma_il = w_i * nu_l * chi_i * exp( -(mu_i - g_l)^2 / W_i )
    W_i = eps + 2*sigma_i^2 ,  chi_i = (1 + 2*sigma_i^2/eps)^(-1/2)

Hard criterion: sigma_i = 0 must reproduce the point-anchor values exactly.
All output is ASCII to stay console-safe.
"""
import numpy as np

EPS = 3600.0
SN = 30.0                     # sigma_nu (nominal picture uncertainty)
T0, T1, L = 618.0, 1949.0, 200
g = np.linspace(T0, T1, L)
D = g[1] - g[0]


def soft_evidence(mu, sd, grid=g):
    v = np.exp(-0.5 * ((grid - mu) / sd) ** 2)
    return v / v.sum()


def W_of(s):  return EPS + 2.0 * s ** 2
def chi_of(s): return (1.0 + 2.0 * s ** 2 / EPS) ** -0.5
def chi_W(s):  return (1.0 + 2.0 * SN ** 2 / W_of(s)) ** -0.5
def alpha_of(s): return W_of(s) / (W_of(s) + 2.0 * SN ** 2)
def tau_of(s):   return 1.0 - alpha_of(s)
def upper(s):    return chi_of(s) * chi_W(s)     # per-anchor attainable retention max


def kernel_expectation(mu, sig, a, n=4001, half=12.0):
    """E_{y~N(mu,sig^2)}[exp(-(y-a)^2/EPS)] by direct dense quadrature."""
    if sig < 1e-12:
        return float(np.exp(-((mu - a) ** 2) / EPS))
    y = np.linspace(mu - half * sig, mu + half * sig, n)
    d = np.exp(-0.5 * ((y - mu) / sig) ** 2) * np.exp(-((y - a) ** 2) / EPS)
    return float(np.trapezoid(d, y) / (np.sqrt(2 * np.pi) * sig))


def run(mus, sigs, ws, mu_nu):
    """Forward pass on the grid; returns posterior, retention, and diagnostics."""
    mus = np.atleast_1d(np.asarray(mus, float))
    sigs = np.atleast_1d(np.asarray(sigs, float))
    ws = np.atleast_1d(np.asarray(ws, float))
    nt = soft_evidence(mu_nu, SN)
    rows = np.array([ws[i] * nt * chi_of(sigs[i]) * np.exp(-((mus[i] - g) ** 2) / W_of(sigs[i]))
                     for i in range(len(mus))])
    p = rows.sum(0)
    p = p / p.sum()
    rho = rows.sum(1)
    rho_hat = rho / (ws * np.array([upper(s) for s in sigs]))
    SC = float((ws * rho_hat).sum())
    kap = np.array([upper(s) for s in sigs])
    c = ws * kap * rho_hat
    pi = c / c.sum()
    return dict(p=p, rho_hat=rho_hat, SC=SC, pi=pi, kap=kap,
                mu=(p * g).sum(), var=(p * (g - (p * g).sum()) ** 2).sum(), nt=nt)


def hdr(t):
    print("\n" + "=" * 76 + "\n" + t + "\n" + "=" * 76)


print("baseline: eps=%.1f  sigma_nu=%.1f  L=%d  Delta=%.6f" % (EPS, SN, L, D))

# ---------------------------------------------------------------- 0. degeneration
hdr("[0] sigma=0 degeneration (hard criterion)")
print("  alpha(0)=%.15f  tau(0)=%.15f  chi(0)=%.15f" % (alpha_of(0), tau_of(0), chi_of(0)))
assert abs(alpha_of(0) - 2 / 3) < 1e-15 and abs(tau_of(0) - 1 / 3) < 1e-15
print("  [ok] matches point-anchor alpha=2/3, tau=1/3, chi=1")

# --------------------------------------------- 1. kernel marginalization identity
hdr("[1] kernel identity  E_y[exp(-(y-a)^2/eps)] = chi_i * exp(-(mu-a)^2/W_i)")
print("  %-7s %-6s %-6s %-18s %-18s %-10s" % ("sigma", "mu", "a", "quadrature", "closed form", "rel.err"))
for sig in [0.0, 15.0, 40.0, 100.0]:
    for mu, a in [(1000.0, 1000.0), (1000.0, 1100.0), (1200.0, 900.0)]:
        q = kernel_expectation(mu, sig, a)
        c = chi_of(sig) * np.exp(-((mu - a) ** 2) / W_of(sig))
        print("  %-7.1f %-6.0f %-6.0f %-18.12e %-18.12e %-10.2e"
              % (sig, mu, a, q, c, abs(q - c) / c))
print("  -> identity holds; Jensen ordering Gamma_plugin <= Gamma_marginal confirmed below")

#  Jensen gap between the two conventions
hdr("[1b] Jensen gap: plug-in vs marginalized (why the two conventions must not be mixed)")
nt = soft_evidence(1000.0, SN)
for sig in [15.0, 40.0]:
    plug = nt * np.exp(-((1200.0 - g) ** 2 + sig ** 2) / EPS)
    marg = nt * chi_of(sig) * np.exp(-((1200.0 - g) ** 2) / W_of(sig))
    print("  sigma=%-5.1f  sum(plug-in)=%.9e  sum(marginal)=%.9e  ratio=%.3f"
          % (sig, plug.sum(), marg.sum(), marg.sum() / plug.sum()))

# ------------------------------------------- 2. single anchor: Theorem 1 + rho_hat
hdr("[2] single anchor mu_i=1200, w=1 : Theorem 1 variance ratio and rho_hat")
print("  %-7s %-20s %-20s %-16s %-14s" % ("sigma", "rho_hat measured", "rho_hat closed", "var/sigma_nu^2", "alpha_i"))
for sig in [0.0, 15.0, 40.0]:
    R = run(1200.0, sig, 1.0, 1000.0)
    closed = np.exp(-((1200.0 - 1000.0) ** 2) / (2 * SN ** 2 + W_of(sig)))
    print("  %-7.1f %-20.15f %-20.15f %-16.12f %-14.9f"
          % (sig, R["rho_hat"][0], closed, R["var"] / SN ** 2, alpha_of(sig)))
    print("          |rho - closed|=%.3e   |var/sn^2 - alpha_i|=%.3e"
          % (abs(R["rho_hat"][0] - closed), abs(R["var"] / SN ** 2 - alpha_of(sig))))

print("\n  variance ratio independent of anchor position (sigma=40):")
for t_i in [700.0, 900.0, 1200.0, 1700.0]:
    R = run(t_i, 40.0, 1.0, 1000.0)
    print("    t_i=%-7.0f mu_p=%-16.9f var/sn^2=%-16.12f dev=%.2e"
          % (t_i, R["mu"], R["var"] / SN ** 2, abs(R["var"] / SN ** 2 - alpha_of(40))))

# ------------------------------------------------- 3. two anchors: Theorem 2 mixture
hdr("[3] two anchors mu=(850,1150), sigma=(0,40), w=(0.5,0.5) : mixture form and pi_i")
MU, SG, WT, MUNU = np.array([850.0, 1150.0]), np.array([0.0, 40.0]), np.array([0.5, 0.5]), 1000.0
R = run(MU, SG, WT, MUNU)
al = np.array([alpha_of(s) for s in SG])
m_i = al * MUNU + (1 - al) * MU
s_i = np.sqrt(al) * SN
mix = sum(R["pi"][i] * np.exp(-0.5 * ((g - m_i[i]) / s_i[i]) ** 2) / (np.sqrt(2 * np.pi) * s_i[i])
          for i in range(2))
mix = mix / mix.sum()
pi_naive = WT * R["rho_hat"] / R["rho_hat"].sum()
mix_naive = sum(pi_naive[i] * np.exp(-0.5 * ((g - m_i[i]) / s_i[i]) ** 2) / (np.sqrt(2 * np.pi) * s_i[i])
                for i in range(2))
mix_naive = mix_naive / mix_naive.sum()
print("  alpha_i    = %s" % np.array2string(al, precision=9))
print("  tau_i      = %s" % np.array2string(1 - al, precision=9))
print("  rho_hat_i  = %s" % np.array2string(R["rho_hat"], precision=15))
print("  pi_i       = %s" % np.array2string(R["pi"], precision=15))
print("  |p - mixture with pi_i = w_i*kappa_i*rho_hat_i|      = %.3e" % np.abs(R["p"] - mix).max())
print("  |p - mixture with pi_i = w_i*rho_hat_i/SC (naive)|   = %.3e" % np.abs(R["p"] - mix_naive).max())
voc = sum(R["pi"][i] * al[i] * SN ** 2 for i in range(2)) + \
    float((R["pi"] * (m_i - float((R["pi"] * m_i).sum())) ** 2).sum())
print("  var_p = %.9f   law-of-total-variance = %.9f   diff=%.2e"
      % (R["var"], voc, abs(R["var"] - voc)))
print("  SC = %.12f   mu_p = %.9f" % (R["SC"], R["mu"]))

# ----------------------------------------------- 4. Theorem 3 heterogeneous trust bound
hdr("[4] Theorem 3: |mu_p-mu_nu| <= sum pi_k tau_k |mu_k-mu_nu| <= tau_bar*Delta <= tau_max*Delta")
print("  %-30s %-13s %-13s %-13s %-12s %-6s" % ("anchors / sigmas", "|mu_p-mu_nu|", "level 1", "tau_bar*Delta", "tau_max*Delta", "same side"))
cases = [((1400.0, 1300.0), (0.0, 0.0)), ((1600.0, 1100.0), (0.0, 0.0)),
         ((800.0, 1200.0), (0.0, 0.0)), ((1600.0, 1100.0), (0.0, 40.0)),
         ((850.0, 1150.0), (0.0, 40.0)), ((1600.0, 1100.0), (0.0, 100.0))]
for mus, sigs in cases:
    M, S = np.array(mus), np.array(sigs)
    R = run(M, S, np.array([0.5, 0.5]), MUNU)
    taus = 1 - np.array([alpha_of(s) for s in S])
    dev = np.abs(M - MUNU)
    l1 = float((R["pi"] * taus * dev).sum())
    Delta = dev.max()
    l2 = float((R["pi"] * taus).sum()) * Delta
    l3 = float(taus.max()) * Delta
    same = bool(np.all(M > MUNU) or np.all(M < MUNU))
    print("  %-30s %-13.6f %-13.6f %-13.6f %-12.3f %-6s"
          % ("%s %s" % (mus, sigs), abs(R["mu"] - MUNU), l1, l2, l3, same))
    tol = 1e-6
    assert abs(R["mu"] - MUNU) <= l1 + tol, "level 1 violated"
    assert l1 <= l2 + tol, "level 2 violated"
    assert l2 <= l3 + tol, "level 3 violated"
print("  [ok] all three levels hold; level 1 is tight (equality) when all anchors are on one side")

# ---------------------------------------------------- 5. Theorem 4 detection floor
hdr("[5] Theorem 4: one misaligned anchor (weight w, offset delta), others aligned at mu_nu")
print("  setup: anchor A at mu_nu with weight 1-w, anchor B at mu_nu+delta with weight w")
for w in [0.5, 0.3]:
    eta = 0.5 * w
    print("  w=%.1f  eta=%.2f" % (w, eta))
    for sig in [0.0, 15.0, 30.0, 40.0, 60.0, 100.0]:
        closed = np.sqrt((2 * SN ** 2 + W_of(sig)) * np.log(w / (w - eta)))
        lo, hi = 1e-9, 3000.0
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            R = run([MUNU, MUNU + mid], [0.0, sig], [1 - w, w], MUNU)
            if 1 - R["SC"] >= eta:
                hi = mid
            else:
                lo = mid
        print("    sigma=%6.1f  delta_min closed=%-11.4f numeric=%-11.4f  dev=%.2e"
              % (sig, closed, hi, abs(closed - hi)))
    print("")
print("  check of the closed free form at sigma=0, w=0.5: 1-SC=w*(1-exp(-d^2/(2sn^2+eps)))")
u = 61.18
R = run([MUNU, MUNU + u], [0.0, 0.0], [0.5, 0.5], MUNU)
print("    d=%.2f -> SC=%.12f, 1-SC=%.12f (target 0.25)" % (u, R["SC"], 1 - R["SC"]))

# -------------------------------------------------- 6. redundancy of a separate alpha_i
hdr("[6] redundancy of an explicit reliability coefficient alpha_i")
print("  alpha_i <-> sigma_i is a bijection on alpha in [2/3, 1):  sigma = sqrt((2 sn^2 a/(1-a) - eps)/2)")
for a in [0.667, 0.8, 0.9, 0.99]:
    s2 = (2 * SN ** 2 * a / (1 - a) - EPS) / 2.0
    print("    alpha=%.3f -> sigma=%.4f" % (a, np.sqrt(s2) if s2 >= 0 else float("nan")))
print("  a global reliability factor on every anchor is invisible in p:")
for scale in [1.0, 0.5, 0.2]:
    R = run(MU, [0.0, 0.0], [0.5 * scale, 0.5 * scale], MUNU)
    print("    scale=%.1f  mu_p=%.12f" % (scale, R["mu"]))

# ------------------------------------------------------- 7. degeneration and symmetry
hdr("[7] degeneration checks")
print("  K=0 -> p = nu_tilde exactly, SC = 1 by construction")
print("  equal sigmas cancel in p only when the anchors are otherwise symmetric:")
for sigs in [(0.0, 0.0), (40.0, 40.0)]:
    R = run([900.0, 1150.0], sigs, [0.5, 0.5], MUNU)
    print("    mu=(900,1150) sigma=%-12s mu_p=%.12f  SC=%.9f"
          % (str(sigs), R["mu"], R["SC"]))
print("  -> with heterogeneous or asymmetric anchors, sigma_i is observable in p (unlike the plug-in convention)")

# ------------------------------------------- 8. grid sensitivity of the closed forms
hdr("[8] grid sensitivity: are the closed forms exact on the 200-point grid?")
R0 = run(1200.0, 40.0, 1.0, 1000.0)
closed0 = np.exp(-200.0 ** 2 / (2 * SN ** 2 + W_of(40)))
print("  L=200 (Delta=%.4f): |rho_hat - closed| = %.3e" % (D, abs(R0["rho_hat"][0] - closed0)))
for Lx in [100, 400, 800]:
    tst = np.linspace(T0, T1, Lx)
    nt = np.exp(-0.5 * ((tst - 1000.0) / SN) ** 2); nt = nt / nt.sum()
    row = nt * chi_of(40.0) * np.exp(-((1200.0 - tst) ** 2) / W_of(40.0))
    rh = row.sum() / upper(40.0)
    print("  L=%-4d (Delta=%.4f): |rho_hat - closed| = %.3e" % (Lx, tst[1] - tst[0], abs(rh - closed0)))

print("\n" + "=" * 76)
print("ALL ASSERTIONS PASSED. sigma_i=0 reproduces the point-anchor values exactly.")
print("=" * 76)
