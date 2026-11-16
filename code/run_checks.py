import os, json
import numpy as np
from scipy.stats import t as student_t
import matplotlib; matplotlib.use("Agg")
import fig_style; fig_style.apply_style()
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
OI = fig_style.OKABE_ITO
from fedopt_stability import (so3, sun, karcher_mean, fedopt_run, rms_to_star,
                              make_data, left_act)

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "..", "figures"), exist_ok=True)

def kmax_of(g):   return 0.25 if g.name == "SO(3)" else 1.0

def sigma_rdp(C, T, delta, N, eps):
    Delta = 2.0 * C / N; Ld = -np.log(delta)
    gap = eps / (np.sqrt(Ld + eps) + np.sqrt(Ld))
    return Delta * np.sqrt(T) / (np.sqrt(2.0) * gap)

def achieved_eps(C, T, delta, N, sigma):
    Delta = 2.0 * C / N; A = T * Delta**2 / (2 * sigma**2); Ld = -np.log(delta)
    return A + 2 * np.sqrt(A * Ld)

def kappa_euclid(eta, gamma, tau):
    return abs(1.0 - gamma * (1.0 - (1.0 - eta) ** tau))

BASE = dict(eta=0.5, gamma=1.0, tau=1, C=0.3, T=20, delta=1e-5)
LPROJ = 1.0
SPREAD = 0.15
KAP0 = kappa_euclid(BASE["eta"], BASE["gamma"], BASE["tau"])
out = {"base": BASE, "kappa_euclid": KAP0}
log = []
def say(s): print(s, flush=True); log.append(s)

import fedopt_stability as _F
CEXP = 1.0
MEASURED_KAP = KAP0

def bclip_hat(grp, A, ts, r, C, eta, samples=300):
    N = len(A); best = 0.0
    for _ in range(samples):
        xi = grp.sample_tangent(1.0); nn = grp.norm(xi)
        th = grp.exp(ts, xi * (_F.RNG.uniform(0.0, r) / nn)) if nn > 0 else ts.copy()
        ov = sum(max(0.0, eta * grp.dist(th, a) - C) for a in A) / N
        best = max(best, ov)
    return best

def floor_and_bound(grp, N, eps, r, M, diagnostic_seed=None):
    A = make_data(grp, N, SPREAD); ts = karcher_mean(grp, A)
    sig = sigma_rdp(BASE["C"], BASE["T"], BASE["delta"], N, eps)
    kw = dict(eta=BASE["eta"], gamma=BASE["gamma"], tau=BASE["tau"], C=BASE["C"],
              sigma=sig, r=r, T=BASE["T"], noisy=True)
    emp = rms_to_star(grp, A, ts, M, **kw)
    if diagnostic_seed is not None:
        _F.RNG = np.random.default_rng(diagnostic_seed)
    bhat = bclip_hat(grp, A, ts, r, BASE["C"], BASE["eta"])
    bound = (LPROJ * CEXP * BASE["gamma"] / (1 - MEASURED_KAP)) * (bhat + sig * np.sqrt(grp.dim))
    return emp, bound, sig

def floor_ci(factory, N, eps, r, M, S, seed0=2000):
    emps, bounds = [], []
    sig = None
    for seed in range(S):
        _F.RNG = np.random.default_rng(seed0 + seed)
        emp, bound, sig = floor_and_bound(
            factory(), N, eps, r, M, diagnostic_seed=100000 + seed0 + seed)
        emps.append(emp)
        bounds.append(bound)
    e = np.asarray(emps, float)
    b = np.asarray(bounds, float)
    tmult = float(student_t.ppf(0.975, S - 1)) if S > 1 else float("inf")
    ehalf = float(tmult * e.std(ddof=1) / np.sqrt(S)) if S > 1 else float("nan")
    bhalf = float(tmult * b.std(ddof=1) / np.sqrt(S)) if S > 1 else float("nan")
    return float(e.mean()), ehalf, float(b.mean()), bhalf, sig


def floor_empirical_ci(factory, N, eps, r, M, S, seed0=2000):
    emps = []
    sig = None
    for seed in range(S):
        _F.RNG = np.random.default_rng(seed0 + seed)
        grp = factory()
        A = make_data(grp, N, SPREAD)
        ts = karcher_mean(grp, A)
        sig = sigma_rdp(BASE["C"], BASE["T"], BASE["delta"], N, eps)
        kw = dict(eta=BASE["eta"], gamma=BASE["gamma"], tau=BASE["tau"], C=BASE["C"],
                  sigma=sig, r=r, T=BASE["T"], noisy=True)
        emps.append(rms_to_star(grp, A, ts, M, **kw))
    e = np.asarray(emps, float)
    tmult = float(student_t.ppf(0.975, S - 1)) if S > 1 else float("inf")
    ehalf = float(tmult * e.std(ddof=1) / np.sqrt(S)) if S > 1 else float("nan")
    return float(e.mean()), ehalf, sig

def slope_ci(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    sl, ic = np.polyfit(x, y, 1)
    r = y - (sl * x + ic)
    se = np.sqrt((r @ r) / (n - 2)) / np.sqrt(((x - x.mean()) ** 2).sum())
    return float(sl), float(student_t.ppf(0.975, n - 2) * se)

say("=== Noise-scale check (C=%.2f, T=%d, delta=%.0e); geometry regime: see"
    " assumption_checks.py / Table 5 (SO(3) in-regime, SU(4) out) ===" %
    (BASE["C"], BASE["T"], BASE["delta"]))
say("  grp     N   eps    sigma  eps_ach  Kmax  r     pi/(2sqrtK)  noise/r")
def check_row(g, N, eps, r):
    Kmax = kmax_of(g); bound_rad = np.pi / (2 * np.sqrt(Kmax))
    sig = sigma_rdp(BASE["C"], BASE["T"], BASE["delta"], N, eps)
    eps_ach = achieved_eps(BASE["C"], BASE["T"], BASE["delta"], N, sig)
    noise = BASE["gamma"] * sig * np.sqrt(g.dim) / (1 - KAP0)
    say("  %-6s %4d %5.2f %7.4f %7.3f %5.2f %5.2f %10.3f %8.3f" %
        (g.name, N, eps, sig, eps_ach, Kmax, r, bound_rad, noise / r))
    return noise < r
for (g, N, eps, r) in [(so3(), 200, 1.0, 0.6), (so3(), 800, 1.0, 0.6),
                       (sun(4), 400, 1.0, 0.4)]:
    check_row(g, N, eps, r)

say("=== (C) Equivariance  max|theta_T(h.D) - h.theta_T(D)| ===")
for g in (so3(), sun(3)):
    A = make_data(g, 10, SPREAD); ts = karcher_mean(g, A)
    h = g.exp(g.I, g.sample_tangent(0.6))
    kw = dict(eta=BASE["eta"], gamma=BASE["gamma"], tau=BASE["tau"], C=BASE["C"],
              sigma=0.0, r=0.6, T=12, noisy=False)
    th = fedopt_run(g, A, ts, ts.copy(), **kw)
    thh = fedopt_run(g, left_act(g, h, A), h @ ts, (h @ ts).copy(), **kw)
    err = float(np.max(np.abs(thh - h @ th))); out[f"equiv_{g.name}"] = err
    say(f"  {g.name}: {err:.2e}")

say("=== (B) Contraction  measured rate vs Euclidean reference (the two-point factor"
    " adds beta_Jac>0; the one-point factor kappa_fed does not) ===")
for g in (so3(), sun(3)):
    A = make_data(g, 20, SPREAD); ts = karcher_mean(g, A)
    xi = g.sample_tangent(1.0); th = g.exp(ts, xi * (0.5 / g.norm(xi)))
    d = [g.dist(th, ts)]
    for _ in range(8):
        th = fedopt_run(g, A, ts, th, eta=BASE["eta"], gamma=BASE["gamma"],
                        tau=BASE["tau"], C=BASE["C"], sigma=0.0, r=0.6, T=1, noisy=False)
        d.append(g.dist(th, ts))
    d = np.array(d)
    k = max(2, int(np.argmax(d < 5 * d[-1])) or 4)
    sl = np.polyfit(np.arange(k + 1), np.log(d[:k + 1]), 1)[0]
    mk = float(np.exp(sl)); out[f"kappa_{g.name}"] = [mk, KAP0]
    say(f"  {g.name}: measured={mk:.3f}  euclid_reference={KAP0:.3f}  "
        f"difference={mk-KAP0:+.3f} (relative to the Euclidean reference)")

MEASURED_KAP = out["kappa_SO(3)"][0]
say(f"  empirical noise-plus-clipping proxy uses measured kappa_hat={MEASURED_KAP:.3f} (diagnostic, not a public bound)")

say("=== (A)+(D) Floor vs 1/(N eps): SO(3), r=0.6, exact RDP sigma ===")
r = 0.6; Ne, fl, bd, cis, bcis = [], [], [], [], []
for (N, eps) in [(100, 1.0), (200, 1.0), (400, 1.0), (800, 1.0),
                 (200, 0.5), (200, 2.0), (200, 4.0)]:
    emp, ci, bound, bci, sig = floor_ci(lambda: so3(), N, eps, r, M=40, S=5)
    Ne.append(N * eps); fl.append(emp); bd.append(bound); cis.append(ci); bcis.append(bci)
    say(f"  N={N:4d} eps={eps:.1f}  Neps={N*eps:6.1f}  sigma={sig:.4f}  "
        f"emp={emp:.4f}+-{ci:.4f}  proxy={bound:.4f}+-{bci:.4f}  emp/proxy={emp/bound:.2f}")
out["so3_Neps"], out["so3_floor"], out["so3_bound"], out["so3_ci"], out["so3_bound_ci"] = Ne, fl, bd, cis, bcis
Ne1, fl1, ci1 = zip(*[(n, f, c) for (n, e), n2, f, c in zip(
    [(100,1.0),(200,1.0),(400,1.0),(800,1.0),(200,0.5),(200,2.0),(200,4.0)], Ne, fl, cis)
    if e == 1.0])
sl, sl_ci = slope_ci(np.log(Ne1), np.log(fl1))
out["so3_slope_vs_Neps"], out["so3_slope_ci95"] = sl, sl_ci
say(f"  log-log slope floor vs (N eps), eps=1 points: {sl:.3f}  95% CI "
    f"[{sl-sl_ci:.3f},{sl+sl_ci:.3f}]  (predict -1)")

say("=== (D) Floor vs sqrt(d): r=0.4, N=400, eps=1 (SU(n) out of the proved regime; sqrt(d) scaling check) ===")
r = 0.4; dims, floors, cis_d = [], [], []
for fac, dd, M in [(lambda: so3(), 3, 40), (lambda: sun(2), 3, 30),
                   (lambda: sun(3), 8, 25), (lambda: sun(4), 15, 20)]:
    emp, ci, sig = floor_empirical_ci(fac, 400, 1.0, r, M=M, S=4)
    dims.append(dd); floors.append(emp); cis_d.append(ci)
    say(f"  d={dd:2d}  sigma={sig:.4f}  emp={emp:.4f}+-{ci:.4f}"
        f"  emp/sqrt(d)={emp/np.sqrt(dd):.4f}")
out["dims"], out["floors_vs_d"], out["ci_d"] = dims, floors, cis_d
sld, sld_ci = slope_ci(np.log(np.sqrt(dims)), np.log(floors))
out["slope_vs_sqrtd"], out["slope_vs_sqrtd_ci95"] = sld, sld_ci
say(f"  log-log slope floor vs sqrt(d): {sld:.3f}  95% CI "
    f"[{sld-sld_ci:.3f},{sld+sld_ci:.3f}]  (predict +1)")

out["so3_empirical_mean_below_sampled_proxy"] = bool(all(f <= b for f, b in zip(fl, bd)))
say("=== Cross-seed empirical mean below the sampled diagnostic proxy in every tested "
    f"SO(3) configuration: {out['so3_empirical_mean_below_sampled_proxy']} ===")

with open(os.path.join(HERE, "..", "results", "results.json"), "w") as f: json.dump(out, f, indent=2)
with open(os.path.join(HERE, "..", "results", "results.txt"), "w") as f: f.write("\n".join(log) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0), constrained_layout=True)
ax[0].errorbar(Ne1, fl1, yerr=ci1, fmt="o-", capsize=3, color=OI["blue"],
               label="Empirical RMS")
ax[0].errorbar(Ne[:4], bd[:4], yerr=bcis[:4], fmt="s--", capsize=3,
               color=OI["vermillion"], label="Noise + clipping proxy")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel(r"$N\varepsilon$"); ax[0].set_ylabel(r"$\sqrt{\mathbb{E}\,d^2(\theta_T,\theta^\star)}$")
ax[0].set_xticks([100, 200, 400, 800])
ax[0].xaxis.set_major_formatter(ScalarFormatter())
ax[0].xaxis.set_minor_formatter(NullFormatter())
ax[0].legend(loc="lower left", fontsize=9, frameon=False)
ax[0].set_ylim(top=ax[0].get_ylim()[1] * 1.8)
fig_style.panel_label(ax[0], "(a)")
ax[1].errorbar(np.sqrt(dims), floors, yerr=cis_d, fmt="o-", color=OI["blue"], capsize=3)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
offsets = {"SO(3)": (-8, 18), "SU(2)": (24, -10), "SU(3)": (-28, 16), "SU(4)": (-28, 14)}
for x, y, nm in zip(np.sqrt(dims), floors, ["SO(3)", "SU(2)", "SU(3)", "SU(4)"]):
    ax[1].annotate(nm, (x, y), textcoords="offset points",
                   xytext=offsets.get(nm, (7, 5)), fontsize=11, ha="center",
                   arrowprops=dict(arrowstyle="-", color="0.55", lw=0.7,
                                   shrinkA=1, shrinkB=3))
ax[1].set_xlabel(r"$\sqrt{d}$"); ax[1].set_ylabel("Empirical floor")
ax[1].set_xlim(1.5, 4.6)
ax[1].set_ylim(top=ax[1].get_ylim()[1] * 1.7)
fig_style.panel_label(ax[1], "(b)")
fig.savefig(os.path.join(HERE, "..", "figures", "stability.png"))
say("saved figs/stability.png and results.json")
