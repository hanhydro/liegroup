import numpy as np
from fedopt_stability import so3, karcher_mean, clip, retract

GRP = so3()
D = GRP.dim
CENTER = GRP.I.copy()
R, GAMMA, C, TAU, ETA, T = 0.20, 1.0, 0.10, 1, 0.5, 10
RHO_STAR = 0.60
B_FIN = 0.35
DELTA, MTRAJ = 1e-5, 2000
EPS = [0.5, 1.0, 2.0, 4.0, 8.0]
N_ANCH, A_SUPP = 8, 0.15

_pub = np.random.default_rng(20260810)
def _ball(rng, radius):
    z = rng.normal(size=D); v = sum(z[k] * GRP.basis[k] for k in range(D))
    n = GRP.norm(v)
    return v * (rng.uniform(0, radius) / n) if n > 0 else v * 0
ANCHORS = [GRP.exp(CENTER, _ball(_pub, A_SUPP)) for _ in range(N_ANCH)]
THETA_STAR = karcher_mean(GRP, ANCHORS)

def sigma_rdp(C, T, delta, N, eps):
    Delta = 2.0 * C / N
    Ld = -np.log(delta)
    gap = eps / (np.sqrt(Ld + eps) + np.sqrt(Ld))
    return Delta * np.sqrt(T) / (np.sqrt(2.0) * gap)

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    ph = k / n; den = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - half), min(1.0, ctr + half)

def trajectory(sigma, B, Zstd):
    theta = CENTER.copy(); exited = False; exit_rounds = 0; fires = 0; dists = []
    for t in range(T):
        us = []
        for a in ANCHORS:
            ti = theta.copy()
            for _ in range(TAU):
                ti = GRP.exp(ti, ETA * GRP.log(ti, a))
            us.append(clip(GRP, GRP.log(theta, ti), C))
        ubar = sum(us) / len(ANCHORS)
        Z = sigma * sum(Zstd[t, k] * GRP.basis[k] for k in range(D))
        v = ubar + Z
        if B is not None:
            nv = GRP.norm(v)
            if nv > B:
                v = v * (B / nv); fires += 1
        pre = GRP.exp(theta, GAMMA * v)
        dc = GRP.dist(CENTER, pre); dists.append(dc)
        if dc > RHO_STAR:
            exit_rounds += 1; exited = True
            theta = CENTER.copy()
        else:
            theta = retract(GRP, CENTER, pre, R)
    return exited, exit_rounds, fires, dists, GRP.dist(theta, THETA_STAR)

print(f"SO(3) declared-domain ablation: center=identity(public), N={N_ANCH} anchors in a "
      f"radius-{A_SUPP} public ball, r={R}, gamma={GAMMA}, C={C}, T={T}, rho*={RHO_STAR}, "
      f"finite B={B_FIN} (r+gamma*B={R+GAMMA*B_FIN:.2f}<rho*), M={MTRAJ} trajectories.")
print("Clipped mechanism: r+gamma*B<rho* gives a DETERMINISTIC zero-exit guarantee.\n")
print(" eps | sigma  | chi   | plain any_exit% [Wilson95]  round% | clip any_exit% | clip% |"
      " n_noexit | plain RMS|noexit | clip RMS | paired dRMS|noexit")
rows = {}
for eps in EPS:
    sig = sigma_rdp(C, T, DELTA, N_ANCH, eps)
    chi = GAMMA * sig * np.sqrt(D) / (RHO_STAR - R - GAMMA * C)
    rng = np.random.default_rng(7000 + int(100 * eps))
    Z = rng.standard_normal((MTRAJ, T, D))
    pe = 0; pr = 0; cf = 0
    pdist = []; cdist = []
    perr = np.empty(MTRAJ); cerr = np.empty(MTRAJ); pexit = np.zeros(MTRAJ, bool)
    for m in range(MTRAJ):
        ex, er, fi, ds, err = trajectory(sig, None, Z[m])
        pe += ex; pr += er; pdist += ds; perr[m] = err; pexit[m] = ex
        ex2, er2, fi2, ds2, err2 = trajectory(sig, B_FIN, Z[m])
        cf += fi2; cdist += ds2; cerr[m] = err2
    lo, hi = wilson(pe, MTRAJ)
    keep = ~pexit
    n_ok = int(keep.sum())
    prms_ok = float(np.sqrt(np.mean(perr[keep]**2))) if n_ok else float("nan")
    crms = float(np.sqrt(np.mean(cerr**2)))
    if n_ok:
        d = cerr[keep] - perr[keep]
        dmean = float(d.mean()); dhalf = float(1.96 * d.std(ddof=1) / np.sqrt(n_ok)) if n_ok > 1 else 0.0
    else:
        dmean = float("nan"); dhalf = float("nan")
    rows[eps] = dict(sigma=sig, chi=chi, plain_any=pe / MTRAJ, plain_ci=(lo, hi),
                     plain_round=pr / (MTRAJ * T), clip_fire=cf / (MTRAJ * T),
                     n_noexit=n_ok, plain_rms_noexit=prms_ok, clip_rms=crms,
                     dmean=dmean, dhalf=dhalf,
                     pdist=np.percentile(pdist, [50, 95, 99]).tolist(),
                     cdist=np.percentile(cdist, [50, 95, 99]).tolist())
    print(f" {eps:4.1f}| {sig:.4f} | {chi:5.2f} | "
          f"{100*pe/MTRAJ:5.1f} [{100*lo:.1f},{100*hi:.1f}]  {100*pr/(MTRAJ*T):4.1f} | "
          f"  0.0          | {100*cf/(MTRAJ*T):4.0f}  | "
          f"{n_ok:5d}    | {prms_ok:.4f}          | {crms:.4f}   | {dmean:+.4f}+-{dhalf:.4f}")
print("\nPrimary result is the run-level exit probability. Utility is reported only on "
      "in-domain (no-exit) plain runs; the fallback-to-center RMS is confounded because the "
      "public center is near the target here, so it is not reported as utility.")

print("\nEndpoint-distance quantiles [p50,p95,p99] (rad):")
for eps in EPS:
    print(f"  eps={eps:4.1f}: plain {np.round(rows[eps]['pdist'],3)}  "
          f"clipped {np.round(rows[eps]['cdist'],3)}")

np.savez("../results/rq1_locality_ablation.npz",
         eps=np.array(EPS),
         sigma=np.array([rows[e]["sigma"] for e in EPS]),
         chi=np.array([rows[e]["chi"] for e in EPS]),
         plain_any=np.array([rows[e]["plain_any"] for e in EPS]),
         plain_ci=np.array([rows[e]["plain_ci"] for e in EPS]),
         plain_round=np.array([rows[e]["plain_round"] for e in EPS]),
         clip_fire=np.array([rows[e]["clip_fire"] for e in EPS]),
         n_noexit=np.array([rows[e]["n_noexit"] for e in EPS]),
         plain_rms_noexit=np.array([rows[e]["plain_rms_noexit"] for e in EPS]),
         clip_rms=np.array([rows[e]["clip_rms"] for e in EPS]),
         dmean=np.array([rows[e]["dmean"] for e in EPS]),
         dhalf=np.array([rows[e]["dhalf"] for e in EPS]),
         pdist=np.array([rows[e]["pdist"] for e in EPS]),
         cdist=np.array([rows[e]["cdist"] for e in EPS]),
         rho_star=RHO_STAR, B_fin=B_FIN, r=R, C=C, gamma=GAMMA, mtraj=MTRAJ)
print("\nsaved -> results/rq1_locality_ablation.npz")
