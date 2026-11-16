import os
import numpy as np
from fedopt_stability import so3, karcher_mean, clip, retract
from capped_schedule import finite_ceiling_schedule

N, C, DELTA, EPS, R = 200, 0.4, 1e-5, 1.0, 0.6
ETA, GAMMA, TAU, KAPPA = 0.5, 1.0, 1, 0.52
B_SAFE, SIG_MAX = 0.87, 1e4
NSEED = int(os.environ.get("RQ4_NSEED", 20))
MB = int(os.environ.get("RQ4_MB", 30))
TS = [int(x) for x in os.environ.get("RQ4_TS", "8,16,32,64").split(",")]
from scipy.stats import t as _tdist
TMUL = float(_tdist.ppf(0.975, NSEED - 1))

G = so3(); D = G.dim
CENTER = G.I.copy()

def sigma_rdp(C, T, delta, N, eps):
    Delta = 2.0 * C / N; Ld = -np.log(delta)
    gap = eps / (np.sqrt(Ld + eps) + np.sqrt(Ld))
    return Delta * np.sqrt(T) / (np.sqrt(2.0) * gap)

def total_rdp_S(C, T, delta, N, eps):
    sig = sigma_rdp(C, T, delta, N, eps); return T / sig**2, sig

def anchors_for(seed):
    rng = np.random.default_rng(1000 + seed)
    A = []
    for _ in range(N):
        z = rng.normal(size=D); v = sum(z[k] * G.basis[k] for k in range(D))
        nn = G.norm(v); v = v * (rng.uniform(0, 0.15) / nn) if nn > 0 else v
        A.append(G.exp(CENTER, v))
    return A

def run_block(A, block, sigmas, count_clip=False):
    theta = CENTER.copy(); cc = 0
    for t in range(len(sigmas)):
        us = []
        for a in A:
            ti = theta.copy()
            for _ in range(TAU):
                ti = G.exp(ti, ETA * G.log(ti, a))
            raw = G.log(theta, ti)
            if count_clip and G.norm(raw) > C:
                cc += 1
            us.append(clip(G, raw, C))
        ubar = sum(us) / len(A)
        Z = sigmas[t] * sum(block[t, k] * G.basis[k] for k in range(D))
        v = ubar + Z
        nv = G.norm(v)
        if nv > B_SAFE:
            v = v * (B_SAFE / nv)
        theta = retract(G, CENTER, G.exp(theta, GAMMA * v), R)
    return G.dist(theta, THETA_STARS[id(A)]), cc

def schedules(T):
    Sfix, sig_c = total_rdp_S(C, T, DELTA, N, EPS)
    w = np.array([KAPPA ** (T - 1 - t) for t in range(T)])
    sig13 = (np.sum(w ** (2 / 3)) / Sfix) ** 0.5 * w ** (-1 / 3)
    res = finite_ceiling_schedule(T, KAPPA, 3, B_SAFE, C, Sfix, SIG_MAX)
    k = max(1, res.active_rounds)
    return {"uniform": np.array([sig_c] * T), "w13": sig13,
            "capped": np.asarray(res.sigmas), "skipped": (np.asarray(res.sigmas), k)}, k

THETA_STARS = {}
ARMS = ["uniform", "w13", "capped", "skipped"]
data = {T: {nm: [] for nm in ARMS} for T in TS}
clip_frac = {T: {nm: [] for nm in ARMS} for T in TS}
for seed in range(NSEED):
    A = anchors_for(seed); THETA_STARS[id(A)] = karcher_mean(G, A)
    for T in TS:
        sch, k = schedules(T)
        rng = np.random.default_rng(50000 + 100 * seed + T)
        blocks = rng.standard_normal((MB, T, D))
        acc = {nm: [] for nm in sch}
        cc_tot = {nm: 0 for nm in sch}; rounds = {nm: 0 for nm in sch}
        for m in range(MB):
            b = blocks[m]
            for nm, sg in sch.items():
                if nm == "skipped":
                    sig_full, kk = sg
                    d, cc = run_block(A, b[T - kk:], sig_full[T - kk:], count_clip=True)
                    rounds[nm] += kk
                else:
                    d, cc = run_block(A, b, sg, count_clip=True)
                    rounds[nm] += T
                cc_tot[nm] += cc
                acc[nm].append(d)
        for nm in sch:
            data[T][nm].append(float(np.sqrt(np.mean(np.square(acc[nm])))))
            clip_frac[T][nm].append(cc_tot[nm] / (rounds[nm] * N))
    print(f"[seed {seed} done]", flush=True)

def paired(a, b):
    d = np.array(a) - np.array(b)
    return d.mean(), (TMUL * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0)

maxrate = max(np.mean(clip_frac[T][nm]) for T in TS for nm in ARMS)
print(f"\nRQ4 (nominal fixed-design): {NSEED} CRN-paired seeds, Mb={MB}, kappa={KAPPA} (nominal), "
      f"eps={EPS}, fixed public center.")
print(f"Per-round client-clip rate at C={C} (support radius R+A=0.75, eta(R+A)=0.375; "
      "inactive when C>0.375):")
for T in TS:
    rates = {nm: 100 * np.mean(clip_frac[T][nm]) for nm in ARMS}
    print(f"  T={T:2d}: " + "  ".join(f"{nm} {rates[nm]:.4f}%" for nm in ARMS))
print(f"  max per-round client-clip rate over all schedules and horizons: {100*maxrate:.4f}%")
print(" T  | uniform |  u - w13        |  u - capped     |  u - skipped    | w13 - capped (resolved?)")
out = {"TS": np.array(TS)}
for T in TS:
    u = data[T]["uniform"]
    m13, h13 = paired(u, data[T]["w13"]); mc, hc = paired(u, data[T]["capped"])
    ms, hs = paired(u, data[T]["skipped"]); mx, hx = paired(data[T]["w13"], data[T]["capped"])
    resolved = "unresolved" if abs(mx) <= hx else f"{mx:+.4f}"
    print(f" {T:2d} | {np.mean(u):.4f}  | {m13:+.4f}+-{h13:.4f} | {mc:+.4f}+-{hc:.4f} | "
          f"{ms:+.4f}+-{hs:.4f} | {resolved}")
    for nm in ARMS:
        out[f"T{T}_{nm}"] = np.array(data[T][nm])
        out[f"T{T}_client_clipfrac_{nm}"] = np.array(clip_frac[T][nm])
np.savez("../results/rq4_schedule_seeds.npz", **out)
print("\nsaved -> results/rq4_schedule_seeds.npz")
print("Nonuniform beats uniform (paired) is the reported result; w13/capped/skipped are "
      "reported as unresolved where their paired interval includes zero.")
