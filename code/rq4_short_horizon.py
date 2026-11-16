import json
import os

import numpy as np
from scipy.stats import t as _tdist

from capped_schedule import finite_ceiling_schedule
from fedopt_stability import clip, karcher_mean, retract, so3

HERE = os.path.dirname(os.path.abspath(__file__))
N, C, DELTA, EPS, R = 200, 0.4, 1e-5, 1.0, 0.6
ETA, GAMMA, KAPPA_DESIGN = 0.5, 1.0, 0.52
B_SAFE, SIG_MAX, RHO_STAR = 0.87, 1e4, 1.50
R_DATA, C1 = 0.15, 0.15
K_MAX = 0.25
NSEED, MB = 20, 30
HALF_TS = [1, 2, 3, 4, 8]
OLD_TS = [8, 16, 32, 64]
TMUL = float(_tdist.ppf(0.975, NSEED - 1))

G = so3()
D = G.dim
CENTER = G.I.copy()


def sigma_rdp(clip_radius, rounds):
    sens = 2.0 * clip_radius / N
    ld = -np.log(DELTA)
    gap = EPS / (np.sqrt(ld + EPS) + np.sqrt(ld))
    return sens * np.sqrt(rounds) / (np.sqrt(2.0) * gap)


def anchors_for(seed):
    rng = np.random.default_rng(1000 + seed)
    out = []
    for _ in range(N):
        z = rng.normal(size=D)
        v = sum(z[k] * G.basis[k] for k in range(D))
        nn = G.norm(v)
        v = v * (rng.uniform(0, 0.15) / nn) if nn > 0 else v
        out.append(G.exp(CENTER, v))
    return out


def run(anchors, target, block, sigmas, eta, client_clip):
    theta = CENTER.copy()
    for t, s in enumerate(sigmas):
        us = [clip(G, G.log(theta, G.exp(theta, eta * G.log(theta, a))), client_clip) for a in anchors]
        v = sum(us) / len(anchors) + s * sum(block[t, k] * G.basis[k] for k in range(D))
        nv = G.norm(v)
        if nv > B_SAFE:
            v = v * (B_SAFE / nv)
        theta = retract(G, CENTER, G.exp(theta, GAMMA * v), R)
    return G.dist(theta, target)


S = 1.0 / sigma_rdp(C, 1) ** 2
x = np.sqrt(K_MAX) * (R + R_DATA)
MU, L_SMOOTH = x / np.tan(x), 1.0
A0, R_STAR = R_DATA, R + R_DATA
assert R_STAR <= GAMMA * B_SAFE < RHO_STAR - R
KW = max(abs(1 - ETA * MU), abs(1 - ETA * L_SMOOTH))
K1 = max(abs(1 - MU), abs(1 - L_SMOOTH))
RCAP, RCAP1 = np.hypot(C, B_SAFE), np.hypot(C1, B_SAFE)


def variance_schedule(T):
    h = sum(KW ** j for j in range(T))
    return np.array([np.sqrt(h / S * KW ** (-(T - 1 - t))) for t in range(T)])


def bounds_iterative(sigmas):
    T = len(sigmas)
    w = np.array([KW ** (T - 1 - t) for t in range(T)])
    rms_bound = KW ** T * A0 + float(np.sum(w * np.minimum(sigmas * np.sqrt(D), RCAP)))
    sq_bound = np.sqrt(KW ** (2 * T) * A0 ** 2 + D * float(np.sum(w ** 2 * sigmas ** 2)))
    return min(rms_bound, R_STAR), min(sq_bound, R_STAR)


def bounds_one_step(sigma1):
    rms_bound = K1 * A0 + min(sigma1 * np.sqrt(D), RCAP1)
    sq_bound = np.sqrt(K1 ** 2 * A0 ** 2 + D * sigma1 ** 2)
    sk = np.sqrt(K_MAX)
    ell = sk * R_DATA / np.sin(sk * R_DATA)
    lam = 2 * sk * R_DATA / np.tan(2 * sk * R_DATA)
    d_tan = (0.25 + ell) * K_MAX * R_DATA ** 3 / lam
    tan_rms = d_tan + min(sigma1 * np.sqrt(D), RCAP1)
    tan_sq = np.sqrt((ell * d_tan) ** 2 + D * sigma1 ** 2)
    return dict(terminal=min(rms_bound, R_STAR), squared=min(sq_bound, R_STAR),
                tangent_terminal=min(tan_rms, R_STAR), tangent_squared=min(tan_sq, R_STAR), d_tan=d_tan)


def old_schedules(T):
    sig_c = sigma_rdp(C, T)
    w = np.array([KAPPA_DESIGN ** (T - 1 - t) for t in range(T)])
    sig13 = (np.sum(w ** (2 / 3)) / S) ** 0.5 * w ** (-1 / 3)
    res = finite_ceiling_schedule(T, KAPPA_DESIGN, 3, B_SAFE, C, S, SIG_MAX)
    return {"uniform": np.full(T, sig_c), "w13": sig13, "capped": np.asarray(res.sigmas)}


def main():
    sigma1 = sigma_rdp(C1, 1)
    designs = ["retain", "one_step"] + [f"half_{T}" for T in HALF_TS]
    rms_rep = {d: [] for d in designs}
    init_dist = []
    for seed in range(NSEED):
        anchors = anchors_for(seed)
        target = karcher_mean(G, anchors)
        d0 = G.dist(CENTER, target)
        init_dist.append(d0)
        rms_rep["retain"].append(d0)
        rng = np.random.default_rng(70000 + seed)
        blocks = rng.standard_normal((MB, max(HALF_TS), D))
        errs = [run(anchors, target, blocks[m], [sigma1], 1.0, C1) for m in range(MB)]
        rms_rep["one_step"].append(float(np.sqrt(np.mean(np.square(errs)))))
        for T in HALF_TS:
            sig = variance_schedule(T)
            errs = [run(anchors, target, blocks[m, :T], sig, ETA, C) for m in range(MB)]
            rms_rep[f"half_{T}"].append(float(np.sqrt(np.mean(np.square(errs)))))
        print(f"[seed {seed}] init {d0:.4f} one-step {rms_rep['one_step'][-1]:.4f} "
              + " ".join(f"T{T} {rms_rep[f'half_{T}'][-1]:.4f}" for T in HALF_TS), flush=True)

    ob = bounds_one_step(sigma1)
    report = dict(public=dict(N=N, C=C, C1=C1, delta=DELTA, eps=EPS, r=R, B=B_SAFE, rho_star=RHO_STAR,
                              R_data=R_DATA, A0=A0, R_star=R_STAR, mu=MU, kappa_w=KW, kappa_1=K1,
                              S=S, sigma_one_step=sigma1, replications=NSEED, trajectories=MB),
                  designs={})
    old = np.load(os.path.join(HERE, "..", "results", "rq4_schedule_seeds.npz"))
    for d in designs:
        v = np.array(rms_rep[d])
        rec = dict(observed=float(v.mean()))
        if d == "retain":
            rec.update(releases=0, terminal=A0, squared=A0)
        elif d == "one_step":
            rec.update(releases=1, **ob)
        else:
            T = int(d.split("_")[1])
            b = bounds_iterative(variance_schedule(T))
            rec.update(releases=T, terminal=b[0], squared=b[1])
        report["designs"][d] = rec
    for T in OLD_TS:
        for nm, sig in old_schedules(T).items():
            b = bounds_iterative(sig)
            report["designs"][f"{nm}_{T}"] = dict(releases=T, observed=float(old[f"T{T}_{nm}"].mean()),
                                                 terminal=b[0], squared=b[1])
    base = np.array(rms_rep["retain"])
    for d in designs[1:]:
        diff = np.array(rms_rep[d]) - base
        report["designs"][d]["minus_retain"] = [float(diff.mean()), float(TMUL * diff.std(ddof=1) / np.sqrt(NSEED))]
    diff = np.array(rms_rep["half_8"]) - old["T8_w13"]
    report["half_8_minus_w13_8"] = [float(diff.mean()), float(TMUL * diff.std(ddof=1) / np.sqrt(NSEED))]
    report["initializer_distance"] = dict(rms=float(np.sqrt(np.mean(np.square(init_dist)))),
                                          max=float(np.max(init_dist)), mean=float(np.mean(init_dist)))
    for key in ("terminal", "squared"):
        pick = min(report["designs"], key=lambda k: report["designs"][k][key])
        report[f"public_pick_{key}"] = pick
    pick_tan = min(["retain", "one_step"] + [f"half_{T}" for T in HALF_TS],
                   key=lambda k: report["designs"][k].get("tangent_squared", report["designs"][k]["squared"]))
    report["public_pick_tangent_squared"] = pick_tan
    report["observed_best"] = min(report["designs"], key=lambda k: report["designs"][k]["observed"])

    np.savez(os.path.join(HERE, "..", "results", "rq4_short_horizon.npz"), init_dist=np.array(init_dist),
             **{d: np.array(rms_rep[d]) for d in designs})
    with open(os.path.join(HERE, "..", "results", "rq4_short_horizon.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
