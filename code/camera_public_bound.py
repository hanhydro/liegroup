import json
from pathlib import Path

import numpy as np
from scipy.stats import ncx2

HERE = Path(__file__).resolve().parent
P = dict(r=0.20, eta=0.5, gamma=1.0, T=20, C=0.20, C_one=0.20, B=0.87, rho=1.10, delta=1e-5,
         K=0.25, d=3, A0=0.20, R_star=0.40, trials=100)


def sigma(C, T, N, eps):
    Ld = -np.log(P["delta"]); D = np.sqrt(Ld + eps) - np.sqrt(Ld)
    return np.sqrt(2) * C * np.sqrt(T) / (N * D)


def constants(q0):
    r, sk = P["r"], np.sqrt(P["K"])
    mu = q0 * (sk * 2 * r) / np.tan(sk * 2 * r)
    alpha, L = P["gamma"] * P["eta"], 1.0
    x = sk * alpha * L * P["R_star"]; beta = np.sin(x) / x - np.cos(x)
    kappa_pop = max(abs(1 - alpha * mu), abs(1 - alpha * L)) + beta
    kappa = max(abs(1 - alpha * mu), abs(1 - alpha * L))
    kappa_w = max(abs(1 - alpha * mu), abs(1 - alpha * L))
    x1 = sk * 1.0 * L * P["R_star"]; beta1 = np.sin(x1) / x1 - np.cos(x1)
    kappa1_pop = max(abs(1 - mu), abs(1 - L)) + beta1
    kappa1 = max(abs(1 - mu), abs(1 - L))
    kappa1_w = max(abs(1 - mu), abs(1 - L))
    r_plus = max(r + alpha * 2 * r, P["rho"])
    slacks = dict(client_clip=P["C"] - P["eta"] * 2 * r + 1e-15, clip_order=P["B"] - P["C"],
                  containment=P["rho"] - r - P["gamma"] * P["B"], geometry=np.pi - 2 * r_plus,
                  target_inside_clip=P["gamma"] * P["B"] - P["R_star"],
                  tube=(r_plus - r) - alpha * L * P["R_star"], tube_one=(r_plus - r) - P["R_star"],
                  contraction=1 - kappa, contraction_w=1 - kappa_w)
    assert all(v > 0 for v in slacks.values()), slacks
    return dict(q0=q0, mu=mu, kappa=kappa, kappa_w=kappa_w, kappa_one=kappa1, kappa_one_w=kappa1_w,
                kappa_pop=kappa_pop, kappa_one_pop=kappa1_pop, beta=beta, r_plus=r_plus, slacks={k: float(v) for k, v in slacks.items()})


def bounds(c, N, eps):
    T, d, A0, Rs = P["T"], P["d"], P["A0"], P["R_star"]
    s = sigma(P["C"], T, N, eps); s1 = sigma(P["C_one"], 1, N, eps)
    k, kw = c["kappa"], c["kappa_w"]
    H = (1 - k ** T) / (1 - k)
    thm71 = np.minimum(Rs, k ** T * A0 + H * np.sqrt(d) * s)
    vT = s ** 2 * (1 - kw ** (2 * T)) / (1 - kw ** 2)
    sq = np.minimum(Rs, np.sqrt(kw ** (2 * T) * A0 ** 2 + d * vT))
    q95 = np.minimum(Rs, np.sqrt(vT * ncx2.ppf(0.95, d, A0 ** 2 * kw ** (2 * T) / vT)))
    one71 = np.minimum(Rs, c["kappa_one"] * A0 + np.sqrt(d) * s1)
    one_sq = np.minimum(Rs, np.sqrt(c["kappa_one_w"] ** 2 * A0 ** 2 + d * s1 ** 2))
    return thm71, sq, q95, one71, one_sq


def main():
    R = np.load(HERE.parent / "results" / "real_ra_results.npz")
    Q = np.load(HERE.parent / "results" / "rq3_oneshot_baseline.npz")
    rec = json.loads((HERE.parent / "results" / "real_ra_edge_records.json").read_text())
    N = R["degrees"].astype(float); acc = R["naccepted"].astype(float)
    ids = [c["camera_id"] for c in rec["cameras"]]
    frac = acc / N
    obs_it, obs_os, eps = Q["priv_iter_C0.2"], Q["priv_oneshot_C0.2"], Q["eps"]
    out = dict(public=P, cameras=ids, degrees=N.tolist(), accepted_fraction=frac.tolist(),
               min_accepted_fraction=float(frac.min()), classes={})
    for q0 in (0.75, 0.80):
        c = constants(q0)
        member = frac >= q0
        rows = []
        for i, e in enumerate(eps):
            t71, sq, q95, o71, osq = (b[member] for b in bounds(c, N, e))
            oi, oo = obs_it[i][member], obs_os[i][member]
            z = (oi ** 2 - sq ** 2) / (sq ** 2 * np.sqrt(2 / 3 / P["trials"]))
            def picks(one, many):
                lab = np.where(one < many, "one_step", f"T{P['T']}")
                lab[P["A0"] < np.minimum(one, many)] = "initializer"
                return {k: int(v) for k, v in zip(*np.unique(lab, return_counts=True))}
            rows.append(dict(eps=float(e), observed_iterative=float(oi.mean()), observed_oneshot=float(oo.mean()),
                             thm71_bound=float(t71.mean()), squared_bound=float(sq.mean()),
                             squared_q95_bound=float(q95.mean()),
                             ratio_thm71=float(t71.mean() / oi.mean()), ratio_squared=float(sq.mean() / oi.mean()),
                             per_camera_ratio_squared_min=float((sq / oi).min()),
                             per_camera_ratio_squared_max=float((sq / oi).max()),
                             max_standardized_excess=float(z.max()),
                             oneshot_thm71_bound=float(o71.mean()), oneshot_squared_bound=float(osq.mean()),
                             per_camera_pick_terminal=picks(o71, t71), per_camera_pick_squared=picks(osq, sq)))
        out["classes"][f"{q0:.2f}"] = dict(constants=c, cameras_in_class=int(member.sum()),
                                          excluded=[ids[j] for j in np.where(~member)[0]], by_budget=rows)
        print(f"q0={q0}: cameras={member.sum()} mu={c['mu']:.10f} kappa={c['kappa']:.6f} kappa_w={c['kappa_w']:.6f}")
        for r_ in rows:
            print(f"  eps={r_['eps']:g}: observed={r_['observed_iterative']:.5f} thm71={r_['thm71_bound']:.5f} ({r_['ratio_thm71']:.2f}x) "
                  f"squared={r_['squared_bound']:.5f} ({r_['ratio_squared']:.3f}x, per-camera {r_['per_camera_ratio_squared_min']:.3f}-{r_['per_camera_ratio_squared_max']:.3f}, "
                  f"max z={r_['max_standardized_excess']:+.2f}) q95={r_['squared_q95_bound']:.5f}")
    (HERE.parent / "results" / "camera_public_bound.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
