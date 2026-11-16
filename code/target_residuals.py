import json
import os

import numpy as np

from fedopt_stability import karcher_mean, so3
from real_rotation_averaging import accepted, candidate_clients, load
from robust_camera_task import (DEGREES, H, MIN_DEG, NCAM, SEED_AXES, TAU_MAX, errors_to, expmat,
                                pseudo_huber_steps, robust_minimizer, rotvec)

HERE = os.path.dirname(os.path.abspath(__file__))
G = so3()
CENTER = G.I.copy()


def karcher_residuals(anchors, extra=3000):
    t = karcher_mean(G, anchors)
    g = G.norm(sum(G.log(t, a) for a in anchors) / len(anchors))
    t2 = t.copy()
    for _ in range(extra):
        t2 = G.exp(t2, 0.5 * sum(G.log(t2, a) for a in anchors) / len(anchors))
    return float(g), float(G.dist(t, t2))


def rq1_anchors():
    rng = np.random.default_rng(20260810)
    out = []
    for _ in range(8):
        z = rng.normal(size=G.dim)
        v = sum(z[k] * G.basis[k] for k in range(G.dim))
        n = G.norm(v)
        v = v * (rng.uniform(0, 0.15) / n) if n > 0 else v * 0
        out.append(G.exp(CENTER, v))
    return out


def rq4_anchors(seed):
    rng = np.random.default_rng(1000 + seed)
    out = []
    for _ in range(200):
        z = rng.normal(size=G.dim)
        v = sum(z[k] * G.basis[k] for k in range(G.dim))
        nn = G.norm(v)
        v = v * (rng.uniform(0, 0.15) / nn) if nn > 0 else v
        out.append(G.exp(CENTER, v))
    return out


def main():
    rep = {}
    g, d = karcher_residuals(rq1_anchors())
    rep["declared_domain"] = dict(grad_norm=g, refinement_shift=d)
    res = [karcher_residuals(rq4_anchors(s), extra=300) for s in range(20)]
    rep["allocation"] = dict(grad_norm_max=max(r[0] for r in res), refinement_shift_max=max(r[1] for r in res))

    n, adj, Rhat, nodes = load()
    cand = [(i, candidate_clients(i, adj, Rhat, 0.20)) for i in range(len(Rhat))]
    cand = [(i, cl) for i, cl in cand if len(cl) >= 120]
    cand.sort(key=lambda t: -len(t[1]))
    res = [karcher_residuals(accepted(cl), extra=300) for i, cl in cand[:16]]
    rep["camera_squared"] = dict(grad_norm_max=max(r[0] for r in res), refinement_shift_max=max(r[1] for r in res))

    order = sorted(((len(adj[i]), i) for i in range(len(Rhat))), reverse=True)
    cams = [i for d_, i in order if d_ >= MIN_DEG][:NCAM]
    rng_axes = np.random.default_rng(SEED_AXES)
    axes = {i: (lambda a: a / np.linalg.norm(a))(rng_axes.standard_normal(3)) for i in cams}
    rows = []
    for deg in DEGREES:
        dl = np.radians(deg)
        for i in cams:
            meas = np.stack([Rij @ Rhat[j] for j, Rij in adj[i]])
            ref = Rhat[i] @ expmat(dl * axes[i])
            acc = np.linalg.norm(rotvec(np.einsum("ji,njk->nik", ref, meas)), axis=-1) < TAU_MAX
            star = robust_minimizer(meas[acc], Rhat[i], H)
            g = float(np.linalg.norm(pseudo_huber_steps(star[None], meas[acc], 1.0, H)[0].mean(axis=0)))
            star2 = robust_minimizer(meas[acc], star, H, iters=5000)
            rows.append((g, float(errors_to(star2[None], star)[0]), float(errors_to(ref[None], star)[0]),
                         0.20 + dl))
    rep["camera_pseudo_huber"] = dict(grad_norm_max=max(r[0] for r in rows),
                                      refinement_shift_max=max(r[1] for r in rows),
                                      target_to_center_max_by_turn=[max(r[2] for r in rows[k * 16:(k + 1) * 16])
                                                                    for k in range(3)],
                                      working_radius_by_turn=[rows[k * 16][3] for k in range(3)])
    with open(os.path.join(HERE, "..", "results", "target_residuals.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
