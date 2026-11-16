import json
import os

import numpy as np

from real_rotation_averaging import load
from robust_camera_task import (DEGREES, DELTA_DP, EPS, GAMMA, H, M, MIN_DEG, NCAM, SEED_AXES,
                                SEED_NOISE, TAU_MAX, errors_to, expmat, rms, robust_minimizer,
                                rotvec, run_arm, sigma_rdp)

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = [1.0, 1.5, 2.0, 3.0]


def main():
    n, adj, Rhat, nodes = load()
    order = sorted(((len(adj[i]), i) for i in range(len(Rhat))), reverse=True)
    cams = [i for d, i in order if d >= MIN_DEG][:NCAM]
    rng_axes = np.random.default_rng(SEED_AXES)
    axes = {i: (lambda a: a / np.linalg.norm(a))(rng_axes.standard_normal(3)) for i in cams}
    rng = np.random.default_rng(SEED_NOISE + 3)
    nd, ne, nc, na = len(DEGREES), len(EPS), len(cams), len(ALPHAS)
    noiseless = np.zeros((nd, na, nc))
    private = np.zeros((nd, na, ne, nc))
    init_err = np.zeros((nd, nc))
    for di, deg in enumerate(DEGREES):
        dl = np.radians(deg)
        r = 0.20 + dl
        for ci, i in enumerate(cams):
            meas = np.stack([Rij @ Rhat[j] for j, Rij in adj[i]])
            N = meas.shape[0]
            ref = Rhat[i] @ expmat(dl * axes[i])
            acc = np.linalg.norm(rotvec(np.einsum("ji,njk->nik", ref, meas)), axis=-1) < TAU_MAX
            star = robust_minimizer(meas[acc], Rhat[i], H)
            init_err[di, ci] = errors_to(ref[None], star)[0]
            for ai, a in enumerate(ALPHAS):
                C = a * H
                th, _ = run_arm(meas, acc, ref, a, C, 0.0, r, 1, 1, rng)
                noiseless[di, ai, ci] = errors_to(th, star)[0]
                for ei, eps in enumerate(EPS):
                    s = sigma_rdp(C, 1, DELTA_DP, N, eps)
                    th, _ = run_arm(meas, acc, ref, a, C, s, r, 1, M, rng)
                    private[di, ai, ei, ci] = rms(errors_to(th, star))
            print(f"deg={deg:4.1f} cam={i:3d} init={init_err[di, ci]:.4f} "
                  f"noiseless={np.round(noiseless[di, :, ci], 4)} eps8={np.round(private[di, :, 3, ci], 4)}",
                  flush=True)
    old = np.load(os.path.join(HERE, "..", "results", "robust_camera_task.npz"))
    summary = dict(alphas=ALPHAS, eps=EPS, degrees=DEGREES, gamma=GAMMA, h=H, trials=M, per_degree=[])
    for di, deg in enumerate(DEGREES):
        row = dict(degrees=deg, init_err_mean=float(init_err[di].mean()),
                   noiseless_mean=[float(noiseless[di, ai].mean()) for ai in range(na)],
                   private_mean=[[float(private[di, ai, ei].mean()) for ei in range(ne)] for ai in range(na)],
                   iterative_mean=[float(old["priv_iter"][di, ei].mean()) for ei in range(ne)],
                   tangent_mean=[float(old["priv_sqoneshot"][di, ei].mean()) for ei in range(ne)],
                   best_alpha=[ALPHAS[int(np.argmin([private[di, ai, ei].mean() for ai in range(na)]))]
                               for ei in range(ne)],
                   iter_below_best_step_cams=[int(np.sum(old["priv_iter"][di, ei]
                                                         < private[di, :, ei].min(axis=0)))
                                              for ei in range(ne)])
        summary["per_degree"].append(row)
        print(f"deg={deg}: noiseless {np.round(row['noiseless_mean'], 4)}")
        for ei, eps in enumerate(EPS):
            print(f"  eps={eps:4.1f} steps {np.round([row['private_mean'][ai][ei] for ai in range(na)], 4)} "
                  f"iter {row['iterative_mean'][ei]:.4f} tangent {row['tangent_mean'][ei]:.4f}")
    np.savez(os.path.join(HERE, "..", "results", "robust_onestep_sweep.npz"), noiseless=noiseless, private=private,
             init_err=init_err, alphas=np.array(ALPHAS), eps=np.array(EPS), degrees=np.array(DEGREES))
    with open(os.path.join(HERE, "..", "results", "robust_onestep_sweep.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
