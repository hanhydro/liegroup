import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from real_rotation_averaging import load

HERE = os.path.dirname(os.path.abspath(__file__))

H, TAU_MAX, ETA, GAMMA, T, DELTA_DP, B_POST, M = 0.10, 2.0, 0.5, 1.0, 20, 1e-5, 0.87, 100
EPS = [1.0, 2.0, 4.0, 8.0, 16.0]
DEGREES = [0.0, 5.0, 10.0]
MIN_DEG, NCAM = 120, 16
SEED_AXES, SEED_NOISE = 20261027, 20261028
MARGIN = 0.03


def sigma_rdp(C, T, delta, N, eps):
    Delta = 2.0 * C / N
    Ld = -np.log(delta)
    gap = eps / (np.sqrt(Ld + eps) + np.sqrt(Ld))
    return Delta * np.sqrt(T) / (np.sqrt(2.0) * gap)


def rotvec(Rm):
    shp = Rm.shape[:-2]
    return Rot.from_matrix(Rm.reshape(-1, 3, 3)).as_rotvec().reshape(shp + (3,))


def expmat(v):
    shp = v.shape[:-1]
    return Rot.from_rotvec(v.reshape(-1, 3)).as_matrix().reshape(shp + (3, 3))


def clip_rows(v, radius):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(1.0, n / radius)


def pseudo_huber_steps(theta, meas, eta, h):
    rel = np.einsum("mji,njk->mnik", theta, meas)
    rv = rotvec(rel)
    d = np.linalg.norm(rv, axis=-1, keepdims=True)
    return eta * rv / np.sqrt(1.0 + (d / h) ** 2)


def squared_distance_steps(theta, meas, eta, h=None):
    rel = np.einsum("mji,njk->mnik", theta, meas)
    return eta * rotvec(rel)


def robust_minimizer(meas, theta0, h, iters=600, step=1.0):
    theta = theta0.copy()
    for _ in range(iters):
        g = pseudo_huber_steps(theta[None], meas, 1.0, h)[0].mean(axis=0)
        theta = theta @ expmat(step * g)
    return theta


def run_arm(meas, acc, ref, eta, C, sigma, r, rounds, trials, rng, step_fn=None):
    step_fn = pseudo_huber_steps if step_fn is None else step_fn
    theta = np.repeat(ref[None], trials, axis=0)
    N = meas.shape[0]
    post_clips = 0
    for _ in range(rounds):
        w = clip_rows(step_fn(theta, meas, eta, H), C) * acc[None, :, None]
        v = w.sum(axis=1) / N
        if sigma > 0:
            v = v + sigma * rng.standard_normal(v.shape)
        post_clips += int(np.sum(np.linalg.norm(v, axis=-1) > B_POST))
        v = clip_rows(v, B_POST)
        end = theta @ expmat(GAMMA * v)
        xi = clip_rows(rotvec(np.einsum("ji,mjk->mik", ref, end)), r)
        theta = ref @ expmat(xi)
    return theta, post_clips / (trials * rounds)


def project_so3(X):
    U, _, Vt = np.linalg.svd(X)
    det = np.linalg.det(U @ Vt)
    U[..., :, 2] *= np.sign(det)[..., None]
    return U @ Vt


def run_arm_ambient(meas, acc, ref, eta, C_E, sigma_E, r, rounds, trials, rng):
    theta = np.repeat(ref[None], trials, axis=0)
    N = meas.shape[0]
    hE, BE = np.sqrt(2.0) * H, np.sqrt(2.0) * B_POST
    for _ in range(rounds):
        diff = meas[None] - theta[:, None]
        dF = np.linalg.norm(diff, axis=(-2, -1), keepdims=True)
        w = eta * diff / np.sqrt(1.0 + (dF / hE) ** 2)
        wn = np.linalg.norm(w, axis=(-2, -1), keepdims=True)
        w = w / np.maximum(1.0, wn / C_E) * acc[None, :, None, None]
        v = w.sum(axis=1) / N
        if sigma_E > 0:
            v = v + sigma_E * rng.standard_normal(v.shape)
        vn = np.linalg.norm(v, axis=(-2, -1), keepdims=True)
        v = v / np.maximum(1.0, vn / BE)
        end = project_so3(theta + GAMMA * v)
        xi = clip_rows(rotvec(np.einsum("ji,mjk->mik", ref, end)), r)
        theta = ref @ expmat(xi)
    return theta


def errors_to(theta, star):
    return np.linalg.norm(rotvec(np.einsum("ji,mjk->mik", star, theta)), axis=-1)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    t0 = time.time()
    n, adj, Rhat, nodes = load()
    order = sorted(((len(adj[i]), i) for i in range(len(Rhat))), reverse=True)
    cams = [i for d, i in order if d >= MIN_DEG][:NCAM]
    rng_axes = np.random.default_rng(SEED_AXES)
    axes = {i: (lambda a: a / np.linalg.norm(a))(rng_axes.standard_normal(3)) for i in cams}
    rng_noise = np.random.default_rng(SEED_NOISE)
    rng_amb = np.random.default_rng(SEED_NOISE + 1)
    rng_sq = np.random.default_rng(SEED_NOISE + 2)

    nd, ne, nc = len(DEGREES), len(EPS), len(cams)
    out = dict(
        init_err=np.zeros((nd, nc)), np_iter=np.zeros((nd, nc)), np_oneshot=np.zeros((nd, nc)),
        priv_iter=np.zeros((nd, ne, nc)), priv_oneshot=np.zeros((nd, ne, nc)),
        priv_ambient=np.zeros((nd, ne, nc)), np_ambient=np.zeros((nd, nc)), sigma_ambient=np.zeros((nd, ne, nc)),
        priv_sqoneshot=np.zeros((nd, ne, nc)), np_sqoneshot=np.zeros((nd, nc)), sigma_sqoneshot=np.zeros((nd, ne, nc)),
        priv_iter_q95=np.zeros((nd, ne, nc)), priv_oneshot_q95=np.zeros((nd, ne, nc)),
        sigma_iter=np.zeros((nd, ne, nc)), sigma_oneshot=np.zeros((nd, ne, nc)),
        post_clip_iter=np.zeros((nd, ne, nc)),
        N=np.zeros(nc, int), accepted=np.zeros((nd, nc), int),
        r=np.zeros(nd), rho_star=np.zeros(nd), L_Pi=np.zeros(nd), two_r_plus=np.zeros(nd),
    )
    C_IT, C_OS = ETA * H, 1.0 * H
    for di, deg in enumerate(DEGREES):
        dl = np.radians(deg)
        r = 0.20 + dl
        rho_star = r + GAMMA * B_POST + MARGIN
        q = 0.5 * rho_star
        out["r"][di], out["rho_star"][di] = r, rho_star
        out["L_Pi"][di], out["two_r_plus"][di] = q / np.sin(q), 2 * max(r + ETA * H, rho_star)
        assert rho_star < np.pi and 2 * max(r + ETA * H, rho_star) < np.pi
        for ci, i in enumerate(cams):
            meas = np.stack([Rij @ Rhat[j] for j, Rij in adj[i]])
            N = meas.shape[0]
            out["N"][ci] = N
            ref = Rhat[i] @ expmat(dl * axes[i])
            acc = np.linalg.norm(rotvec(np.einsum("ji,njk->nik", ref, meas)), axis=-1) < TAU_MAX
            out["accepted"][di, ci] = int(acc.sum())
            star = robust_minimizer(meas[acc], Rhat[i], H)
            out["init_err"][di, ci] = errors_to(ref[None], star)[0]
            th, _ = run_arm(meas, acc, ref, ETA, C_IT, 0.0, r, T, 1, rng_noise)
            out["np_iter"][di, ci] = errors_to(th, star)[0]
            th, _ = run_arm(meas, acc, ref, 1.0, C_OS, 0.0, r, 1, 1, rng_noise)
            out["np_oneshot"][di, ci] = errors_to(th, star)[0]
            C_E = np.sqrt(2.0) * C_IT
            th = run_arm_ambient(meas, acc, ref, ETA, C_E, 0.0, r, T, 1, rng_amb)
            out["np_ambient"][di, ci] = errors_to(th, star)[0]
            C_SQ = r
            th, _ = run_arm(meas, acc, ref, 1.0, C_SQ, 0.0, r, 1, 1, rng_sq, squared_distance_steps)
            out["np_sqoneshot"][di, ci] = errors_to(th, star)[0]
            for ei, eps in enumerate(EPS):
                s_it = sigma_rdp(C_IT, T, DELTA_DP, N, eps)
                s_os = sigma_rdp(C_OS, 1, DELTA_DP, N, eps)
                out["sigma_iter"][di, ei, ci], out["sigma_oneshot"][di, ei, ci] = s_it, s_os
                th, pc = run_arm(meas, acc, ref, ETA, C_IT, s_it, r, T, M, rng_noise)
                e = errors_to(th, star)
                out["priv_iter"][di, ei, ci], out["priv_iter_q95"][di, ei, ci] = rms(e), float(np.quantile(e, 0.95))
                out["post_clip_iter"][di, ei, ci] = pc
                th, _ = run_arm(meas, acc, ref, 1.0, C_OS, s_os, r, 1, M, rng_noise)
                e = errors_to(th, star)
                out["priv_oneshot"][di, ei, ci], out["priv_oneshot_q95"][di, ei, ci] = rms(e), float(np.quantile(e, 0.95))
                s_E = sigma_rdp(C_E, T, DELTA_DP, N, eps)
                out["sigma_ambient"][di, ei, ci] = s_E
                th = run_arm_ambient(meas, acc, ref, ETA, C_E, s_E, r, T, M, rng_amb)
                out["priv_ambient"][di, ei, ci] = rms(errors_to(th, star))
                s_SQ = sigma_rdp(C_SQ, 1, DELTA_DP, N, eps)
                out["sigma_sqoneshot"][di, ei, ci] = s_SQ
                th, _ = run_arm(meas, acc, ref, 1.0, C_SQ, s_SQ, r, 1, M, rng_sq, squared_distance_steps)
                out["priv_sqoneshot"][di, ei, ci] = rms(errors_to(th, star))
            print(f"deg={deg:4.1f} cam={i:3d} N={N} acc={acc.sum()} init={out['init_err'][di,ci]:.4f} "
                  f"np_it={out['np_iter'][di,ci]:.2e} np_os={out['np_oneshot'][di,ci]:.4f} "
                  f"priv_it={np.round(out['priv_iter'][di,:,ci],4)} priv_os={np.round(out['priv_oneshot'][di,:,ci],4)}",
                  flush=True)

    tmult = 2.131
    summary = dict(public_settings=dict(h=H, tau_max=TAU_MAX, eta=ETA, gamma=GAMMA, T=T, delta_dp=DELTA_DP,
                                        B=B_POST, trials=M, client_clip_iter=C_IT, client_clip_oneshot=C_OS,
                                        degrees=DEGREES, eps=EPS, cameras=[int(nodes[i]) for i in cams],
                                        seed_axes=SEED_AXES, seed_noise=SEED_NOISE),
                   per_degree=[])
    print("\n=== cross-camera means (rad); paired difference = iterative minus one-shot ===")
    for di, deg in enumerate(DEGREES):
        row = dict(degrees=deg, r=float(out["r"][di]), rho_star=float(out["rho_star"][di]),
                   L_Pi=float(out["L_Pi"][di]), two_r_plus=float(out["two_r_plus"][di]),
                   init_err_mean=float(out["init_err"][di].mean()),
                   np_iter_mean=float(out["np_iter"][di].mean()), np_oneshot_mean=float(out["np_oneshot"][di].mean()),
                   np_ambient_mean=float(out["np_ambient"][di].mean()),
                   np_sqoneshot_mean=float(out["np_sqoneshot"][di].mean()),
                   accepted_median=float(np.median(out["accepted"][di])), eps=[])
        print(f"deg={deg}: r={row['r']:.4f} rho*={row['rho_star']:.4f} L_Pi={row['L_Pi']:.4f} "
              f"init={row['init_err_mean']:.4f} nonpriv iter={row['np_iter_mean']:.2e} nonpriv one-shot={row['np_oneshot_mean']:.4f}")
        for ei, eps in enumerate(EPS):
            it, os_ = out["priv_iter"][di, ei], out["priv_oneshot"][di, ei]
            diff = it - os_
            hw = tmult * diff.std(ddof=1) / np.sqrt(nc)
            amb = out["priv_ambient"][di, ei]
            rec = dict(eps=eps, iter_mean=float(it.mean()), oneshot_mean=float(os_.mean()),
                       ambient_mean=float(amb.mean()), ambient_over_iter=float(amb.mean() / it.mean()),
                       iter_below_ambient_cams=int(np.sum(it < amb)),
                       sqoneshot_mean=float(out["priv_sqoneshot"][di, ei].mean()),
                       iter_below_sqoneshot_cams=int(np.sum(it < out["priv_sqoneshot"][di, ei])),
                       sq_paired_diff_mean=float((it - out["priv_sqoneshot"][di, ei]).mean()),
                       sq_paired_diff_halfwidth=float(tmult * (it - out["priv_sqoneshot"][di, ei]).std(ddof=1) / np.sqrt(nc)),
                       paired_diff_mean=float(diff.mean()), paired_diff_halfwidth=float(hw),
                       iter_below_oneshot_cams=int(np.sum(it < os_)), iter_below_init_cams=int(np.sum(it < out["init_err"][di])),
                       sigma_iter_median=float(np.median(out["sigma_iter"][di, ei])),
                       sigma_oneshot_median=float(np.median(out["sigma_oneshot"][di, ei])),
                       post_clip_rate_max=float(out["post_clip_iter"][di, ei].max()))
            row["eps"].append(rec)
            print(f"  eps={eps:4.1f} iter={rec['iter_mean']:.4f} one-shot={rec['oneshot_mean']:.4f} ambient={rec['ambient_mean']:.4f} sq-one-shot={rec['sqoneshot_mean']:.4f} iter<sq {rec['iter_below_sqoneshot_cams']}/16 "
                  f"diff={rec['paired_diff_mean']:+.4f}+-{hw:.4f} iter<one-shot {rec['iter_below_oneshot_cams']}/16 "
                  f"iter<init {rec['iter_below_init_cams']}/16 sigma_it={rec['sigma_iter_median']:.4f}")
        summary["per_degree"].append(row)
    np.savez(os.path.join(HERE, "..", "results", "robust_camera_task.npz"), **out, eps=np.array(EPS), degrees=np.array(DEGREES))
    with open(os.path.join(HERE, "..", "results", "robust_camera_task.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"saved robust_camera_task.npz/.json  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
