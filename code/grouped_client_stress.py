import os
import numpy as np
from real_rotation_averaging import load, build_cameras, sigma_rdp, GRP
from fedopt_stability import clip, retract

HERE = os.path.dirname(os.path.abspath(__file__))
TAU, ETA, GAMMA, T, DELTA, M, EPS, C = 0.20, 0.5, 1.0, 20, 1e-5, 100, 4.0, 0.20
B_SAFE = 0.87
GROUPS = [1, 2, 4, 8, 16]
ZERO = 0.0 * GRP.basis[0]


def group_disp(theta, grp, C):
    g_full = len(grp)
    s = ZERO
    for m, a in grp:
        if a:
            s = s + GRP.log(theta, m)
    agg = ETA * s / g_full
    fired = GRP.norm(agg) > C
    return clip(GRP, agg, C), fired


def run_grouped(groups, theta_ref, C, sigma, r, N):
    theta = theta_ref.copy(); fires = 0
    for _ in range(T):
        tot = ZERO
        for grp in groups:
            d, f = group_disp(theta, grp, C)
            tot = tot + (len(grp) / N) * d
            fires += int(f)
        vhat = tot + GRP.sample_tangent(sigma)
        nv = GRP.norm(vhat)
        if nv > B_SAFE:
            vhat = vhat * (B_SAFE / nv)
        theta = retract(GRP, theta_ref, GRP.exp(theta, GAMMA * vhat), r)
    return GRP.dist(theta, THETA_STAR), fires


n, adj, Rhat, nodes = load()
cams = build_cameras(adj, Rhat, TAU)
print(f"Ellis_Island: {len(cams)} cameras; synthetic multi-record privacy-unit stress test, "
      f"eps={EPS}, C={C}, T={T}, M={M}, matched half-step eta={ETA}, weight |g|/N.\n")
print(" g | records/client | N_g (med [min,max]) | group sizes (min,max) | N_eff (med) | "
      "sigma (med) | group-clip% | mean RMS | RMS p95 | p99 | client contacts")
out = {"g": np.array(GROUPS), "eps": EPS, "C": C, "eta": ETA}
for g in GROUPS:
    rms_cam = []; sig_cam = []; ng_cam = []; clipf_cam = []; neff_cam = []
    errs = np.zeros((len(cams), M)); gsize_min = 10**9; gsize_max = 0
    for ci, c in enumerate(cams):
        cl = c["clients"]; N = len(cl)
        groups = [cl[i:i + g] for i in range(0, len(cl), g)]
        Ng = len(groups); ng_cam.append(Ng)
        maxg = max(len(x) for x in groups)
        gsize_min = min(gsize_min, min(len(x) for x in groups))
        gsize_max = max(gsize_max, max(len(x) for x in groups))
        N_eff = N / maxg
        neff_cam.append(N_eff)
        sigma = sigma_rdp(C, T, DELTA, N_eff, EPS); sig_cam.append(sigma)
        global THETA_STAR; THETA_STAR = c["theta"]
        fsum = 0
        for m in range(M):
            d, f = run_grouped(groups, c["ref"], C, sigma, c["r"], N); errs[ci, m] = d; fsum += f
        rms_cam.append(float(np.sqrt(np.mean(errs[ci] ** 2))))
        clipf_cam.append(fsum / (M * T * Ng))
    rms_cam = np.array(rms_cam)
    ng_median = float(np.median(ng_cam))
    contacts = ng_median * T
    out[f"rms_g{g}"] = rms_cam
    out[f"Ng_g{g}"] = np.array(ng_cam)
    out[f"Neff_g{g}"] = np.array(neff_cam)
    out[f"sigma_g{g}"] = np.array(sig_cam)
    out[f"client_clipfrac_g{g}"] = np.array(clipf_cam)
    out[f"errs_g{g}"] = errs
    out[f"gsize_g{g}"] = np.array([gsize_min, gsize_max])
    print(f" {g:2d} | {g:14d} | {ng_median:5.1f} [{min(ng_cam)},{max(ng_cam)}] | "
          f"({gsize_min},{gsize_max}) | {int(np.median(neff_cam)):5d} | {np.median(sig_cam):.4f} | "
          f"{100*np.mean(clipf_cam):9.0f}%  | {rms_cam.mean():.4f}   | "
          f"{np.percentile(rms_cam,95):.4f}  | {np.percentile(rms_cam,99):.4f} | {contacts:.0f}")

np.savez(os.path.join(HERE, "..", "results", "grouped_client_stress.npz"), **out)
print("\nsaved -> results/grouped_client_stress.npz")
print("Reported dispersion is across the 16 per-camera RMS summaries (descriptive cross-camera, "
      "not run-level tail error). The synthetic groups are a privacy-unit stress test, not a "
      "natural user or device model.")
