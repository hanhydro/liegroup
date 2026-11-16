import os
import numpy as np
from real_rotation_averaging import (load, build_cameras, sigma_rdp, rms_to_star, GRP,
                                      client_disp, accepted, retract)

HERE = os.path.dirname(os.path.abspath(__file__))
TAU, ETA_IT, GAMMA, TAU_LOC, T_IT, DELTA, M = 0.20, 0.5, 1.0, 1, 20, 1e-5, 100
B_SAFE = 0.87
EPS = [1.0, 2.0, 4.0, 8.0, 16.0]
C_LIST = [0.20, 0.30]

n, adj, Rhat, nodes = load()
cams = build_cameras(adj, Rhat, TAU)
ncam = len(cams)
anchor = np.array([GRP.dist(c["theta"], c["ref"]) for c in cams])
print(f"Ellis_Island: {ncam} cameras; anchor mean RMS={anchor.mean():.4f} rad "
      f"(median={np.median(anchor):.4f})\n")

def arm(c, eta, gamma, tau_loc, C, sigma, T, reps):
    return rms_to_star(c["clients"], c["theta"], c["ref"], c["r"], eta, gamma, tau_loc,
                       C, sigma, T, reps, B=B_SAFE)[0]

out = {"eps": np.array(EPS), "C_list": np.array(C_LIST), "anchor": anchor}
for C in C_LIST:
    npr_it = np.array([arm(c, ETA_IT, GAMMA, TAU_LOC, C, 0.0, T_IT, 1) for c in cams])
    npr_os = np.array([arm(c, 1.0, 1.0, 1, C, 0.0, 1, 1) for c in cams])
    pr_it = np.zeros((len(EPS), ncam)); pr_os = np.zeros((len(EPS), ncam))
    for ei, eps in enumerate(EPS):
        for ci, c in enumerate(cams):
            s_it = sigma_rdp(C, T_IT, DELTA, c["N"], eps)
            s_os = sigma_rdp(C, 1, DELTA, c["N"], eps)
            pr_it[ei, ci] = arm(c, ETA_IT, GAMMA, TAU_LOC, C, s_it, T_IT, M)
            pr_os[ei, ci] = arm(c, 1.0, 1.0, 1, C, s_os, 1, M)
    out[f"nonpriv_iter_C{C}"] = npr_it
    out[f"nonpriv_oneshot_C{C}"] = npr_os
    out[f"priv_iter_C{C}"] = pr_it
    out[f"priv_oneshot_C{C}"] = pr_os
    tag = "PRIMARY" if C == 0.20 else "secondary"
    print(f"=== C={C} ({tag}) : cross-camera mean RMS to theta* (rad) ===")
    print(f"  nonprivate iterative (sigma=0,T=20): {npr_it.mean():.4f}   "
          f"nonprivate one-shot (sigma=0,T=1): {npr_os.mean():.4f}")
    print("  eps | private iterative | private one-shot | ratio it/os | one-shot beats anchor")
    for ei, eps in enumerate(EPS):
        it_m = pr_it[ei].mean(); os_m = pr_os[ei].mean()
        print(f"  {eps:4.1f} | {it_m:.4f}           | {os_m:.4f}          | "
              f"{it_m/os_m:5.2f}       | {100*np.mean(pr_os[ei] < anchor):3.0f}% of cams")
    print()

res_fd, res_acc = [], []
for c in cams:
    clients, ref, star, r, N = c["clients"], c["ref"], c["theta"], c["r"], len(c["clients"])
    s = sum(client_disp(ref, cl, 1.0, 1, 0.20) for cl in clients)
    nacc = len(accepted(clients))
    est_fd = retract(GRP, ref, GRP.exp(ref, s / N),    r)
    est_ac = retract(GRP, ref, GRP.exp(ref, s / nacc), r)
    res_fd.append(GRP.dist(est_fd, star)); res_acc.append(GRP.dist(est_ac, star))
res_fd, res_acc = np.array(res_fd), np.array(res_acc)
out["oneshot_fixeddenom_resid_C020"] = res_fd
out["oneshot_acceptednorm_resid_C020"] = res_acc
print("Offline decomposition (nonprivate one-shot, C=0.20):")
print(f"  fixed-denominator residual (mean): {res_fd.mean():.3e} rad")
print(f"  accepted-count-normalized (mean):  {res_acc.mean():.3e} rad  (diagnostic; record-dependent)")
print(f"  -> {100*(1-res_acc.mean()/res_fd.mean()):.1f}% of the residual is fixed-denominator dilution.\n")

np.savez(os.path.join(HERE, "..", "results", "rq3_oneshot_baseline.npz"), **out)
print("saved -> results/rq3_oneshot_baseline.npz")
print("\nInterpretation: on this static warm-start task, the single private fixed-denominator")
print("tangent update had lower RMS error and used one release rather than 20.")
print("Repeated private optimization remains relevant when the objective requires multiple rounds.")
