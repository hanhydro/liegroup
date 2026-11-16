import os
import numpy as np
from scipy.optimize import minimize_scalar
from real_rotation_averaging import load, build_cameras, sigma_rdp, GRP
from fedopt_stability import clip, retract
from camera_public_bound import P as CAM_P, constants as cam_constants

HERE = os.path.dirname(os.path.abspath(__file__))
ETA, GAMMA, TAU, T, DELTA, M, EPS, C = 0.5, 1.0, 0.20, 20, 1e-5, 100, 4.0, 0.20
B_SAFE = 0.87
QS = [1.0, 0.75, 0.5, 0.25, 0.1]
D = GRP.dim
RSUB = 4000
ZERO = 0.0 * GRP.basis[0]
Q0 = 0.75

assert (ETA, GAMMA, TAU, C, B_SAFE, T, DELTA) == (CAM_P["eta"], CAM_P["gamma"], CAM_P["r"], CAM_P["C"],
                                                  CAM_P["B"], CAM_P["T"], CAM_P["delta"])


def count_rdp_eps(sigma, N, m):
    p, Delta, ell = m / N, 2.0 * C / m, -np.log(DELTA)

    def eps_at(order):
        x = order * (order - 1) * Delta ** 2 / (2 * sigma ** 2)
        if p >= 1:
            rdp = T * x / (order - 1)
        else:
            rdp = T * np.logaddexp(np.log1p(-p), np.log(p) + x) / (order - 1)
        return rdp + ell / (order - 1)

    grid = 1 + np.logspace(-4, 4, 4001)
    vals = np.array([eps_at(a) for a in grid])
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    res = minimize_scalar(eps_at, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    order = res.x if res.fun < vals[k] else grid[k]
    return float(eps_at(order)), float(order)


def sigma_count(N, m):
    target = EPS * (1 - 1e-9)
    hi = sigma_rdp(C, T, DELTA, m, EPS)
    while count_rdp_eps(hi, N, m)[0] > target:
        hi *= 1.0000001
    lo = hi / 100
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if count_rdp_eps(mid, N, m)[0] <= target:
            hi = mid
        else:
            lo = mid
    e, order = count_rdp_eps(hi, N, m)
    assert e <= target
    return hi, order


def squared_bound(N, m, sigma, kw):
    f = (N - m) / (m * (N - 1))
    A0, Rs, d = CAM_P["A0"], CAM_P["R_star"], CAM_P["d"]
    v = kw ** (2 * T) * A0 ** 2 + GAMMA ** 2 * (f * C ** 2 + d * sigma ** 2) * (1 - kw ** (2 * T)) / (1 - kw ** 2)
    return float(min(Rs, np.sqrt(v)))


def u_direct(theta, client, Cval):
    m, acc = client
    if not acc:
        return ZERO
    return clip(GRP, ETA * GRP.log(theta, m), Cval)


def gauss_scaled(row, sigma):
    return sigma * sum(row[k] * GRP.basis[k] for k in range(D))


def run_arm(clients, theta_ref, target, r, m, sigma, perm, gauss):
    theta = theta_ref.copy(); fires = 0
    for t in range(T):
        gS = sum(u_direct(theta, clients[i], C) for i in perm[t][:m]) / m
        vhat = gS + gauss_scaled(gauss[t], sigma)
        nv = GRP.norm(vhat)
        if nv > B_SAFE:
            vhat = vhat * (B_SAFE / nv); fires += 1
        theta = retract(GRP, theta_ref, GRP.exp(theta, GAMMA * vhat), r)
    return GRP.dist(theta, target), fires


n, adj, Rhat, nodes = load()
cams = build_cameras(adj, Rhat, TAU)
kw = cam_constants(Q0)["kappa_w"]
assert all(c["r"] == TAU for c in cams)
assert all(np.mean([x[1] for x in c["clients"]]) >= Q0 for c in cams)
print(f"Ellis_Island: {len(cams)} cameras; public fixed-size partial participation "
      f"(REPLACEMENT adjacency, sensitivity 2C/m), eps={EPS}, C={C}, T={T}, M={M}, kappa_w={kw:.6f}.\n")

print("lem:sampling-variance check (per-round participation RMS at the public center):")
print("   q   |  m (med) | closed-form | empirical | rel.err")
out = {"QS": np.array(QS), "eps": EPS, "C": C, "T": T, "adjacency": "replacement",
       "sensitivity_rule": "2C/m", "hidden_membership_amplification": "none",
       "kappa_w": kw, "q0": Q0}
svc = {q: [] for q in QS}; sve = {q: [] for q in QS}
for ci, c in enumerate(cams):
    cl = c["clients"]; N = len(cl)
    gs = [u_direct(c["ref"], x, C) for x in cl]
    gbar = sum(gs) / N
    var_g = float(np.mean([GRP.norm(g - gbar) ** 2 for g in gs]))
    rng = np.random.default_rng(70000 + ci)
    for q in QS:
        m = max(1, round(q * N))
        svc[q].append(0.0 if m >= N else np.sqrt((N - m) / (m * (N - 1))) * np.sqrt(var_g))
        if m >= N:
            sve[q].append(0.0)
        else:
            acc = 0.0
            for _ in range(RSUB):
                idx = rng.choice(N, size=m, replace=False)
                acc += GRP.norm(sum(gs[i] for i in idx) / m - gbar) ** 2
            sve[q].append(np.sqrt(acc / RSUB))
for q in QS:
    mm = float(np.median([max(1, round(q * len(c["clients"]))) for c in cams]))
    cf = float(np.mean(svc[q])); em = float(np.mean(sve[q])); rel = abs(cf - em) / em if em > 0 else 0.0
    print(f"  {q:.2f} |  {mm:5.1f}  |   {cf:.5f}   |  {em:.5f}  | {rel:.2%}")
    out[f"svclosed_q{q}"] = np.array(svc[q]); out[f"svemp_q{q}"] = np.array(sve[q])

for q in QS:
    for tag in ["", "count_"]:
        out[f"{tag}rms_q{q}"] = np.zeros(len(cams)); out[f"{tag}errs_q{q}"] = np.zeros((len(cams), M))
        out[f"{tag}sigma_q{q}"] = np.zeros(len(cams)); out[f"{tag}client_clipfrac_q{q}"] = np.zeros(len(cams))
        out[f"{tag}bound_q{q}"] = np.zeros(len(cams))
    out[f"count_order_q{q}"] = np.zeros(len(cams)); out[f"count_eps_q{q}"] = np.zeros(len(cams))
    out[f"sens_q{q}"] = np.zeros(len(cams)); out[f"m_q{q}"] = np.zeros(len(cams), dtype=int)
    out[f"N_q{q}"] = np.zeros(len(cams), dtype=int)
for ci, c in enumerate(cams):
    cl = c["clients"]; N = len(cl); target = c["theta"]; r = c["r"]
    m_by_q = {q: max(1, round(q * N)) for q in QS}
    sig_by_q = {q: sigma_rdp(C, T, DELTA, m_by_q[q], EPS) for q in QS}
    csig_by_q = {}
    for q in QS:
        s, order = sigma_count(N, m_by_q[q])
        if q == 1.0:
            assert abs(s / sig_by_q[q] - 1) < 1e-6
            s = sig_by_q[q]
        csig_by_q[q] = s
        out[f"count_order_q{q}"][ci] = order; out[f"count_eps_q{q}"][ci] = count_rdp_eps(s, N, m_by_q[q])[0]
        out[f"m_q{q}"][ci] = m_by_q[q]; out[f"N_q{q}"][ci] = N; out[f"sens_q{q}"][ci] = 2.0 * C / m_by_q[q]
        out[f"sigma_q{q}"][ci] = sig_by_q[q]; out[f"count_sigma_q{q}"][ci] = s
        out[f"bound_q{q}"][ci] = squared_bound(N, m_by_q[q], sig_by_q[q], kw)
        out[f"count_bound_q{q}"][ci] = squared_bound(N, m_by_q[q], s, kw)
    pgen = np.random.default_rng(90000 + ci)
    ngen = np.random.default_rng(91000 + ci)
    fires = {(q, tag): 0 for q in QS for tag in ["", "count_"]}
    for k in range(M):
        perm = [pgen.permutation(N) for _ in range(T)]
        gauss = ngen.standard_normal((T, D))
        for q in QS:
            d, f = run_arm(cl, c["ref"], target, r, m_by_q[q], sig_by_q[q], perm, gauss)
            out[f"errs_q{q}"][ci, k] = d; fires[(q, "")] += f
            if q < 1.0:
                d, f = run_arm(cl, c["ref"], target, r, m_by_q[q], csig_by_q[q], perm, gauss)
            out[f"count_errs_q{q}"][ci, k] = d; fires[(q, "count_")] += f
    for q in QS:
        for tag in ["", "count_"]:
            out[f"{tag}rms_q{q}"][ci] = float(np.sqrt(np.mean(out[f"{tag}errs_q{q}"][ci] ** 2)))
            out[f"{tag}client_clipfrac_q{q}"][ci] = fires[(q, tag)] / (M * T)
    print(f"  camera {ci + 1:2d}/{len(cams)} done (N={N})", flush=True)

print("\nEnd-to-end private mechanism vs participation q (paired; replacement sensitivity 2C/m):")
print("   q   | m (med) | sigma worst | sigma count | RMS worst | RMS count | bound worst | bound count | contacts")
for q in QS:
    mm = float(np.median(out[f"m_q{q}"])); contacts = mm * T
    out[f"contacts_q{q}"] = contacts; out[f"scalars_q{q}"] = contacts * D
    print(f"  {q:.2f} |  {mm:5.1f}  |   {np.median(out[f'sigma_q{q}']):.4f}    |   "
          f"{np.median(out[f'count_sigma_q{q}']):.4f}    |  {out[f'rms_q{q}'].mean():.4f}   |  "
          f"{out[f'count_rms_q{q}'].mean():.4f}   |   {out[f'bound_q{q}'].mean():.4f}    |   "
          f"{out[f'count_bound_q{q}'].mean():.4f}    | {contacts:.0f}")

out["seeds"] = {"participation_base": 90000, "noise_base": 91000, "svcheck_base": 70000}
np.savez(os.path.join(HERE, "..", "results", "fixed_size_participation.npz"), **out)
print("\nsaved -> results/fixed_size_participation.npz")
print("Both calibrations release the selected indices and use no hidden-membership amplification.")
print("The count calibration needs subsets independent of the data and averages over them. The target is used only for evaluation.")
