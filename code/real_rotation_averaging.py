import os
import numpy as np
from fedopt_stability import so3, karcher_mean, clip, retract, RNG

HERE = os.path.dirname(os.path.abspath(__file__))
GRP = so3()


def parse_egs(path):
    edges = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) < 11:
                continue
            i, j = int(t[0]), int(t[1])
            R = np.array([float(x) for x in t[2:11]]).reshape(3, 3)
            edges.append((i, j, R))
    return edges


def parse_gt(path):
    gt = {}
    with open(path) as f:
        lines = f.read().split("\n")
    assert lines[0].startswith("# Bundle")
    ncam = int(lines[1].split()[0])
    p = 2
    for c in range(ncam):
        foc = float(lines[p].split()[0])
        R = np.array([[float(x) for x in lines[p + 1 + k].split()] for k in range(3)])
        p += 5
        if foc > 0 and np.isfinite(R).all() and abs(np.linalg.det(R) - 1) < 1e-2:
            gt[c] = R
    return gt


def build_component(edges, gt, ncap=250):
    adj_all = {}
    for i, j, R in edges:
        if i in gt and j in gt:
            adj_all.setdefault(i, []).append((j, R))
            adj_all.setdefault(j, []).append((i, R.T))
    start = max(adj_all, key=lambda k: len(adj_all[k]))
    seen, fr = {start}, [start]
    while fr and len(seen) < ncap:
        nxt = []
        for u in fr:
            for v, _ in adj_all[u]:
                if v not in seen and len(seen) < ncap:
                    seen.add(v); nxt.append(v)
        fr = nxt
    nodes = sorted(seen)
    idx = {c: k for k, c in enumerate(nodes)}
    adj = {k: [] for k in range(len(nodes))}
    for i, j, R in edges:
        if i in idx and j in idx:
            adj[idx[i]].append((idx[j], R))
            adj[idx[j]].append((idx[i], R.T))
    Rhat = [gt[c] for c in nodes]
    return len(nodes), adj, Rhat, nodes


def candidate_clients(i, adj, Rhat, tau):
    out = []
    for j, Rij in adj[i]:
        acc = GRP.dist(Rij, Rhat[i] @ Rhat[j].T) < tau
        out.append((Rij @ Rhat[j], bool(acc)))
    return out


def accepted(clients):
    return [m for m, a in clients if a]


def client_raw_disp(theta, client, eta, tau_loc):
    m, acc = client
    if not acc:
        return 0.0 * GRP.log(theta, m)
    ti = theta.copy()
    for _ in range(tau_loc):
        ti = GRP.exp(ti, eta * GRP.log(ti, m))
    return GRP.log(theta, ti)


def client_disp(theta, client, eta, tau_loc, C):
    return clip(GRP, client_raw_disp(theta, client, eta, tau_loc), C)


def fedopt_public(clients, theta_ref, theta0, eta, gamma, tau_loc, C, sigma, r, T,
                  B=None):
    theta = theta0.copy(); fires = 0; N = len(clients)
    for _ in range(T):
        vhat = sum(client_disp(theta, c, eta, tau_loc, C) for c in clients) / N
        vhat = vhat + (GRP.sample_tangent(sigma) if sigma > 0 else 0.0 * vhat)
        if B is not None:
            nv = GRP.norm(vhat)
            if nv > B:
                vhat = vhat * (B / nv); fires += 1
        theta = retract(GRP, theta_ref, GRP.exp(theta, gamma * vhat), r)
    return theta, fires


def sigma_rdp(C, T, delta, N, eps):
    Delta = 2.0 * C / N
    Ld = -np.log(delta)
    gap = eps / (np.sqrt(Ld + eps) + np.sqrt(Ld))
    return Delta * np.sqrt(T) / (np.sqrt(2.0) * gap)


def clip_stats(clients, theta_ref, eta, tau_loc, C, r, samples=300):
    hit = tot = 0
    b = 0.0
    N = len(clients)
    for _ in range(samples):
        xi = GRP.sample_tangent(1.0); nn = GRP.norm(xi)
        th = GRP.exp(theta_ref, xi * (RNG.uniform(0, r) / nn)) if nn > 0 else theta_ref.copy()
        ov = 0.0
        for c in clients:
            un = GRP.norm(client_raw_disp(th, c, eta, tau_loc))
            if c[1]:
                tot += 1; hit += (un > C)
            ov += max(0.0, un - C)
        b = max(b, ov / N)
    return hit / max(1, tot), b


def instance_regime(r, eta, gamma, tau_loc, C, B):
    K = 0.25
    rho_conv = np.pi / (2 * np.sqrt(K))
    Gmax = 2 * r
    U_client = 2.0 * eta * r
    rho_loc = U_client
    reach = r + gamma * B
    rho_star = reach + 0.03
    rplus = max(r + rho_loc, rho_star)
    q = np.sqrt(K) * rho_star
    L_Pi = float(q / np.sin(q)) if 0 < q < np.pi else float("inf")
    M = rho_star - r - gamma * C
    return dict(K=K, r=r, rho_loc=rho_loc, B=B, rho_star=rho_star, rplus=rplus,
                two_rp=2 * rplus, rho_conv=rho_conv, M=M, L_Pi=L_Pi,
                safe_ok=bool(rho_star < rho_conv), in_regime=bool(2 * rplus < rho_conv))


def measured_kappa(clients, theta_ref, eta, gamma, tau_loc, C, r, trials=150):
    N = len(clients)
    def step(theta):
        vhat = sum(client_disp(theta, c, eta, tau_loc, C) for c in clients) / N
        return retract(GRP, theta_ref, GRP.exp(theta, gamma * vhat), r)
    ratios = []
    for _ in range(trials):
        x = GRP.exp(theta_ref, _ball(r)); y = GRP.exp(theta_ref, _ball(r))
        d0 = GRP.dist(x, y)
        if d0 > 1e-6:
            ratios.append(GRP.dist(step(x), step(y)) / d0)
    return float(np.mean(ratios))


def _ball(r):
    xi = GRP.sample_tangent(1.0); nn = GRP.norm(xi)
    return xi * (RNG.uniform(0, r) / nn) if nn > 0 else xi * 0


def rms_to_star(clients, theta_star, theta_ref, r, eta, gamma, tau_loc, C, sigma, T, M,
                B=None):
    ds = []; fires = 0
    for _ in range(M):
        th, f = fedopt_public(clients, theta_ref, theta_ref.copy(), eta, gamma,
                              tau_loc, C, sigma, r, T, B=B)
        ds.append(GRP.dist(th, theta_star)); fires += f
    return float(np.sqrt(np.mean(np.square(ds)))), fires / (M * T)


def load(scene="Ellis_Island", ncap=250):
    sc = os.path.join(HERE, "..", "data", scene)
    edges = parse_egs(os.path.join(sc, "EGs.txt"))
    gt = parse_gt(os.path.join(sc, "gt_bundle.out"))
    return build_component(edges, gt, ncap)


def export_edge_records(cams, adj, Rhat, nodes, tau, scene, n_comp, path):
    import json
    recs = []
    for c in cams:
        i = c["i"]
        edges = []
        for j, Rij in adj[i]:
            res = float(GRP.dist(Rij, Rhat[i] @ Rhat[j].T))
            edges.append(dict(neighbor_id=int(nodes[j]), neighbor_node=int(j),
                              residual_rad=res, accepted=bool(res < tau)))
        recs.append(dict(camera_id=int(nodes[i]), node_index=int(i),
                         degree=int(c["N"]), accepted=int(c["nacc"]),
                         rejected=int(c["N"] - c["nacc"]),
                         ball_radius_rad=float(c["r"]), kappa_hat=float(c["kappa"]),
                         edges=edges))
    out = dict(scene=scene, tau=float(tau), n_component=int(n_comp),
               n_target_cameras=len(recs), cameras=recs)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return sum(len(r["edges"]) for r in recs)


def build_cameras(adj, Rhat, tau, min_deg=120, ncam=16):
    cand = [(i, candidate_clients(i, adj, Rhat, tau)) for i in range(len(Rhat))]
    cand = [(i, cl) for i, cl in cand if len(cl) >= min_deg]
    cand.sort(key=lambda t: -len(t[1]))
    out = []
    for i, cl in cand[:ncam]:
        acc = accepted(cl); ref = Rhat[i]
        ts = karcher_mean(GRP, acc)
        r = float(tau)
        kap = measured_kappa(cl, ref, 0.5, 1.0, 1, 0.3, r)
        out.append(dict(i=i, clients=cl, acc=acc, theta=ts, ref=ref, r=r, kappa=kap,
                        N=len(cl), nacc=len(acc)))
    return out


def crosscam_multiplier(ncam):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, ncam - 1))
    except Exception:
        return 2.131


if __name__ == "__main__":
    TAU, ETA, GAMMA, TAU_LOC, T, DELTA, M = 0.20, 0.5, 1.0, 1, 20, 1e-5, 100
    B_SAFE = 0.87
    EPS = [1.0, 2.0, 4.0, 8.0, 16.0]
    n, adj, Rhat, nodes = load()
    cams = build_cameras(adj, Rhat, TAU)
    crosscam_mult = crosscam_multiplier(len(cams))
    Nd = [c["N"] for c in cams]; nacc = [c["nacc"] for c in cams]; rs = [c["r"] for c in cams]
    kaps = [c["kappa"] for c in cams]
    rej = [1 - a / d for a, d in zip(nacc, Nd)]
    print(f"Ellis_Island component n={n}; {len(cams)} target cameras selected by PUBLIC degree",
          flush=True)
    print(f"  N=deg(i) (fixed denominator): median={np.median(Nd):.1f} [{min(Nd)},{max(Nd)}]; "
          f"accepted: median={int(np.median(nacc))} [{min(nacc)},{max(nacc)}]; "
          f"filter-rejection: median={100*np.median(rej):.2f}%", flush=True)
    print(f"  kappa-hat (measured, public-centred): median={np.median(kaps):.3f} "
          f"[{min(kaps):.3f},{max(kaps):.3f}]  "
          f"(descriptive cross-camera multiplier={crosscam_mult:.3f})", flush=True)
    rec = [GRP.dist(c["theta"], c["ref"]) for c in cams]
    print(f"  d(theta*, public anchor): median={np.median(rec):.4f} rad "
          f"({np.degrees(np.median(rec)):.2f} deg); theta* is OFFLINE evaluation only", flush=True)

    keys = ["r", "rho_loc", "B", "rho_star", "rplus", "two_rp", "L_Pi", "M"]
    regs = [instance_regime(c["r"], ETA, GAMMA, TAU_LOC, 0.3, B_SAFE) for c in cams]
    print(f"  REGIME TUPLE (direct client bound U_client<=2*eta*r, B={B_SAFE}, "
          f"two-radius limit={regs[0]['rho_conv']:.3f}):", flush=True)
    for k in keys:
        v = np.array([r[k] for r in regs])
        print(f"    {k:9s} median={np.median(v):.3f} [{v.min():.3f},{v.max():.3f}]", flush=True)
    print(f"    post-noise reach rho*<two-radius limit: {all(r['safe_ok'] for r in regs)}  "
          f"numerical radius condition 2r+<limit: {all(r['in_regime'] for r in regs)}", flush=True)
    reg_tuple = {k: [float(r[k]) for r in regs] for k in keys}

    print("\n[eps-sweep] C=0.3 (clip inactive: thm:main), public center + fixed N, post-noise clip B=0.87",
          flush=True)
    print(" eps | mean RMS to theta* (rad) +/- cross-camera multiplier*s/sqrt(16) "
          "[descriptive, not CI] | median sigma | post-noise activation", flush=True)
    A_mean = []; A_crosscam_err = []; A_sig = []; A_percam = []; A_fire = []
    for eps in EPS:
        vals = []; sigs = []; fire = 0.0
        for c in cams:
            sig = sigma_rdp(0.3, T, DELTA, c["N"], eps); sigs.append(sig)
            rms, fr = rms_to_star(c["clients"], c["theta"], c["ref"], c["r"], ETA, GAMMA,
                                  TAU_LOC, 0.3, sig, T, M, B=B_SAFE)
            vals.append(rms); fire = max(fire, fr)
        vals = np.array(vals)
        crosscam_err = crosscam_mult * vals.std(ddof=1) / np.sqrt(len(vals))
        A_mean.append(vals.mean()); A_crosscam_err.append(crosscam_err)
        A_sig.append(np.median(sigs)); A_percam.append(vals); A_fire.append(fire)
        print(f" {eps:5.1f}| {vals.mean():.4f} +/- {crosscam_err:.4f} | {np.median(sigs):.4f} | "
              f"max {fire*100:.2f}%", flush=True)

    print("\n[active-clipping] cor:active-clipping, eps=4 (bhat_clip is an EMPIRICAL estimate)",
          flush=True)
    epsB = 4.0; B_C = [0.30, 0.06, 0.04, 0.025]
    B_rows = []; B_percam = []
    noupd = float(np.median(rec))
    noupd_mean = float(np.mean(rec))
    nonpriv0 = float(np.mean([rms_to_star(c["clients"], c["theta"], c["ref"], c["r"], ETA, GAMMA,
                                          TAU_LOC, 0.3, 0.0, T, 1, B=B_SAFE)[0] for c in cams]))
    print(f"  [baseline] no-update (public anchor) d to theta*: mean={noupd_mean:.4f} median={noupd:.4f} rad "
          f"({np.degrees(noupd):.2f} deg); non-private unclipped RMS={nonpriv0:.5f}", flush=True)
    for C in B_C:
        rms = []; crates = []; bclips = []; bounds = []; nonpriv = []
        for c, reg in zip(cams, regs):
            sig = sigma_rdp(C, T, DELTA, c["N"], epsB)
            cr, bcl = clip_stats(c["clients"], c["ref"], ETA, TAU_LOC, C, c["r"], samples=60)
            floor = reg["L_Pi"] / (1 - c["kappa"]) * sig * np.sqrt(GRP.dim)
            bclip_floor = reg["L_Pi"] * GAMMA * bcl / (1 - c["kappa"])
            r_, _ = rms_to_star(c["clients"], c["theta"], c["ref"], c["r"], ETA, GAMMA,
                                TAU_LOC, C, sig, T, M, B=B_SAFE)
            npr, _ = rms_to_star(c["clients"], c["theta"], c["ref"], c["r"], ETA, GAMMA,
                                 TAU_LOC, C, 0.0, T, 1, B=B_SAFE)
            rms.append(r_); crates.append(cr); bclips.append(bcl)
            bounds.append(floor + bclip_floor); nonpriv.append(npr)
        rms = np.array(rms); bounds = np.array(bounds)
        frac_improve = float(np.mean(rms < np.array(rec)))
        B_percam.append(rms.copy())
        B_rows.append((C, np.mean(crates), np.median(bclips), rms.mean(), np.median(bounds),
                       np.mean(rms <= bounds), np.mean(nonpriv), frac_improve))
        print(f"  C={C:.3f}: accepted-client clip={np.mean(crates)*100:4.0f}%  "
              f"bhat_clip={np.median(bclips):.4f}  "
              f"emp RMS={rms.mean():.4f}  nonpriv-clip(bias)={np.mean(nonpriv):.4f}  "
              f"noise+overflow={np.median(bounds):.4f}  below? {np.mean(rms <= bounds)*100:.0f}% cams  "
              f"beats-anchor={frac_improve*100:.0f}% cams",
              flush=True)

    np.savez(os.path.join(HERE, "..", "results", "real_ra_results.npz"),
             eps=np.array(EPS), A_mean=np.array(A_mean),
             A_crosscam_err=np.array(A_crosscam_err),
             A_sig=np.array(A_sig), A_percam=np.array(A_percam), A_fire=np.array(A_fire),
             B_C=np.array(B_C), B_rows=np.array(B_rows), B_percam=np.array(B_percam),
             r_med=np.median(rs),
             kappa_med=np.median(kaps), n_cams=len(cams), deg_med=float(np.median(Nd)),
             nacc_med=int(np.median(nacc)), rej_med=float(np.median(rej)),
             degrees=np.array(Nd), naccepted=np.array(nacc),
             crosscam_multiplier=crosscam_mult,
             B_safe=B_SAFE, noupd=noupd, noupd_mean=noupd_mean, nonpriv0=nonpriv0,
             **{f"tuple_{k}": np.array(reg_tuple[k]) for k in keys})
    nedges = export_edge_records(cams, adj, Rhat, nodes, TAU, "Ellis_Island", n,
                                 os.path.join(HERE, "..", "results", "real_ra_edge_records.json"))
    print(f"edge records -> results/real_ra_edge_records.json "
          f"({len(cams)} target cameras, {nedges} candidate edges)", flush=True)
    try:
        import fig_style; fig_style.apply_style(); import matplotlib.pyplot as plt
        OI = fig_style.OKABE_ITO
        fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)
        e = np.array(EPS); am = np.array(A_mean)
        ax[0].errorbar(e, am, yerr=A_crosscam_err, fmt="o-", capsize=3,
                       color=OI["blue"],
                       label=r"Mean per-camera RMS")
        ref = am[1] * (e[1] / e)
        ax[0].plot(e, ref, "--", lw=1.4, color=OI["black"], label=r"$\propto 1/\varepsilon$")
        ax[0].axhline(noupd_mean, color=OI["green"], ls="-.", lw=1.6,
                      label="No-update (public anchor)")
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel(r"Privacy budget $\varepsilon$")
        ax[0].set_ylabel(r"Mean per-camera RMS to $\theta^*$ (rad)")
        ax[0].legend(loc="lower left")
        ax[0].set_ylim(top=ax[0].get_ylim()[1] * 1.8)
        fig_style.panel_label(ax[0], "(a)")
        Cs = [r[0] for r in B_rows]; emp = [r[3] for r in B_rows]; bnd = [r[4] for r in B_rows]
        npv = [r[6] for r in B_rows]; clp = [r[1] * 100 for r in B_rows]
        x = np.arange(len(Cs))
        ax[1].bar(x - 0.27, emp, 0.27, color=OI["blue"], label="Private RMS")
        ax[1].bar(x, npv, 0.27, color=OI["orange"], label="Non-private clipped (bias)")
        ax[1].bar(x + 0.27, bnd, 0.27, color=OI["skyblue"],
                  label=r"Noise $+$ overflow")
        ymax = float(max(bnd))
        ax[1].set_ylim(0, ymax * 1.5)
        for k, cp in enumerate(clp):
            ax[1].text(x[k], bnd[k] * 0.5, f"{cp:.0f}%", ha="center", va="center",
                       fontsize=10, color=OI["black"],
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7",
                                 lw=0.5, alpha=0.9))
        ax[1].axhline(noupd_mean, color=OI["green"], ls="-.", lw=1.6,
                      label="No-update (public anchor)")
        ax[1].set_xticks(x); ax[1].set_xticklabels([f"{c:.3f}" for c in Cs])
        ax[1].set_xlabel(r"Clip $C$ (accepted-client rate labeled)")
        ax[1].set_ylabel(r"Mean per-camera RMS to $\theta^*$ (rad)")
        ax[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
        fig_style.panel_label(ax[1], "(b)")
        fig.savefig(os.path.join(HERE, "..", "figures", "real_rotation_averaging.png"))
        print("\nfigure -> figures/real_rotation_averaging.png", flush=True)
    except Exception as ex:
        print(f"\n[figure skipped: {ex}]", flush=True)
