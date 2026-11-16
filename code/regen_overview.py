import os

import numpy as np
import fig_style

fig_style.apply_style()
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib import patheffects

mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["mathtext.cal"] = "cmsy10"
HERE = os.path.dirname(os.path.abspath(__file__))
OI = fig_style.OKABE_ITO
INK = "#222222"
MUTED = "#6E6E6E"
DOMAIN = "#6A6A6A"
NOISE = "#9B4F96"
FS = 11.0
FS_AXIS = 12.0
LABEL_FS = 14.0
FIG_W = fig_style.TEXTWIDTH_IN
H_A, H_L = 1.97, 3.36
FIG_H = H_A + H_L


def draw_pipeline(fig, ax, width, height):
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    pad, hb, line = 0.09, 0.46, 0.105
    lab_gap, brk = 0.22, (0.04, 0.10)
    y_b = 0.02 + 0.33 + hb / 2
    y_a = y_b + hb + 0.28
    server_top = y_a + hb / 2 + 0.38

    def two_line(title, formula):
        return (ax.text(0, 0, title, fontsize=FS, color=INK, ha="center", va="center", zorder=4),
                ax.text(0, 0, formula, fontsize=FS, color=INK, ha="center", va="center",
                        zorder=4))

    boxes = {
        "clients": two_line("Clients", r"send $u_i^t$"),
        "clipC": two_line(r"Clip at $C$", r"average $\bar u_t$"),
        "noise": two_line("Add noise", r"$y_t=\bar u_t+Z_t$"),
        "clipB": two_line(r"Clip at $B$", r"$\widehat v_t=\Pi_B(y_t)$"),
        "exp": two_line("Exponential step", r"$\mathrm{Exp}_{\theta_t}(\gamma\,\widehat v_t)$"),
        "ret": two_line("Return map", r"$\theta_{t+1}\in\overline{B_r}$"),
    }
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def text_w(t):
        b = t.get_window_extent(r)
        return inv.transform((b.x1, 0))[0] - inv.transform((b.x0, 0))[0]

    w = {k: max(text_w(a), text_w(b)) + 2 * pad for k, (a, b) in boxes.items()}
    x_cl = 0.10
    server_left = x_cl + w["clients"] + 0.40
    x_left = server_left + 0.08
    server_right = width - 0.02
    gap = min(0.34, (server_right - 0.08 - x_left - w["ret"] - w["exp"] - w["clipB"]) / 2)
    xc = {"clipC": x_left + w["clipC"] / 2}
    xc["noise"] = x_left + w["clipC"] + gap + w["noise"] / 2
    xc["ret"] = x_left + w["ret"] / 2
    xc["exp"] = x_left + w["ret"] + gap + w["exp"] / 2
    xc["clipB"] = x_left + w["ret"] + gap + w["exp"] + gap + w["clipB"] / 2
    xc["clients"] = x_cl + w["clients"] / 2
    yc = {"clipC": y_a, "noise": y_a, "ret": y_b, "exp": y_b, "clipB": y_b,
          "clients": (y_a + y_b) / 2}
    hgt = {k: hb for k in boxes}
    hgt["clients"] = y_a - y_b + hb
    edge = {"clients": MUTED, "noise": NOISE}
    fill = {"clients": "#F3F3F3", "noise": "#F4ECF4"}
    for k, (ta, tb) in boxes.items():
        ta.set_position((xc[k], yc[k] + line))
        tb.set_position((xc[k], yc[k] - line))
        ax.add_patch(FancyBboxPatch((xc[k] - w[k] / 2, yc[k] - hgt[k] / 2), w[k], hgt[k],
                                    boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=fill.get(k, "#EAF2F8"), ec=edge.get(k, OI["blue"]),
                                    lw=0.9, zorder=3))
    for j in (2, 1):
        ax.add_patch(FancyBboxPatch((x_cl - 0.04 * j, yc["clients"] - hgt["clients"] / 2 + 0.04 * j),
                                    w["clients"], hgt["clients"],
                                    boxstyle="round,pad=0,rounding_size=0.07",
                                    fc="#FAFAFA", ec="#C4C4C4", lw=0.6, zorder=2))

    def left(k):
        return xc[k] - w[k] / 2

    def right(k):
        return xc[k] + w[k] / 2

    def arrow(p, q, color=INK, lw=0.9):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=lw,
                                     color=color, shrinkA=0, shrinkB=0, zorder=5))

    c = 0.04
    lines = {}
    arrow((right("clients") + c, y_a), (left("clipC") - c, y_a))
    arrow((right("clipC") + c, y_a), (left("noise") - c, y_a))
    ax.plot([right("noise") + c, xc["clipB"]], [y_a, y_a], color=INK, lw=0.9, zorder=5)
    arrow((xc["clipB"], y_a), (xc["clipB"], y_b + hb / 2 + c))
    arrow((left("clipB") - c, y_b), (right("exp") + c, y_b))
    arrow((left("exp") - c, y_b), (right("ret") + c, y_b))
    arrow((left("ret") - c, y_b), (right("clients") + c, y_b), color=MUTED, lw=0.8)
    lines["wrap"] = np.array([[right("noise"), y_a], [xc["clipB"], y_a], [xc["clipB"], y_b + hb / 2]])
    lines["return"] = np.array([[left("ret"), y_b], [right("clients"), y_b]])
    lines["to_server"] = np.array([[right("clients"), y_a], [left("clipC"), y_a]])

    server_bottom = 0.02
    ax.add_patch(FancyBboxPatch((server_left, server_bottom), server_right - server_left,
                                server_top - server_bottom,
                                boxstyle="round,pad=0,rounding_size=0.09", fc="none",
                                ec="#B5B5B5", lw=0.7, ls=(0, (3, 2)), zorder=1))
    lab = {}
    top_label_y = y_a + hb / 2 + lab_gap
    lab["server"] = ax.text(server_right - 0.07, top_label_y, "trusted server", fontsize=FS,
                            color=MUTED, ha="right", va="center")
    lab["back"] = ax.text((right("clients") + server_left) / 2, y_b + 0.15,
                          r"$\theta_{t+1}$", fontsize=FS, color=MUTED, ha="center", va="center")

    ya = y_a + hb / 2
    ax.plot([left("clipC"), left("clipC"), right("noise"), right("noise")],
            [ya + brk[0], ya + brk[1], ya + brk[1], ya + brk[0]], color=NOISE, lw=0.8)
    lab["mech"] = ax.text(left("clipC"), top_label_y, "Gaussian mechanism (privacy cost)",
                          fontsize=FS, color=NOISE, ha="left", va="center")
    yb = y_b - hb / 2
    ax.plot([left("ret"), left("ret"), right("clipB"), right("clipB")],
            [yb - brk[0], yb - brk[1], yb - brk[1], yb - brk[0]], color=OI["blue"], lw=0.8)
    lab["post"] = ax.text((left("ret") + right("clipB")) / 2, yb - lab_gap,
                          "post-processing (no privacy cost)", fontsize=FS, color=OI["blue"],
                          ha="center", va="center")
    return [t for pair in boxes.values() for t in pair] + list(lab.values()), lines


def unit(v):
    return v / np.linalg.norm(v)


def proj_tangent(p, v):
    return v - (v @ p) * p


def tangent_basis(p, up=np.array([0.0, 1.0, 0.0])):
    e2 = unit(proj_tangent(p, up))
    e1 = np.cross(e2, p)
    if e1[0] < 0:
        e1 = -e1
    return e1, e2


def exp_s2(p, v):
    n = np.linalg.norm(v)
    return p if n == 0 else np.cos(n) * p + np.sin(n) * v / n


def log_s2(p, q):
    c = np.clip(p @ q, -1, 1)
    w = q - c * p
    nw = np.linalg.norm(w)
    return np.zeros(3) if nw == 0 else np.arccos(c) * w / nw


def cap(c, a, n=240):
    e1, e2 = tangent_basis(c)
    phi = np.linspace(0, 2 * np.pi, n)
    return np.cos(a) * c[None, :] + np.sin(a) * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)


def visible_runs(P):
    vis = P[:, 2] >= 0
    runs, start = [], 0
    for i in range(1, len(P) + 1):
        if i == len(P) or vis[i] != vis[start]:
            runs.append((bool(vis[start]), P[max(start - 1, 0):i]))
            start = i
    return runs


_tilt = np.deg2rad(20)
TILT = np.array([[1, 0, 0], [0, np.cos(_tilt), -np.sin(_tilt)], [0, np.sin(_tilt), np.cos(_tilt)]])


def draw_group(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    n = 420
    xs = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(xs, xs[::-1])
    R2 = X ** 2 + Y ** 2
    Z = np.sqrt(np.clip(1 - R2, 0, None))
    light = unit(np.array([-0.5, 0.6, 0.65]))
    lam = np.clip(X * light[0] + Y * light[1] + Z * light[2], 0, 1)
    spec = np.clip(lam, 0, 1) ** 24
    hi, lo = np.array([0.975, 0.98, 0.99]), np.array([0.78, 0.81, 0.86])
    img = lo + (hi - lo) * (lam[..., None] ** 0.8) + 0.04 * spec[..., None]
    img = np.clip(img, 0, 1)
    im = ax.imshow(np.dstack([img, (R2 <= 1).astype(float)]), extent=(-1, 1, -1, 1),
                   interpolation="bilinear", zorder=0)
    im.set_clip_path(Circle((0, 0), 1, transform=ax.transData))
    ax.add_patch(Circle((0, 0), 1, fc="none", ec="#98A2AD", lw=0.7, zorder=1))
    phi = np.linspace(0, 2 * np.pi, 400)
    for lat in np.deg2rad([-60, -30, 0, 30, 60]):
        P = np.c_[np.cos(lat) * np.cos(phi), np.full_like(phi, np.sin(lat)), np.cos(lat) * np.sin(phi)] @ TILT.T
        for vis, run in visible_runs(P):
            if vis:
                ax.plot(run[:, 0], run[:, 1], color="#CDD2D8", lw=0.35, zorder=1)
    th = np.linspace(0, 2 * np.pi, 400)
    for lon in np.deg2rad(np.arange(0, 180, 30)):
        P = np.c_[np.cos(th) * np.cos(lon), np.sin(th), np.cos(th) * np.sin(lon)] @ TILT.T
        for vis, run in visible_runs(P):
            if vis:
                ax.plot(run[:, 0], run[:, 1], color="#CDD2D8", lw=0.35, zorder=1)

    a_r, a_rho, gB = 0.30, 0.86, 0.50
    lon, lat = np.deg2rad(-22), np.deg2rad(10)
    c = np.array([np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)])
    ce1, ce2 = tangent_basis(c)
    theta = exp_s2(c, -0.20 * ce2)
    t1, t2 = tangent_basis(theta)
    d = unit(t1 + 0.22 * t2)
    D = cap(c, a_rho)
    ax.plot(D[:, 0], D[:, 1], color=DOMAIN, lw=0.9, ls=(0, (3.2, 2.2)), zorder=3)
    Bc = cap(c, a_r)
    ax.add_patch(Polygon(Bc[:, :2], closed=True, fc="#CFE2F0", ec=OI["blue"], lw=0.8, zorder=3))

    rng = np.random.default_rng(7)
    smp = rng.normal(size=(60, 2)) * 0.19 + np.array([0.08, 0.02])
    pts = np.array([exp_s2(theta, a * t1 + b * t2) for a, b in smp])
    ax.scatter(pts[:, 0], pts[:, 1], s=2.0, color=NOISE, alpha=0.7, lw=0, zorder=5)
    ring = np.array([exp_s2(theta, gB * (np.cos(f) * t1 + np.sin(f) * t2)) for f in phi])
    ax.plot(ring[:, 0], ring[:, 1], color=OI["blue"], lw=0.8, ls=(0, (2.2, 1.4)), zorder=5)

    P = np.array([exp_s2(theta, s * 1.62 * d) for s in np.linspace(0, 1, 160)])
    ax.plot(P[:-2, 0], P[:-2, 1], color=OI["vermillion"], lw=1.3, zorder=6)
    end_plain = P[-1]
    ax.plot(end_plain[0], end_plain[1], marker="X", ms=6.0, mew=0, color=OI["vermillion"], zorder=8)
    Q = np.array([exp_s2(theta, s * gB * d) for s in np.linspace(0, 1, 60)])
    ax.plot(Q[:-4, 0], Q[:-4, 1], color=OI["blue"], lw=1.9, zorder=7, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(Q[-8, :2], Q[-1, :2], arrowstyle="-|>", mutation_scale=7.5,
                                 lw=1.3, color=OI["blue"], shrinkA=0, shrinkB=0, zorder=7))
    y_end = Q[-1]
    lv = log_s2(c, y_end)
    theta_next = exp_s2(c, a_r * lv / np.linalg.norm(lv))
    ret = np.array([exp_s2(c, s * lv) for s in np.linspace(1, a_r / np.linalg.norm(lv), 30)])
    ax.plot(ret[:-3, 0], ret[:-3, 1], color=OI["blue"], lw=0.9, ls=(0, (1.2, 1.1)), zorder=7)
    ax.add_patch(FancyArrowPatch(ret[-6, :2], ret[-1, :2], arrowstyle="-|>", mutation_scale=6,
                                 lw=0.9, color=OI["blue"], shrinkA=0, shrinkB=0, zorder=7))
    for p, ms in ((c, 3.6), (theta, 3.6), (theta_next, 3.2)):
        ax.plot(p[0], p[1], "o", ms=ms, color=INK, mec="white", mew=0.5, zorder=9)

    halo = [patheffects.withStroke(linewidth=2.2, foreground=(1, 1, 1, 0.85))]
    lowest = D[np.argmin(D[:, 1])]
    lab = {
        "ref": ax.text(c[0] - 0.03, c[1] - 0.03, r"$\theta_{\mathrm{ref}}$", fontsize=FS,
                       ha="right", va="top", color=INK, zorder=10),
        "theta": ax.text(theta[0] - 0.04, theta[1] - 0.03, r"$\theta_t$", fontsize=FS,
                         ha="right", va="top", color=INK, zorder=10),
        "next": ax.text(theta_next[0] + 0.005, theta_next[1] + 0.05, r"$\theta_{t+1}$",
                        fontsize=FS, ha="right", va="bottom", color=INK, zorder=10),
        "ball": ax.text(c[0] - 0.05, c[1] + 0.05, r"$\overline{B_r}$", fontsize=FS,
                        ha="center", va="bottom", color=OI["blue"], zorder=10),
        "domain": ax.text(lowest[0], lowest[1] - 0.05, r"$\mathcal{D}_{\mathrm{inj}}$",
                          fontsize=FS, ha="center", va="top", color=DOMAIN, zorder=10),
    }
    for t in lab.values():
        t.set_path_effects(halo)
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    lines = {"plain": P, "clipped": Q, "return": ret, "ring": ring}
    return list(lab.values()), lines


def draw_bound(ax):
    import rq4_short_horizon as rq
    T = np.arange(1, 17)
    weighted = np.array([rq.bounds_iterative(rq.variance_schedule(t))[1] for t in T])
    uniform = np.array([rq.bounds_iterative(np.full(t, rq.sigma_rdp(rq.C, t)))[1] for t in T])
    one_step = rq.bounds_one_step(rq.sigma_rdp(rq.C1, 1))["tangent_squared"]
    a0 = rq.A0
    ax.plot(T, uniform, color=INK, lw=1.3, zorder=3)
    ax.plot(T, weighted, color=OI["green"], lw=1.5, zorder=3)
    k = int(np.argmin(weighted))
    ax.plot(T[k], weighted[k], "o", ms=3.6, color=OI["green"], mec="white", mew=0.6, zorder=5)
    ax.plot(0, a0, "s", ms=3.8, color=MUTED, zorder=5, clip_on=False)
    ax.plot(1, one_step, "D", ms=3.6, color=OI["orange"], zorder=5)
    ax.set_yscale("log")
    ax.set_xlim(-0.6, 16.8)
    ax.set_ylim(0.0095, 0.6)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_xticklabels(["0", "4", "8", "12", "16"], fontsize=FS)
    ax.set_yticks([0.01, 0.03, 0.1, 0.3])
    ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3"], fontsize=FS)
    ax.minorticks_off()
    ax.tick_params(length=2.5, width=0.6, pad=1.5)
    ax.set_xlabel(r"Rounds $T$", fontsize=FS_AXIS, labelpad=2)
    ax.set_ylabel("Error bound (rad)", fontsize=FS_AXIS, labelpad=2)
    for s in ax.spines.values():
        s.set_visible(True)
    lab = {
        "init": ax.text(0.9, a0, "initializer", fontsize=FS, color=MUTED, ha="left",
                        va="center"),
        "one": ax.text(1.9, one_step, "one full step", fontsize=FS, color="#A86E00",
                       ha="left", va="center"),
        "uniform": ax.text(16.5, uniform[-1] * 1.12, "uniform", fontsize=FS, color=INK,
                           ha="right", va="bottom"),
        "weighted": ax.text(16.5, weighted[-1] * 0.86, "variance\nschedule", fontsize=FS,
                            color="#00805C", ha="right", va="top", linespacing=1.05),
    }
    lines = {"uniform": np.c_[T, uniform], "weighted": np.c_[T, weighted],
             "init": np.c_[[0, 0], [a0, a0]], "one": np.c_[[1, 1], [one_step, one_step]]}
    return list(lab.values()), lines


def draw_exits(ax):
    d = np.load(os.path.join(HERE, "..", "results", "rq1_locality_ablation.npz"))
    eps = d["eps"]
    plain = 100 * d["plain_any"]
    ci = 100 * d["plain_ci"]
    clipped = np.zeros_like(plain)
    ax.fill_between(eps, ci[:, 0], ci[:, 1], color=OI["vermillion"], alpha=0.18, lw=0, zorder=1)
    ax.plot(eps, plain, "o-", color=OI["vermillion"], ms=3.2, lw=1.4, zorder=3)
    ax.plot(eps, clipped, "o-", color=OI["blue"], ms=3.2, lw=1.4, zorder=3, clip_on=False)
    ax.set_xscale("log", base=2)
    ax.set_xticks(eps)
    ax.set_xticklabels([f"{e:g}" for e in eps], fontsize=FS)
    ax.minorticks_off()
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50", "100"], fontsize=FS)
    ax.set_ylim(-3, 128)
    ax.set_xlim(eps[0] / 1.22, eps[-1] * 1.22)
    ax.tick_params(length=2.5, width=0.6, pad=1.5)
    ax.set_xlabel(r"Privacy budget $\varepsilon$", fontsize=FS_AXIS, labelpad=2)
    ax.set_ylabel(r"Runs leaving $\mathcal{D}_{\mathrm{inj}}$ (%)", fontsize=FS_AXIS, labelpad=2)
    for s in ax.spines.values():
        s.set_visible(True)
    lab = {
        "plain": ax.text(1.75, 62, "plain\nGaussian", fontsize=FS, color=OI["vermillion"],
                         ha="left", va="bottom", linespacing=1.05),
        "clip": ax.text(8.0, 8, "clipped", fontsize=FS, color=OI["blue"], ha="right",
                        va="bottom"),
    }
    lines = {"plain": np.c_[eps, plain], "clipped": np.c_[eps, clipped]}
    return list(lab.values()), lines


def build():
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def axes(x, y, w, h):
        return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])

    ax_a = axes(0.0, H_L, FIG_W, H_A)
    d_b = 2.42
    ax_b = axes(0.0, (H_L - d_b) / 2, d_b, d_b)
    x_cd, w_cd, h_cd, bottom = 3.10, FIG_W - 3.10 - 0.06, 1.18, 0.43
    ax_d = axes(x_cd, bottom, w_cd, h_cd)
    ax_c = axes(x_cd, H_L - 0.10 - h_cd, w_cd, h_cd)
    ta, la = draw_pipeline(fig, ax_a, FIG_W, H_A)
    out = {"a": (ax_a, ta, la)}
    tb, lb = draw_group(ax_b)
    out["b"] = (ax_b, tb, lb)
    tc, lc = draw_bound(ax_c)
    out["c"] = (ax_c, tc, lc)
    td, ld = draw_exits(ax_d)
    out["d"] = (ax_d, td, ld)
    kw = dict(ha="left", va="top", fontsize=LABEL_FS, fontweight="bold", color=INK, zorder=20)
    for key, (x, y) in {"a": (0.004, 0.99), "b": (0.004, 0.99), "c": (0.03, 0.955),
                        "d": (0.03, 0.955)}.items():
        ax, texts, lines = out[key]
        texts.append(ax.text(x, y, f"({key})", transform=ax.transAxes, **kw))
    return fig, out, []


def main():
    fig, _, _ = build()
    out = os.path.join(HERE, "..", "figures", "overview")
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out + ".pdf")
        fig.savefig(out + ".png", dpi=600)
    plt.close(fig)
    print("figure -> figures/overview.pdf/png")


if __name__ == "__main__":
    main()
