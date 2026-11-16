import os
import numpy as np
import fig_style
fig_style.apply_style()
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["savefig.bbox"] = "standard"
HERE = os.path.dirname(os.path.abspath(__file__))
OI = fig_style.OKABE_ITO
d = np.load(os.path.join(HERE, "..", "results", "robust_camera_task.npz"))
sw = np.load(os.path.join(HERE, "..", "results", "robust_onestep_sweep.npz"))
eps, degs = d["eps"], d["degrees"]
step_best = sw["private"].mean(axis=-1).min(axis=1)

fig, axes = plt.subplots(1, len(degs), figsize=(fig_style.TEXTWIDTH_IN, 3.05), sharey=True,
                         constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.02)
for k, (ax, deg) in enumerate(zip(axes, degs)):
    it = d["priv_iter"][k].mean(axis=1)
    os_ = d["priv_sqoneshot"][k].mean(axis=1)
    ax.plot(eps, it, "o-", color=OI["blue"], label="Private iterative")
    ax.plot(eps, os_, "s--", color=OI["orange"], label="Private tangent mean")
    ax.plot(eps, step_best[k], "D-.", color=OI["purple"], ms=4.5, label="Gradient step, best of four")
    amb = d["priv_ambient"][k].mean(axis=1)
    ax.plot(eps, amb, "^:", color=OI["green"], label="Private ambient")
    ax.axhline(d["init_err"][k].mean(), color=OI["black"], ls="-.", lw=1.4, label="Public initializer")
    ax.axhline(d["np_sqoneshot"][k].mean(), color=OI["gray"], ls=":", lw=1.6, label="Nonprivate tangent mean")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(eps); ax.set_xticklabels([f"{e:g}" for e in eps])
    ax.set_xlabel(r"Privacy budget $\varepsilon$")
    fig_style.panel_label(ax, f"({'abc'[k]})")
axes[0].set_ylabel("RMS geodesic error (rad)")
for ax in axes:
    ax.set_ylim(1.2e-3, 0.5)
    ax.set_yticks([0.003, 0.01, 0.03, 0.1, 0.3])
    ax.set_yticklabels(["0.003", "0.01", "0.03", "0.1", "0.3"])
    ax.minorticks_off()
handles, labels = axes[0].get_legend_handles_labels()
assert len(handles) == 6
fig.legend(handles, labels, loc="outside lower center", ncol=2, handlelength=2.2,
           handletextpad=0.5, labelspacing=0.3, columnspacing=1.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "..", "figures", f"robust_camera.{ext}"))
print("figure -> figures/robust_camera.pdf/png")
