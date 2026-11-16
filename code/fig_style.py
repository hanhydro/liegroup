import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

OKABE_ITO = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "gray":       "#999999",
}

CYCLE = [OKABE_ITO[k] for k in
         ["blue", "vermillion", "green", "orange", "skyblue", "purple", "black"]]

WIDTH_2PANEL = (6.5, 3.0)

TEXTWIDTH_IN = 372.0 / 72.0


def apply_style():
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11.5,
        "axes.labelsize": 12.0,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "legend.fontsize": 11.0,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4,
        "axes.prop_cycle": cycler(color=CYCLE),
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.9,
        "lines.markersize": 5.5,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    try:
        import matplotlib.font_manager as fm
        fm.findfont("Helvetica", fallback_to_default=False)
        rc.update({
            "mathtext.fontset": "custom",
            "mathtext.rm": "Helvetica",
            "mathtext.it": "Helvetica:italic",
            "mathtext.bf": "Helvetica:bold",
            "mathtext.sf": "Helvetica",
        })
    except Exception:
        rc["mathtext.fontset"] = "stixsans"
    plt.rcParams.update(rc)


def panel_label(ax, text, dx=0.028, dy=0.97):
    ax.text(dx, dy, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.75))
