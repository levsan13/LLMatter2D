"""Figures: the 11 Archimedean tilings and their nearest-neighbour tight-binding bands.

    python paper/scripts/fig_archimedean.py [--out paper/figures]

Writes archimedean_tilings.{pdf,png} and archimedean_bands.{pdf,png}.
Requires matplotlib (pip install -e ".[paper]").
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from llmatter2d.lattices import ARCHIMEDEAN, analyze_bands, band_structure  # noqa: E402

# validated categorical pair (CVD-safe, >= 3:1 contrast on white)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
LABELS = {"G": r"$\Gamma$", "M": "M", "K": "K", "X": "X", "Y": "Y"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def vertex_symbol(name: str) -> str:
    """'3^4.6' -> mathtext '$3^{4}\\cdot 6$'."""
    import re

    body = re.sub(r"\^(\d+)", r"^{\1}", name).replace(".", r"\cdot ")
    return f"${body}$"


def _title(ax, lat):
    ax.set_title(f"{vertex_symbol(lat.name)}\n{lat.alias}", pad=3)


def draw_tiling(ax, lat, reps: int = 3, window: float = 3.2):
    colors = lat.sublattice_coloring()
    cells = [(n1, n2) for n1 in range(-reps, reps + 1) for n2 in range(-reps, reps + 1)]
    segs = []
    for n1, n2 in cells:
        o = n1 * lat.a1 + n2 * lat.a2
        for b in lat.bonds:
            p = lat.basis[b.i] + o
            segs.append((p, p + lat.bond_vector(b)))
    for p, q in segs:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=0.8, zorder=1,
                solid_capstyle="round")
    for n1, n2 in cells:
        o = n1 * lat.a1 + n2 * lat.a2
        for k, r in enumerate(lat.basis):
            c = ORANGE if colors and colors[k] == 1 else BLUE
            ax.scatter(*(r + o), s=9, color=c, zorder=2, linewidths=0)
    # primitive cell
    corners = np.array([[0, 0], lat.a1, lat.a1 + lat.a2, lat.a2, [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], color=INK, lw=0.6, ls="--", zorder=3)
    center = (lat.a1 + lat.a2) / 2
    ax.set_xlim(center[0] - window, center[0] + window)
    ax.set_ylim(center[1] - window, center[1] + window)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    _title(ax, lat)


def draw_bands(ax, lat):
    bs = band_structure(lat, t=1.0, n_per_segment=80)
    flat = set(np.round(analyze_bands(lat).flat_bands, 4))
    for n in range(bs.energies.shape[1]):
        e = bs.energies[:, n]
        is_flat = np.ptp(e) < 1e-6 and round(float(e.mean()), 4) in flat
        ax.plot(bs.kdist, e, color=ORANGE if is_flat else BLUE, lw=1.3 if is_flat else 1.0)
    for x in bs.label_positions:
        ax.axvline(x, color=GRID, lw=0.6, zorder=0)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.set_xticks(bs.label_positions, [LABELS[label] for label in bs.labels])
    ax.set_xlim(bs.kdist[0], bs.kdist[-1])
    ax.set_ylim(-6.4, 4.4)
    _title(ax, lat)


def grid(draw, legend_handles, path: Path, ylabel: str | None = None):
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 6.2), constrained_layout=True)
    lats = list(ARCHIMEDEAN.values())
    for ax, lat in zip(axes.flat, lats):
        draw(ax, lat)
        if ylabel and ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
    legend_ax = axes.flat[len(lats)]
    legend_ax.axis("off")
    legend_ax.legend(handles=legend_handles, loc="center", frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main(out: Path) -> None:
    from matplotlib.lines import Line2D

    out.mkdir(parents=True, exist_ok=True)
    grid(draw_tiling, [
        Line2D([], [], marker="o", ls="", color=BLUE, label="site (sublattice A)"),
        Line2D([], [], marker="o", ls="", color=ORANGE, label="sublattice B (bipartite)"),
        Line2D([], [], color=MUTED, lw=0.8, label="nearest-neighbour bond"),
        Line2D([], [], color=INK, lw=0.6, ls="--", label="primitive cell"),
    ], out / "archimedean_tilings")
    grid(draw_bands, [
        Line2D([], [], color=BLUE, lw=1.0, label="dispersive band"),
        Line2D([], [], color=ORANGE, lw=1.3, label="flat band"),
    ], out / "archimedean_bands", ylabel=r"$E/t$")
    print(f"figures written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "figures")
    main(p.parse_args().out)
