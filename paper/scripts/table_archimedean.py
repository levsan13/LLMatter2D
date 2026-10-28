"""Table: geometry and nearest-neighbour band features of the 11 Archimedean tilings.

    python paper/scripts/table_archimedean.py [--out paper/tables]

Writes archimedean.md (Markdown) and archimedean.tex (LaTeX tabular, booktabs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llmatter2d.lattices import ARCHIMEDEAN, analyze_bands


def rows() -> list[dict]:
    out = []
    for name, lat in ARCHIMEDEAN.items():
        f = analyze_bands(lat)
        out.append({
            "tiling": name,
            "alias": lat.alias,
            "sites": lat.n_sites,
            "z": int(lat.coordination()[0]),
            "density": lat.n_sites / lat.area,  # sites per unit area, unit bond
            "bipartite": "yes" if lat.sublattice_coloring() is not None else "no",
            "flat": ", ".join(f"{e:+.0f}" for e in f.flat_bands) or "-",
            "dirac": ", ".join(sorted({s.split(":")[0] for s in f.linear_crossings})) or "-",
            "quadratic": ", ".join(sorted({s.split(":")[0] for s in f.quadratic_touchings}))
            or "-",
            "multifold": ", ".join(sorted({s.split(":")[0] for s in f.multifold_degeneracies}))
            or "-",
        })
    return out


def to_markdown(data: list[dict]) -> str:
    head = ("| Tiling | Name | Sites/cell | z | Density (d^-2) | Bipartite* | Flat bands (E/t) "
            "| Linear crossings at | Quadratic touchings at | >2-fold degeneracies at |")
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for r in data:
        lines.append(f"| {r['tiling']} | {r['alias']} | {r['sites']} | {r['z']} | "
                     f"{r['density']:.3f} | {r['bipartite']} | {r['flat']} | {r['dirac']} | "
                     f"{r['quadratic']} | {r['multifold']} |")
    lines.append("")
    lines.append("*A/B decoration possible within the primitive cell. G = Gamma. "
                 "Nearest-neighbour hopping -t, unit bond length d.")
    return "\n".join(lines) + "\n"


def to_latex(data: list[dict]) -> str:
    def tex(s: str) -> str:
        s = s.replace("G", r"$\Gamma$")
        return s

    def sym(name: str) -> str:
        return "$" + name.replace(".", r"\!\cdot\!") + "$"

    lines = [r"\begin{tabular}{llrrrlcccc}", r"\toprule",
             r"Tiling & Name & $N$ & $z$ & $\rho d^2$ & Bip. & Flat ($E/t$) & Linear & "
             r"Quadratic & Multifold \\", r"\midrule"]
    for r in data:
        lines.append(f"{sym(r['tiling'])} & {r['alias']} & {r['sites']} & {r['z']} & "
                     f"{r['density']:.3f} & {r['bipartite']} & {r['flat']} & "
                     f"{tex(r['dirac'])} & {tex(r['quadratic'])} & {tex(r['multifold'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data = rows()
    (out / "archimedean.md").write_text(to_markdown(data), encoding="utf-8")
    (out / "archimedean.tex").write_text(to_latex(data), encoding="utf-8")
    print(to_markdown(data))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "tables")
    main(p.parse_args().out)
