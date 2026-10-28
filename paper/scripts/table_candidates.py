"""Table: top-ranked new candidates from the Archimedean generator (before DFT).

    python paper/scripts/table_candidates.py [--elements B C N Si Ge Sn] [--top 12]

Writes candidates.md and candidates.tex to paper/tables. Candidates already present in
the knowledge base (graphene, hBN, Xenes, graphenylene) are listed separately as
benchmarks recovered by the novelty check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llmatter2d.predict import enumerate_candidates, score_candidates
from llmatter2d.rag import KnowledgeBase


def ranked(elements: list[str]):
    return score_candidates(enumerate_candidates(elements), kb=KnowledgeBase.from_seed())


def _features(c) -> tuple[str, str]:
    b = c.bands
    flat = ", ".join(f"{e:+.0f}" for e in b.flat_bands) or "-"
    lin = ", ".join(sorted({s.split(":")[0] for s in b.linear_crossings})) or "-"
    return flat, lin


def to_markdown(novel, known) -> str:
    lines = ["| Rank | Formula | Tiling | z | d (A) | Flat bands (E/t) | Linear crossings at "
             "| Score |", "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(novel, 1):
        flat, lin = _features(c)
        lines.append(f"| {i} | {c.formula} | {c.lattice.name} ({c.lattice.alias}) | "
                     f"{int(c.lattice.coordination()[0])} | {c.bond_length:.2f} | {flat} | {lin} "
                     f"| {c.score:.2f} |")
    lines += ["", "Recovered as known: " + ", ".join(
        f"{c.formula}@{c.lattice.name} ({c.known_as})" for c in known), ""]
    return "\n".join(lines)


def to_latex(novel) -> str:
    def sym(name: str) -> str:
        return "$" + name.replace(".", r"\!\cdot\!") + "$"

    lines = [r"\begin{tabular}{rllrrccr}", r"\toprule",
             r"\# & Formula & Tiling & $z$ & $d$ (\AA) & Flat ($E/t$) & Linear & Score \\",
             r"\midrule"]
    for i, c in enumerate(novel, 1):
        flat, lin = _features(c)
        lin = lin.replace("G", r"$\Gamma$")
        lines.append(f"{i} & {c.formula} & {sym(c.lattice.name)} & "
                     f"{int(c.lattice.coordination()[0])} & {c.bond_length:.2f} & {flat} & {lin} "
                     f"& {c.score:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(out: Path, elements: list[str], top: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cands = ranked(elements)
    novel = [c for c in cands if c.known_as is None][:top]
    known = [c for c in cands if c.known_as]
    (out / "candidates.md").write_text(to_markdown(novel, known), encoding="utf-8")
    (out / "candidates.tex").write_text(to_latex(novel), encoding="utf-8")
    print(to_markdown(novel, known))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).parents[1] / "tables")
    p.add_argument("--elements", nargs="+", default=["B", "C", "N", "Si", "Ge", "Sn"])
    p.add_argument("--top", type=int, default=12)
    a = p.parse_args()
    main(a.out, a.elements, a.top)
