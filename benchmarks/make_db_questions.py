"""Generate benchmark questions from database records (MC2D, JARVIS-DFT).

    python benchmarks/make_db_questions.py data/raw/mc2d.jsonl data/raw/jarvis.jsonl \
        -o benchmarks/db_questions.jsonl

Each question names the database, because the reference answer is the value stored in
that database (a closed-book model can only approximate it). Only formulas that are
unique within their database are used, so that the question identifies one record.
Band-gap questions skip metals (gap < 0.1 eV); lattice-constant questions use hexagonal
and square cells only. Sampling is deterministic (``--seed``).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from llmatter2d.schema import Material2D, load_materials

DB_NAMES = {"MC2D": "the Materials Cloud 2D database (MC2D)", "JARVIS-DFT": "JARVIS-DFT"}
GAP_WORDING = {
    "DFT (MC2D)": "computed band gap",
    "DFT-optB88vdW": "OptB88vdW band gap",
    "DFT-TBmBJ": "TBmBJ band gap",
    "DFT-HSE06": "HSE06 band gap",
}
# questions per (source, kind); kinds are "gap:<method>" or "lattice"
DEFAULT_PLAN = {
    "MC2D": {"gap:DFT (MC2D)": 100, "lattice": 50},
    "JARVIS-DFT": {"gap:DFT-optB88vdW": 50, "gap:DFT-TBmBJ": 50, "lattice": 50},
}
MIN_GAP = 0.1


def _source(m: Material2D) -> str:
    return m.provenance[0].source if m.provenance else "unknown"


def candidates(materials: list[Material2D]) -> dict[tuple[str, str], list[dict]]:
    by_source: dict[str, list[Material2D]] = defaultdict(list)
    for m in materials:
        by_source[_source(m)].append(m)
    pool: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for src, mats in by_source.items():
        if src not in DB_NAMES:
            continue
        counts = Counter(m.formula for m in mats)
        for m in mats:
            if counts[m.formula] != 1:
                continue
            db = DB_NAMES[src]
            for g in m.electronic.band_gaps:
                if g.method in GAP_WORDING and g.value_eV >= MIN_GAP:
                    pool[(src, f"gap:{g.method}")].append({
                        "question": f"What is the {GAP_WORDING[g.method]} of monolayer "
                                    f"{m.formula} in {db}?",
                        "category": "band gap (database)",
                        "relevant": [m.uid],
                        "answer_value": round(g.value_eV, 4),
                        "unit": "eV",
                        "tolerance": round(max(0.05, 0.05 * g.value_eV), 4),
                    })
            s = m.structure
            if s is not None and s.bravais_lattice() in ("hexagonal", "square"):
                pool[(src, "lattice")].append({
                    "question": f"What is the in-plane lattice constant of monolayer "
                                f"{m.formula} in {db}?",
                    "category": "structure (database)",
                    "relevant": [m.uid],
                    "answer_value": round(s.a, 4),
                    "unit": "A",
                    "tolerance": round(max(0.03, 0.02 * s.a), 4),
                })
    return pool


def make_questions(materials: list[Material2D], plan: dict = DEFAULT_PLAN,
                   seed: int = 2026) -> list[dict]:
    rng = random.Random(seed)
    pool = candidates(materials)
    out = []
    for src, kinds in plan.items():
        for kind, n in kinds.items():
            items = sorted(pool.get((src, kind), []), key=lambda q: q["relevant"][0])
            chosen = rng.sample(items, min(n, len(items)))
            tag = "mc2d" if src == "MC2D" else "jarvis"
            label = kind.split(":")[0] if kind == "lattice" else (
                "gap-" + kind.split(":")[1].split("-")[-1].split()[0].strip("()").lower())
            for i, q in enumerate(chosen, 1):
                out.append({"id": f"{tag}-{label}-{i:03d}", **q})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("materials", nargs="+", help="JSON-lines files from `llmatter2d ingest`")
    p.add_argument("-o", "--output", default="benchmarks/db_questions.jsonl")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    materials = [m for path in a.materials for m in load_materials(path)]
    questions = make_questions(materials, seed=a.seed)
    with open(a.output, "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    counts = Counter(q["id"].rsplit("-", 1)[0] for q in questions)
    print(f"wrote {len(questions)} questions to {a.output}: {dict(counts)}")


if __name__ == "__main__":
    main()
