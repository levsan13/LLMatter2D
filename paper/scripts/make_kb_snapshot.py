"""Bundle the knowledge base used in the manuscript into a zip for a GitHub release.

    python paper/scripts/make_kb_snapshot.py            # -> dist/LLMatter2D-kb-<date>.zip

The snapshot contains the converted database records (data/raw/*.jsonl) and the two indexed
knowledge bases (kb/hashing, kb/bge-small), plus a README with the sources, licences,
citations, record counts and SHA-256 checksums. The original downloads (for instance the
JARVIS zip in data/raw/jarvis_store) are not included; they are available from the sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[2]

SOURCES = {
    "MC2D": {
        "file": "data/raw/mc2d.jsonl",
        "origin": "Materials Cloud 2D database, OPTIMADE endpoint "
                  "https://optimade.materialscloud.org/main/mc2d",
        "licence": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "cite": ["N. Mounet et al., Nat. Nanotechnol. 13, 246 (2018)",
                 "D. Campi et al., ACS Nano 17, 11268 (2023)"],
    },
    "JARVIS-DFT": {
        "file": "data/raw/jarvis.jsonl",
        "origin": "JARVIS-DFT 2D dataset, figshare doi:10.6084/m9.figshare.6815705.v8 "
                  "(file d2-12-12-2022.json.zip, via jarvis-tools)",
        "licence": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "cite": ["K. Choudhary et al., npj Comput. Mater. 6, 173 (2020)",
                 "K. Choudhary, JARVIS-DFT 2D dataset, figshare (2018), "
                 "doi:10.6084/m9.figshare.6815705.v8"],
    },
}
KB_DIRS = ["kb/hashing", "kb/bge-small"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def snapshot_files(root: Path) -> list[Path]:
    files = [root / s["file"] for s in SOURCES.values()]
    for d in KB_DIRS:
        files += sorted(p for p in (root / d).iterdir() if p.is_file())
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing (run `python paper/reproduce.py --data`): {missing}")
    return files


def build_readme(root: Path, files: list[Path]) -> tuple[str, str]:
    lines = ["# LLMatter2D knowledge-base snapshot", ""]
    dates = Counter()
    for name, s in SOURCES.items():
        recs = _records(root / s["file"])
        for r in recs:
            for p in r.get("provenance", []):
                if p.get("retrieved"):
                    dates[p["retrieved"]] += 1
        lines += [f"## {name}", "",
                  f"- File: `{s['file']}` ({len(recs)} records, converted to LLMatter2D "
                  "`Material2D` JSON lines)",
                  f"- Origin: {s['origin']}",
                  f"- Licence of the original data: {s['licence']}",
                  "- Please cite: " + "; ".join(s["cite"]), ""]
    built = max(dates) if dates else "unknown"
    lines += ["## Knowledge bases", "",
              "- `kb/hashing`: lexical feature-hashing embedder (2048 dimensions), used for the "
              "results of the manuscript.",
              "- `kb/bge-small`: dense embedder BAAI/bge-small-en-v1.5 (sentence-transformers).",
              "- Both contain the MC2D and JARVIS-DFT records above and the 13 curated seed records "
              "of LLMatter2D (MIT licence).",
              "- Load with `llmatter2d.rag.KnowledgeBase.load(\"kb/hashing\")` or use "
              "`llmatter2d benchmark ... --kb kb/hashing`.", "",
              "## Licence of this snapshot", "",
              "The records are derived from data distributed under CC BY 4.0 (converted to another "
              "format and indexed with embeddings). This snapshot is distributed under CC BY 4.0 "
              "as well: credit the original databases as listed above and LLMatter2D "
              "(doi:10.5281/zenodo.23779294).", "",
              f"Records retrieved on: {built}.", "",
              "## SHA-256 checksums", "", "```"]
    for p in files:
        lines.append(f"{sha256(p)}  {p.relative_to(root).as_posix()}")
    lines += ["```", ""]
    return "\n".join(lines), built


def main(root: Path = ROOT, out_dir: Path | None = None) -> Path:
    files = snapshot_files(root)
    readme, built = build_readme(root, files)
    out_dir = out_dir or root / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"LLMatter2D-kb-{built}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("README.md", readme)
        for p in files:
            zf.write(p, p.relative_to(root).as_posix())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files + README)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path)
    main(out_dir=p.parse_args().out_dir)
