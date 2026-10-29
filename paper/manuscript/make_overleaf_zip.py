"""Bundle the manuscript into a flat zip for Overleaf (New Project -> Upload Project).

    python paper/manuscript/make_overleaf_zip.py

Collects main.tex, references.bib, the figure PDFs and the LaTeX tables into
paper/manuscript/LLMatter2D_overleaf.zip (git-ignored). Regenerate the figures and tables
first with `python paper/reproduce.py` if the code changed.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

HERE = Path(__file__).parent
PAPER = HERE.parent

FILES = {
    "main.tex": HERE / "main.tex",
    "references.bib": HERE / "references.bib",
    "figures/archimedean_tilings.pdf": PAPER / "figures" / "archimedean_tilings.pdf",
    "figures/archimedean_bands.pdf": PAPER / "figures" / "archimedean_bands.pdf",
    "tables/archimedean.tex": PAPER / "tables" / "archimedean.tex",
    "tables/candidates.tex": PAPER / "tables" / "candidates.tex",
    "tables/retrieval.tex": PAPER / "tables" / "retrieval.tex",
}


def main() -> Path:
    out = HERE / "LLMatter2D_overleaf.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, src in FILES.items():
            if not src.exists():
                raise FileNotFoundError(f"{src} missing; run python paper/reproduce.py first")
            zf.write(src, name)
    print(f"wrote {out} ({len(FILES)} files)")
    return out


if __name__ == "__main__":
    main()
