import json
import runpy
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "paper" / "scripts" / "make_kb_snapshot.py"


def test_snapshot_bundles_records_kbs_and_readme(tmp_path):
    rec = {"uid": "mc2d:1", "provenance": [{"source": "MC2D", "retrieved": "2026-10-29"}]}
    for rel in ("data/raw/mc2d.jsonl", "data/raw/jarvis.jsonl"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    for d in ("kb/hashing", "kb/bge-small"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "config.json").write_text("{}", encoding="utf-8")

    out = runpy.run_path(str(SCRIPT))["main"](tmp_path, tmp_path / "dist")
    assert out.name == "LLMatter2D-kb-2026-10-29.zip"
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        readme = zf.read("README.md").decode("utf-8")
    assert {"data/raw/mc2d.jsonl", "kb/hashing/config.json", "kb/bge-small/config.json"} <= names
    assert "CC BY 4.0" in readme and "Campi et al." in readme and "figshare.6815705" in readme
    assert readme.count("  data/raw/") == 2 and "1 records" in readme
