"""Command-line interface: ``llmatter2d <command> ...``"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _kb(args):
    from .rag import KnowledgeBase, get_embedder

    if args.kb and Path(args.kb, "config.json").exists():
        return KnowledgeBase.load(args.kb)
    if args.kb:
        print(f"[knowledge base {args.kb!r} not found; using the built-in seed set]",
              file=sys.stderr)
    return KnowledgeBase.from_seed(get_embedder(getattr(args, "embedder", None) or "hashing"))


def _pipeline(args):
    from .llm import get_llm
    from .rag import RAGPipeline

    return RAGPipeline(_kb(args), get_llm(args.llm), k=args.k)


def _filters(args):
    from .rag import Filters

    return Filters(elements_all=args.elements or [], gap_min=args.gap_min, gap_max=args.gap_max,
                   magnetic=args.magnetic, tags_any=args.tag or [],
                   require_arpes=args.require_arpes)


def cmd_ingest(args) -> None:
    from .ingest import get_connector
    from .schema import save_materials

    conn = get_connector(args.source, path=args.path, filter=args.filter)
    n = save_materials(args.output, conn.fetch(limit=args.limit))
    print(f"wrote {n} materials from {args.source} to {args.output}")


def cmd_index(args) -> None:
    from .ingest.seed import SeedConnector
    from .rag import KnowledgeBase, get_embedder
    from .schema import load_materials

    kb = KnowledgeBase(get_embedder(args.embedder))
    if args.seed or not args.materials:
        kb.add(SeedConnector().fetch())
    for path in args.materials:
        kb.add(load_materials(path))
    kb.build()
    kb.save(args.output)
    print(f"indexed {len(kb)} materials with {kb.embedder.spec} into {args.output}")


def cmd_ask(args) -> None:
    pipe = _pipeline(args)
    answer = pipe.ask(args.question, filters=_filters(args))
    print(answer.text)
    print("\nsources: " + ", ".join(answer.sources))


def cmd_analyze(args) -> None:
    from .analysis import analyze_structure
    from .lattices.io import read_poscar

    report = analyze_structure(read_poscar(args.poscar), _pipeline(args), args.question)
    print(report.as_text())
    print("\n" + report.answer.text)


def cmd_archimedean(args) -> None:
    from .lattices import ARCHIMEDEAN, analyze_bands, archimedean_lattice, band_structure

    if not args.name:
        for name, lat in ARCHIMEDEAN.items():
            print(f"{name:10s} {lat.alias:24s} sites/cell={lat.n_sites:2d} "
                  f"z={int(lat.coordination()[0])}  bipartite={lat.sublattice_coloring() is not None}")
        return
    lat = archimedean_lattice(args.name)
    print(f"{lat.name} ({lat.alias}): {lat.n_sites} sites/cell, coordination "
          f"{int(lat.coordination()[0])}")
    print(analyze_bands(lat, t=args.t).summary())
    if args.bands:
        bs = band_structure(lat, t=args.t)
        with open(args.bands, "w") as fh:
            fh.write("k," + ",".join(f"E{n}" for n in range(bs.energies.shape[1])) + "\n")
            fh.writelines(f"{k:.6f}," + ",".join(f"{e:.6f}" for e in row) + "\n" for k, row in zip(bs.kdist, bs.energies))
        print(f"bands along {'-'.join(bs.labels)} written to {args.bands}")
    if args.species:
        from .elements import bond_length
        from .lattices.io import write_poscar

        species = args.species if len(args.species) > 1 else args.species[0]
        els = args.species if len(args.species) > 1 else args.species * 2
        d = args.bond or bond_length(els[0], els[1])
        out = args.poscar or f"{lat.alias}_{''.join(args.species)}.vasp"
        write_poscar(lat.to_structure(species, d), out)
        print(f"structure with bond length {d:.3f} A written to {out}")


def cmd_predict(args) -> None:
    from .predict import enumerate_candidates, review_with_llm, score_candidates, write_candidates

    pipe = _pipeline(args)
    cands = score_candidates(enumerate_candidates(args.elements, args.lattices,
                                                  binary=not args.no_binary), kb=pipe.kb)
    top = cands[:args.top]
    for c in top:
        print(c.describe())
    if args.output:
        print(f"\nPOSCARs and summary written to {write_candidates(top, args.output)}")
    if args.llm != "offline":
        print("\n" + review_with_llm(cands, pipe, top=args.top).text)


def cmd_wannier(args) -> None:
    from .ingest.wannier import read_wannier90_hr

    model = read_wannier90_hr(args.hr, threshold=args.threshold)
    print(model.summary())


def cmd_benchmark(args) -> None:
    import json
    import time

    from .evaluation import evaluate_item, format_summary, load_benchmark, summarize
    from .llm import LLMServerError, get_llm

    kb = _kb(args)
    if args.formula_boost is not None:
        kb.formula_boost = args.formula_boost
    needs_llm = any(m != "retrieval" for m in args.mode)
    llm = get_llm(args.llm) if needs_llm else None
    items = load_benchmark(args.questions)
    if args.limit and args.limit < len(items):
        import random

        items = random.Random(args.seed).sample(items, args.limit)

    out = Path(args.output) if args.output else None
    rows: list[dict] = []
    if out and args.resume and out.exists():  # keep finished answers, redo failed ones
        rows = [r for r in (json.loads(line) for line in
                            out.read_text(encoding="utf-8").splitlines() if line.strip())
                if not r.get("error")]
        print(f"resuming: {len(rows)} answers kept from {out}", file=sys.stderr)
    done = {(r["id"], r["mode"]) for r in rows}
    fh = None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fh = out.open("w", encoding="utf-8")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
    total = len(items) * len(args.mode)
    consecutive_errors, aborted = 0, False
    try:
        for mode in args.mode:
            for item in items:
                if (item.id, mode) in done:
                    continue
                t0 = time.perf_counter()
                for attempt in range(args.retries + 1):
                    try:
                        row = evaluate_item(item, kb, llm, mode, args.k, args.citation_style)
                        break
                    except (LLMServerError, TimeoutError, OSError) as exc:
                        row = {"id": item.id, "category": item.category, "mode": mode,
                               "error": str(exc), "correct": False}
                        if attempt < args.retries:
                            print(f"  retry {attempt + 1}/{args.retries} after error: {exc}",
                                  file=sys.stderr, flush=True)
                            time.sleep(args.retry_wait)
                row["seconds"] = round(time.perf_counter() - t0, 2)
                rows.append(row)
                if fh:  # written as we go, so a long run survives interruptions
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                if needs_llm:
                    status = "ERROR" if row.get("error") else f"correct={row.get('correct')}"
                    print(f"[{len(rows)}/{total}] {mode:11s} {item.id:28s} {status} "
                          f"({row['seconds']:.1f} s)", file=sys.stderr, flush=True)
                consecutive_errors = consecutive_errors + 1 if row.get("error") else 0
                if consecutive_errors >= 3:
                    aborted = True
                    break
            if aborted:
                break
    finally:
        if fh:
            fh.close()
    if aborted:
        sys.exit("error: the LLM server stopped answering (3 consecutive failures). "
                 "Answers so far are saved; fix the connection and rerun with --resume.")
    title = (f"{Path(args.questions).name} | {kb.embedder.spec} | boost={kb.formula_boost} "
             f"| k={args.k}" + (f" | {llm.name}" if llm else ""))
    report = format_summary(summarize(rows), title)
    print(report)
    if out:
        out.with_suffix(".md").write_text(report + "\n", encoding="utf-8")
        print(f"\nper-question results: {out}; summary: {out.with_suffix('.md')}")


def cmd_regrade(args) -> None:
    import json

    from .evaluation import format_summary, load_benchmark, regrade, summarize

    path = Path(args.results)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = regrade(rows, load_benchmark(args.questions))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    md = path.with_suffix(".md")
    title = md.read_text(encoding="utf-8").splitlines()[0].lstrip("# ") if md.exists() else ""
    report = format_summary(summarize(rows), title + " | regraded")
    md.write_text(report + "\n", encoding="utf-8")
    print(report)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmatter2d", description=__doc__)
    p.add_argument("--version", action="version", version=f"llmatter2d {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    from .llm import default_llm_spec

    def rag_options(sp):
        sp.add_argument("--kb", default="kb", help="knowledge-base directory (default: kb)")
        sp.add_argument("--llm", default=default_llm_spec(),
                        help="lmstudio (default) | lmstudio:<model> | ollama:<model> | "
                             "claude[:<model>] | offline; default can be set with LLMATTER2D_LLM")
        sp.add_argument("-k", type=int, default=8, help="records to retrieve")

    s = sub.add_parser("ingest", help="fetch materials from a source into JSON lines")
    s.add_argument("source", choices=["seed", "mc2d", "c2db", "2dmatpedia", "jarvis"])
    s.add_argument("--path", help="local database file (c2db, 2dmatpedia)")
    s.add_argument("--filter", help="OPTIMADE filter (mc2d) or ASE selection (c2db)")
    s.add_argument("--limit", type=int)
    s.add_argument("-o", "--output", required=True)
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("index", help="build a knowledge base from JSON-lines files")
    s.add_argument("materials", nargs="*")
    s.add_argument("--seed", action="store_true", help="also include the built-in seed set")
    s.add_argument("--embedder", default="hashing", help="hashing | st:<sentence-transformer>")
    s.add_argument("-o", "--output", default="kb")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("ask", help="ask a question (RAG)")
    s.add_argument("question")
    rag_options(s)
    s.add_argument("--elements", nargs="+")
    s.add_argument("--gap-min", type=float)
    s.add_argument("--gap-max", type=float)
    s.add_argument("--magnetic", action=argparse.BooleanOptionalAction, default=None)
    s.add_argument("--tag", nargs="+")
    s.add_argument("--require-arpes", action="store_true")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("analyze", help="analyse a new structure (POSCAR)")
    s.add_argument("poscar")
    s.add_argument("--question")
    rag_options(s)
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("archimedean", help="list or inspect the 11 Archimedean lattices")
    s.add_argument("name", nargs="?")
    s.add_argument("--t", type=float, default=1.0, help="hopping (eV)")
    s.add_argument("--bands", help="write band structure CSV")
    s.add_argument("--species", nargs="+", help="decorate: one element or an A B pair")
    s.add_argument("--bond", type=float, help="bond length (A); default covalent radii")
    s.add_argument("--poscar", help="output POSCAR path")
    s.set_defaults(func=cmd_archimedean)

    s = sub.add_parser("predict", help="generate and screen Archimedean-lattice candidates")
    s.add_argument("--elements", nargs="+", required=True)
    s.add_argument("--lattices", nargs="+", help="subset of tilings (default: all 11)")
    s.add_argument("--no-binary", action="store_true")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("-o", "--output", help="directory for POSCARs + candidates.json")
    rag_options(s)
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("wannier", help="summarise a Wannier90 _hr.dat file")
    s.add_argument("hr")
    s.add_argument("--threshold", type=float, default=1e-3)
    s.set_defaults(func=cmd_wannier)

    s = sub.add_parser("benchmark", help="evaluate retrieval and answers (RAG vs closed-book)")
    s.add_argument("questions", help="benchmark JSON-lines file")
    rag_options(s)
    s.add_argument("--embedder", default="hashing", help="embedder when no --kb exists")
    s.add_argument("--mode", nargs="+", default=["retrieval"],
                   choices=["retrieval", "rag", "closed_book"])
    s.add_argument("--formula-boost", type=float,
                   help="score bonus for exact formula/name matches (default 0.25; 0 disables)")
    s.add_argument("--limit", type=int, help="evaluate a random subset of N questions")
    s.add_argument("--seed", type=int, default=2026, help="seed for --limit sampling")
    s.add_argument("--citation-style", choices=["number", "id"], default="number",
                   help="ask the model to cite record numbers (mapped to ids) or the ids")
    s.add_argument("-o", "--output", help="per-question results (.jsonl) + summary (.md)")
    s.add_argument("--resume", action="store_true",
                   help="keep answers already in --output and run only the missing/failed ones")
    s.add_argument("--retries", type=int, default=2, help="retries per question on errors")
    s.add_argument("--retry-wait", type=float, default=15.0, help="seconds between retries")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("regrade", help="re-grade stored benchmark answers")
    s.add_argument("results", help="per-question results (.jsonl) written by benchmark")
    s.add_argument("questions", help="the benchmark JSON-lines file used for the run")
    s.set_defaults(func=cmd_regrade)
    return p


def main(argv: list[str] | None = None) -> None:
    from .llm import LLMServerError

    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except LLMServerError as exc:
        sys.exit(f"error: {exc}\nStart the LM Studio server (Developer tab -> Start Server, or "
                 "`lms server start`) with a model loaded, or choose another backend with --llm "
                 "(e.g. --llm offline).")


if __name__ == "__main__":
    main()
