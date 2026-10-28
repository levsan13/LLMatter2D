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
