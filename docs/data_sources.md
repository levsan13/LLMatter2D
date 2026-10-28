# Data sources

| Source | Access | Connector | Content mapped to `Material2D` |
|---|---|---|---|
| **MC2D** (Materials Cloud) | Public OPTIMADE API, `https://optimade.materialscloud.org/main/mc2d` | `llmatter2d ingest mc2d` | structure, band gap (`_mcloud_band_gap`) |
| **C2DB** (DTU) | ASE database `c2db.db`; per the [2dhub.org](https://2dhub.org/c2db/c2db.html) page the full dataset is provided on request | `llmatter2d ingest c2db --path c2db.db` | structure, PBE / HSE06 / G0W0 gaps, magnetic state, formation energy, energy above hull |
| **2DMatPedia** | JSON dump from the 2DMatPedia website (one record per line) | `llmatter2d ingest 2dmatpedia --path db.json` | structure, PBE gap, exfoliation and decomposition energies, magnetisation, discovery process |
| **JARVIS-DFT 2D** (NIST) | `jarvis-tools` (`dft_2d` dataset, downloaded on first use) | `llmatter2d ingest jarvis` | structure, optB88vdW / TBmBJ / HSE gaps, formation and exfoliation energies, magnetic moment, spin-orbit spillage |
| **Wannier90** | Your own `_hr.dat` files | `llmatter2d.ingest.wannier.read_wannier90_hr` | full tight-binding model (hoppings divided by the R-point degeneracy) |
| **ARPES** | Your own maps as `.npz` or HDF5 (`k`, `energy`, `intensity`) | `llmatter2d.ingest.arpes.features_from_map` | Fermi velocity, effective mass, gaps, spin-orbit splitting, with measurement metadata |
| **Seed set** | Shipped with the package | `llmatter2d ingest seed` | 13 well-known monolayers with approximate literature values (demo/testing only) |

## Notes

* Field names for C2DB (`gap`, `gap_hse`, `gap_gw`, `magstate`, `hform`, `ehull`, `uid`),
  2DMatPedia and JARVIS follow the respective documentation. The connectors read them with
  `.get()`, so a missing field becomes an empty value instead of an error. Check a few
  records after the first ingestion of a new database release.
* Records from different sources are kept apart (`uid = "<source>:<id>"`). Cross-source
  merging of the same material is supported by `Material2D.merge` but is not automatic,
  because matching structures across databases needs a structure-matching step (planned).
* Licences: each database has its own terms of use and citation requests. Do not commit
  downloaded databases to this repository (`data/raw/` is git-ignored).

## ARPES data

There is no single public database of ARPES dispersions for 2D materials yet. The
intended sources are:

1. your own measurements, converted to `(k, E, I)` arrays (for example with PyARPES);
2. values reported in papers, entered as `ARPESFeatures` with the DOI in `provenance`;
3. literature mining (see arXiv:2511.20691 in [prior_art.md](prior_art.md)).

The LLM only sees extracted quantities with their measurement conditions (photon energy,
temperature, polarisation, substrate), never raw spectra.
