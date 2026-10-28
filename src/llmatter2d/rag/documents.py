"""Turn :class:`Material2D` records into text "cards" and filterable metadata.

A card is what the LLM reads. It keeps every number next to its method and
source so answers can be cited and checked.
"""

from __future__ import annotations

from typing import Any

from ..schema import Material2D


def _fmt(x: float | None, unit: str = "", digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}g}{(' ' + unit) if unit else ''}"


def material_to_card(m: Material2D) -> str:
    lines = [f"Material: {m.display_name}", f"ID: {m.uid}",
             f"Elements: {', '.join(m.elements)}"]
    if m.tags:
        lines.append(f"Tags: {', '.join(m.tags)}")
    if m.structure is not None:
        s = m.structure
        proto = f"; prototype {s.prototype}" if s.prototype else ""
        lines.append(f"Structure: {s.describe()}{proto}")

    e = m.electronic
    if e.band_gaps:
        gaps = []
        for g in e.band_gaps:
            extra = [x for x in (("direct" if g.direct else "indirect") if g.direct is not None
                                 else None, g.location, "with SOC" if g.with_soc else None) if x]
            gaps.append(f"{g.value_eV:.3g} eV ({g.method}{'; ' + ', '.join(extra) if extra else ''})")
        lines.append("Band gap: " + "; ".join(gaps))
    elec = []
    if e.is_metal is not None:
        elec.append("metallic" if e.is_metal else "non-metallic")
    if e.dirac_cone:
        elec.append("Dirac cone")
    if e.soc_gap_eV is not None:
        elec.append(f"SOC-induced gap {_fmt(e.soc_gap_eV, 'eV')}")
    if e.topology:
        elec.append(f"topology: {e.topology}")
    if e.magnetic_state:
        elec.append(f"magnetic state: {e.magnetic_state}")
    if e.magnetic_moment_muB is not None:
        elec.append(f"magnetic moment {_fmt(e.magnetic_moment_muB, 'muB/cell')}")
    if e.curie_temperature_K is not None:
        elec.append(f"Curie temperature {_fmt(e.curie_temperature_K, 'K')}")
    for key, val in e.effective_masses.items():
        elec.append(f"m*({key}) = {val:.3g} m_e")
    if elec:
        lines.append("Electronic: " + "; ".join(elec))

    st = m.stability
    stab = []
    if st.formation_energy_eV_atom is not None:
        stab.append(f"formation energy {_fmt(st.formation_energy_eV_atom, 'eV/atom')}")
    if st.energy_above_hull_eV_atom is not None:
        stab.append(f"E above hull {_fmt(st.energy_above_hull_eV_atom, 'eV/atom')}")
    if st.exfoliation_energy_meV_atom is not None:
        stab.append(f"exfoliation energy {_fmt(st.exfoliation_energy_meV_atom, 'meV/atom')}")
    if st.dynamically_stable is not None:
        stab.append("dynamically stable" if st.dynamically_stable else "dynamically UNSTABLE")
    if st.experimentally_realized is not None:
        stab.append("experimentally realized" if st.experimentally_realized
                    else "not yet synthesized")
    if stab:
        lines.append("Stability: " + "; ".join(stab))

    for tb in m.tight_binding:
        lines.append(f"Tight-binding: {tb.summary()}")
    for a in m.arpes:
        parts = []
        if a.fermi_velocity_m_s is not None:
            parts.append(f"Fermi velocity {a.fermi_velocity_m_s:.3g} m/s")
        if a.effective_mass_me is not None:
            parts.append(f"effective mass {a.effective_mass_me:.3g} m_e")
        if a.gap_eV is not None:
            parts.append(f"gap {a.gap_eV:.3g} eV")
        if a.soc_splitting_eV is not None:
            parts.append(f"SOC splitting {a.soc_splitting_eV:.3g} eV")
        if a.k_point:
            parts.append(f"at {a.k_point}")
        for label, val in (("hv", a.photon_energy_eV), ("T", a.temperature_K)):
            if val is not None:
                parts.append(f"{label}={val:g}")
        if a.substrate:
            parts.append(f"substrate {a.substrate}")
        if a.notes:
            parts.append(a.notes)
        ref = a.provenance.reference if a.provenance and a.provenance.reference else None
        lines.append("ARPES: " + "; ".join(parts) + (f" [{ref}]" if ref else ""))

    srcs = []
    for p in m.provenance:
        s = p.source + (f" {p.source_id}" if p.source_id else "")
        if p.method:
            s += f", {p.method}"
        if p.reference:
            s += f"; {p.reference}"
        srcs.append(s)
    if srcs:
        lines.append("Sources: " + " | ".join(srcs))
    if m.notes:
        lines.append(f"Notes: {m.notes}")
    return "\n".join(lines)


def material_metadata(m: Material2D) -> dict[str, Any]:
    gaps = [g.value_eV for g in m.electronic.band_gaps]
    s = m.structure
    return {
        "uid": m.uid,
        "formula": m.formula,
        "elements": m.elements,
        "tags": [t.lower() for t in m.tags],
        "gap_min": min(gaps) if gaps else None,
        "gap_max": max(gaps) if gaps else None,
        "magnetic": (m.electronic.magnetic_state not in (None, "NM")) if
        m.electronic.magnetic_state is not None else None,
        "lattice": s.bravais_lattice() if s is not None else None,
        "prototype": s.prototype if s is not None else None,
        "sources": sorted({p.source for p in m.provenance}),
        "has_arpes": bool(m.arpes),
        "has_tight_binding": bool(m.tight_binding),
    }
