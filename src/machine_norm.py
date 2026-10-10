import re


# ============================================================
# Canonical Machine Definitions & Aliases
# ============================================================

KNOWN_MACHINES = {
    "cnc-mill": {
        "canonical_id": "cnc-mill",
        "display_name": "CNC Mill",
        "aliases": [
            "cnc-001", "cnc-mill", "cnc_mill", "cnc mill",
            "cnc-milling-machine", "cnc milling machine",
            "haas vf-2", "haas-vf-2", "vf-2"
        ],
    },
    "cnc-lathe": {
        "canonical_id": "cnc-lathe",
        "display_name": "CNC Lathe",
        "aliases": [
            "lat-001", "cnc-lathe", "cnc_lathe", "cnc lathe",
            "CNC lathe", "dmg mori", "nlx 2500", "nlx-2500"
        ],
    },
    "hydraulic-press": {
        "canonical_id": "hydraulic-press",
        "display_name": "Hydraulic Press",
        "aliases": [
            "prs-001", "hydraulic-press", "hydraulic_press",
            "hydraulic press", "schuler", "hp-160", "hp 160"
        ],
    },
    "industrial-drill": {
        "canonical_id": "industrial-drill",
        "display_name": "Industrial Drill",
        "aliases": [
            "drl-001", "industrial-drill", "industrial_drill",
            "industrial drill", "bosch rexroth", "id-500", "id 500"
        ],
    },
    "robotic-welding-cell": {
        "canonical_id": "robotic-welding-cell",
        "display_name": "Robotic Welding Cell",
        "aliases": [
            "robotic-welding-cell", "robotic welding cell", "weld-001"
        ],
    },
    "air-compressor": {
        "canonical_id": "air-compressor",
        "display_name": "Air Compressor",
        "aliases": [
            "air-compressor", "air compressor", "comp-001"
        ],
    },
    "conveyor-system": {
        "canonical_id": "conveyor-system",
        "display_name": "Conveyor System",
        "aliases": [
            "conveyor-system", "conveyor system", "conv-001"
        ],
    },
    "injection-molding-machine": {
        "canonical_id": "injection-molding-machine",
        "display_name": "Injection Molding Machine",
        "aliases": [
            "injection-molding-machine", "injection molding machine", "imm-001"
        ],
    },
    "surface-grinder": {
        "canonical_id": "surface-grinder",
        "display_name": "Surface Grinder",
        "aliases": [
            "surface-grinder", "surface grinder", "grd-001"
        ],
    },
    "milling-machine": {
        "canonical_id": "milling-machine",
        "display_name": "Milling Machine",
        "aliases": [
            "milling-machine", "milling machine"
        ],
    },
}

# Reverse lookup mapping from clean slug to canonical ID
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for entry in KNOWN_MACHINES.values():
    cid = str(entry["canonical_id"])
    for alias in entry["aliases"]:
        slug = re.sub(r"[^a-z0-9]+", "-", str(alias).strip().lower()).strip("-")
        _ALIAS_TO_CANONICAL[slug] = cid


def normalize_machine_id(raw_id: str | None) -> str:
    """
    Normalize machine identifier to a canonical format.

    - Trims whitespace and lowercases.
    - Resolves recognized equipment aliases (e.g. CNC-001 -> cnc-mill).
    - Preserves unique slugs for unknown/custom machines without merging.
    - Returns 'unknown' for empty or None inputs.
    """
    if not raw_id or not str(raw_id).strip():
        return "unknown"

    cleaned = str(raw_id).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")

    if not slug:
        return "unknown"

    if slug in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[slug]

    return slug


def get_machine_retrieval_aliases(raw_id: str | None) -> list[str]:
    """
    Return all valid string variants that may match this machine in
    either new canonical metadata or legacy indexed ChromaDB metadata.
    """
    canonical = normalize_machine_id(raw_id)
    if canonical == "unknown":
        return []

    variants = {canonical}

    # Include known display names and legacy formatting
    if canonical in KNOWN_MACHINES:
        entry = KNOWN_MACHINES[canonical]
        disp_name = str(entry["display_name"])
        variants.add(disp_name)
        variants.add(disp_name.lower())
        variants.add(canonical.replace("-", " "))
        variants.add(canonical.replace("-", "_"))
        # Include all explicitly registered aliases (including legacy casings like 'CNC lathe')
        for alias in entry.get("aliases", []):
            variants.add(alias)
            variants.add(alias.strip())

    if raw_id:
        raw_clean = str(raw_id).strip()
        variants.add(raw_clean)
        variants.add(raw_clean.lower())
        variants.add(raw_clean.title())

    return sorted(list(variants))


def resolve_canonical_machine(
    machine: str | None,
    machine_id: str | None = None,
) -> str:
    """
    Safely resolve canonical machine identifier from either machine or machine_id input.
    - If machine matches a known machine alias, returns its canonical ID.
    - Else if machine_id matches a known machine alias (e.g. CNC-001 -> cnc-mill), returns its canonical ID.
    - Else if machine is provided and non-empty, returns normalize_machine_id(machine).
    - Else if machine_id is provided and non-empty, returns normalize_machine_id(machine_id).
    - Otherwise returns 'unknown'.
    """
    m_norm = normalize_machine_id(machine)
    if m_norm != "unknown" and m_norm in KNOWN_MACHINES:
        return m_norm

    mid_norm = normalize_machine_id(machine_id)
    if mid_norm != "unknown" and mid_norm in KNOWN_MACHINES:
        return mid_norm

    if m_norm != "unknown":
        return m_norm

    if mid_norm != "unknown":
        return mid_norm

    return "unknown"


def generate_document_id(
    document_type: str,
    machine: str | None,
    filename: str,
) -> str:
    """
    Generate a collision-resistant deterministic document identifier.
    Format: {doc_type}_{canonical_machine}_{clean_filename}
    """
    from pathlib import Path

    doc_type_clean = (
        re.sub(r"[^a-z0-9]+", "_", str(document_type).strip().lower()).strip("_")
        or "doc"
    )
    norm_machine = normalize_machine_id(machine)
    clean_fname = Path(filename).name.strip().lower()
    clean_stem = re.sub(r"[^a-z0-9.]+", "_", clean_fname).strip("_") or "file"
    return f"{doc_type_clean}_{norm_machine}_{clean_stem}"


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Generate deterministic, collision-resistant chunk identifier.
    Format: {doc_id}_c{chunk_index}
    """
    return f"{doc_id}_c{chunk_index}"

