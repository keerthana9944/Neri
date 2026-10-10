import re


# ============================================================
# Safety-Critical Terms
# ============================================================

SAFETY_PHRASES = {
    "electrical spark",
    "electric spark",
    "electrical sparking",
    "electric sparking",
    "electric shock",
    "electrical shock",
    "exposed wiring",
    "exposed wire",
    "exposed electrical",
    "cable spark",
    "cable sparking",
    "arc flash",
    "voltage leak",
    "high voltage",
    "short circuit",
    "smoke",
    "fire",
    "flame",
    "burning smell",
    "combustion",
    "explosion",
    "exploding",
    "gas leak",
    "chemical leak",
    "chemical exposure",
    "toxic gas",
    "toxic fumes",
    "pressurized leak",
    "pressurized line",
    "line rupture",
    "pipe rupture",
    "valve blowout",
    "fluid spray",
    "hydraulic leak",
    "serious injury",
    "injury",
    "emergency",
    "unsafe",
    "danger",
    "hazard",
    "entrapment",
    "crush hazard",
    "uncontrolled movement",
    "blade guard loose",
}


SAFETY_WORDS = {
    "spark",
    "sparking",
    "sparks",
    "smoke",
    "fire",
    "flames",
    "shock",
    "electrocution",
    "injury",
    "emergency",
    "unsafe",
    "danger",
    "hazard",
    "hazards",
    "entrapment",
    "explosion",
    "rupture",
    "blowout",
}


# ============================================================
# Safety Detection
# ============================================================

def _contains_phrase(
    text: str,
    phrase: str,
) -> bool:
    """
    Check whether a complete safety phrase exists
    in the text.
    """

    pattern = (
        r"\b"
        + re.escape(phrase)
        + r"\b"
    )

    return re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    ) is not None


def is_safety_critical(
    query: str,
) -> bool:
    """
    Determine whether a troubleshooting request
    contains a safety-critical condition.

    Normal operational problems such as:
        - low pressure
        - pressure fluctuation
        - overheating
        - slow operation

    are NOT automatically treated as safety-critical.

    Explicit hazards such as:
        - electrical spark / arc flash
        - smoke / fire / burning smell
        - electric shock / exposed wiring
        - line rupture / toxic leak
        - injury / emergency

    are treated as safety-critical.
    """

    if not query or not query.strip():
        return False

    query_lower = query.lower()

    # Check multi-word safety phrases first.
    for phrase in SAFETY_PHRASES:

        if _contains_phrase(
            query_lower,
            phrase,
        ):
            return True

    # Check individual safety words.
    for word in SAFETY_WORDS:

        if _contains_phrase(
            query_lower,
            word,
        ):
            return True

    return False


# ============================================================
# Hazard Domains & Safety Verification
# ============================================================

HAZARD_DOMAINS = {
    "electrical": {
        "hazard_terms": [
            "spark", "sparking", "sparks", "arc flash", "shock", "electrocution",
            "voltage", "wire", "wiring", "electrical", "short circuit", "cable spark"
        ],
        "mitigation_terms": [
            "electrical", "electric", "voltage", "de-energiz", "power supply",
            "circuit breaker", "lockout", "tagout", "loto", "grounding", "shock",
            "arc flash", "insulat", "disconnect power"
        ],
    },
    "fire_thermal": {
        "hazard_terms": [
            "smoke", "fire", "flame", "flames", "burning", "burning smell",
            "combustion", "explosion", "exploding"
        ],
        "mitigation_terms": [
            "fire", "extinguisher", "flame", "smoke", "evacuat", "thermal",
            "heat", "burn", "suppression", "combustible", "flammable", "shutdown"
        ],
    },
    "chemical_gas": {
        "hazard_terms": [
            "gas leak", "chemical leak", "chemical exposure", "toxic gas",
            "toxic fumes", "fumes", "chemical"
        ],
        "mitigation_terms": [
            "chemical", "gas", "toxic", "ventilation", "respirat", "msds",
            "spill", "containment", "inhalation", "eyewash", "hazardous material"
        ],
    },
    "pressure_hydraulic": {
        "hazard_terms": [
            "pressurized leak", "pressurized line", "hydraulic leak",
            "line rupture", "pipe rupture", "valve blowout", "fluid spray",
            "high pressure"
        ],
        "mitigation_terms": [
            "pressure", "depressuriz", "hydraulic", "relief valve", "fluid injection",
            "bleed", "pneumatic", "burst", "rupture", "accumulator"
        ],
    },
    "mechanical_crush": {
        "hazard_terms": [
            "entrapment", "crush hazard", "blade guard loose", "uncontrolled movement",
            "pinch point", "moving parts"
        ],
        "mitigation_terms": [
            "guard", "interlock", "pinch", "crush", "entrapment", "emergency stop",
            "e-stop", "zero energy", "moving parts", "barrier", "lockout"
        ],
    },
}


def _detect_hazard_domains(text: str) -> list[str]:
    """Identify which hazard categories are present in the text."""
    text_lower = text.lower()
    matched_domains = []
    for domain, config in HAZARD_DOMAINS.items():
        for term in config["hazard_terms"]:
            if _contains_phrase(text_lower, term):
                matched_domains.append(domain)
                break
    return matched_domains


def is_safety_chunk_relevant_to_hazard(chunk: dict, query: str | None = None) -> bool:
    """
    Check whether a safety chunk contains substantive, relevant safety guidance
    that specifically addresses the hazard condition in the query.
    """
    metadata = chunk.get("metadata", {})
    if metadata.get("document_type") != "safety":
        return False

    text = chunk.get("text", "").strip()
    if not text or len(text) < 25:
        return False

    # Distance calibration:
    # In ChromaDB, the collection uses squared L2 distance: D^2 = 2 * (1 - cos_sim)
    # for unit-normalized embeddings. A threshold of 0.70 corresponds to
    # cos_sim >= 0.65 (conservative semantic relevance). Chunks exceeding 0.70
    # represent weak semantic similarity and must not authorize hazardous actions.
    distance = chunk.get("distance")
    if distance is not None and distance > 0.70:
        return False

    if not query or not query.strip():
        # Fallback when query is not provided: verify chunk has non-trivial safety guidance
        return True

    detected_domains = _detect_hazard_domains(query)

    # If specific hazard domains were detected, the safety chunk must contain
    # mitigation guidance for at least one of those domains.
    if detected_domains:
        text_lower = text.lower()
        for domain in detected_domains:
            mitigation_terms = HAZARD_DOMAINS[domain]["mitigation_terms"]
            if any(_contains_phrase(text_lower, term) or term in text_lower for term in mitigation_terms):
                return True
        return False

    # If query is safety-critical but matched general emergency/injury words
    # (e.g. "serious injury", "emergency", "unsafe", "danger"):
    # Require generic emergency/safety protocol guidance in the chunk.
    general_safety_terms = [
        "emergency stop", "e-stop", "lockout", "tagout", "loto",
        "evacuat", "first aid", "shutdown", "isolate", "ppe",
        "safety procedure", "hazard"
    ]
    text_lower = text.lower()
    return any(term in text_lower for term in general_safety_terms)


# ============================================================
# Evidence Checks
# ============================================================

def has_safety_evidence(
    retrieved_chunks: list[dict],
    query: str | None = None,
) -> bool:
    """
    Check whether retrieved evidence contains valid, approved safety documentation
    that is genuinely relevant to the requested query hazard.
    """
    if not retrieved_chunks:
        return False

    for chunk in retrieved_chunks:
        if is_safety_chunk_relevant_to_hazard(chunk, query):
            return True

    return False



def has_reliable_evidence(
    retrieved_chunks: list[dict],
) -> bool:
    """
    Check whether usable documentation
    was retrieved.
    """

    if not retrieved_chunks:
        return False

    for chunk in retrieved_chunks:

        text = chunk.get(
            "text",
            "",
        ).strip()

        if text:
            return True

    return False


# ============================================================
# No-Answer Responses
# ============================================================

def build_no_answer_response() -> dict:
    """
    Build the standard response when Neri
    cannot find enough approved documentation.
    """

    return {
        "reliable": False,
        "message": (
            "We couldn't find enough information in "
            "approved documentation to provide a reliable answer."
        ),
        "possible_causes": [],
        "safety_warning": None,
        "troubleshooting_steps": [],
        "sources": [],
    }


def build_safety_evidence_response() -> dict:
    """
    Build a response when a request is safety-critical
    but approved safety documentation is unavailable.
    """

    return {
        "reliable": False,
        "message": (
            "This request may involve a safety-critical "
            "condition, but no approved safety documentation "
            "was found. Neri cannot provide safety instructions "
            "without supporting safety documentation."
        ),
        "possible_causes": [],
        "safety_warning": (
            "Follow the organization's approved emergency "
            "and safety procedures before continuing."
        ),
        "troubleshooting_steps": [],
        "sources": [],
    }