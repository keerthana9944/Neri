from src.cache import (
    cache_response,
    get_cached_response,
)
from src.citations import build_sources
from src.config import CACHE_TTL_SECONDS
from src.guardrails import (
    build_no_answer_response,
    build_safety_evidence_response,
    has_reliable_evidence,
    has_safety_evidence,
    is_safety_critical,
)
from src.llm import generate_troubleshooting_response
from src.retrieval import retrieve_documents


def troubleshoot(
    machine: str,
    machine_id: str,
    problem: str,
    error_code: str | None = None,
    n_results: int = 5,
) -> dict:
    """
    Run the complete Neri troubleshooting pipeline.
    """

    # --------------------------------------------------
    # Step 1: Check cache
    # --------------------------------------------------

    cached_response = get_cached_response(
        machine=machine,
        machine_id=machine_id,
        problem=problem,
        error_code=error_code,
        ttl_seconds=CACHE_TTL_SECONDS,
    )

    if cached_response is not None:
        return cached_response

    # --------------------------------------------------
    # Step 2: Build retrieval query
    # --------------------------------------------------

    query_parts = [
        f"Machine: {machine}",
        f"Machine ID: {machine_id}",
        f"Problem: {problem}",
    ]

    if error_code:
        query_parts.append(
            f"Error Code: {error_code}"
        )

    query = "\n".join(query_parts)

    # --------------------------------------------------
    # Step 3: Retrieve approved documentation
    # --------------------------------------------------

    from src.machine_norm import resolve_canonical_machine

    canonical_target = resolve_canonical_machine(
        machine=machine,
        machine_id=machine_id,
    )

    retrieved_chunks = retrieve_documents(
        query=query,
        machine=canonical_target if canonical_target != "unknown" else machine,
        n_results=n_results,
    )

    # --------------------------------------------------
    # Step 4: Safety guardrail
    # --------------------------------------------------

    safety_critical = is_safety_critical(query)

    safety_evidence_available = has_safety_evidence(
        retrieved_chunks,
        query=query,
    )

    if safety_critical and not safety_evidence_available:
        response = build_safety_evidence_response()

        response["machine"] = machine
        response["machine_id"] = machine_id
        response["problem"] = problem
        response["error_code"] = error_code

        response["sources"] = build_sources(
            retrieved_chunks
        )

        cache_response(
            machine=machine,
            machine_id=machine_id,
            problem=problem,
            error_code=error_code,
            response=response,
        )

        return response

    # --------------------------------------------------
    # Step 5: General evidence check
    # --------------------------------------------------

    if not has_reliable_evidence(
        retrieved_chunks
    ):
        response = build_no_answer_response()

        response["machine"] = machine
        response["machine_id"] = machine_id
        response["problem"] = problem
        response["error_code"] = error_code

        cache_response(
            machine=machine,
            machine_id=machine_id,
            problem=problem,
            error_code=error_code,
            response=response,
        )

        return response

    # --------------------------------------------------
    # Step 6: Generate grounded response
    # --------------------------------------------------

    response = generate_troubleshooting_response(
        machine=machine,
        machine_id=machine_id,
        problem=problem,
        error_code=error_code,
        retrieved_chunks=retrieved_chunks,
        safety_evidence_available=safety_evidence_available,
    )

    # --------------------------------------------------
    # Step 7: Add source references
    # --------------------------------------------------

    response["sources"] = build_sources(
        retrieved_chunks
    )

    # --------------------------------------------------
    # Step 9: Add original request information
    # --------------------------------------------------

    response["machine"] = machine
    response["machine_id"] = machine_id
    response["problem"] = problem
    response["error_code"] = error_code

    # --------------------------------------------------
    # Step 10: Cache response
    # --------------------------------------------------

    cache_response(
        machine=machine,
        machine_id=machine_id,
        problem=problem,
        error_code=error_code,
        response=response,
    )

    return response