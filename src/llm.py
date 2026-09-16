import json

from openai import OpenAI

from src.config import (
    OPENAI_BASE_URL,
    CHAT_MODEL,
    get_openai_api_key,
    validate_config,
)


def _get_client():
    validate_config()
    api_key = get_openai_api_key()
    return OpenAI(
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
    )


SYSTEM_PROMPT = """
You are Neri, an AI maintenance troubleshooting assistant
for manufacturing floor technicians.

Your job is to provide troubleshooting guidance ONLY from
the approved documentation provided in the context.

IMPORTANT RULES:

1. Never invent maintenance procedures.
2. Never invent safety instructions.
3. Do not use outside knowledge.
4. If the provided documentation does not contain enough
   evidence, say that there is not enough information.
5. Safety-critical instructions must be supported by the
   provided safety documentation.
6. Every cause and troubleshooting step must reference
   one or more provided source IDs.
7. Keep troubleshooting steps clear and practical.
8. Do not recommend autonomous machine control or repair.
9. Return ONLY valid JSON.

Use this JSON structure:

{
    "reliable": true,
    "possible_causes": [
        {
            "cause": "Cause description",
            "sources": [1]
        }
    ],
    "safety_warning": {
        "message": "Safety warning",
        "sources": [2]
    },
    "troubleshooting_steps": [
        {
            "step": "Troubleshooting step",
            "sources": [1, 2]
        }
    ]
}

If the evidence is insufficient, return:

{
    "reliable": false,
    "possible_causes": [],
    "safety_warning": null,
    "troubleshooting_steps": []
}

SAFETY RULES:
- Never invent safety warnings, safety procedures, lockout/tagout steps, or electrical/mechanical precautions.
- A safety warning or safety-critical instruction may be included only when supported by retrieved documentation whose document_type is "safety".
- If safety documentation is unavailable, do not create a safety warning from general knowledge.
- If the retrieved documentation contains a safety procedure, use it as the authoritative source for safety guidance.
- Every safety-critical instruction must be traceable to a retrieved source.

"""


def build_context(retrieved_chunks: list[dict]) -> str:
    """Build grounded context from retrieved documents."""

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = chunk["metadata"]

        context_parts.append(
            f"""
SOURCE ID: {index}
DOCUMENT: {metadata.get("document")}
DOCUMENT TYPE: {metadata.get("document_type")}
SECTION: {metadata.get("section")}
PAGE: {metadata.get("page")}

CONTENT:
{chunk["text"]}
"""
        )

    return "\n".join(context_parts)


def generate_troubleshooting_response(
    machine: str,
    machine_id: str,
    problem: str,
    error_code: str | None,
    retrieved_chunks: list[dict],
    safety_evidence_available: bool = False,
) -> dict:
    """Generate a grounded Neri troubleshooting response."""

    context = build_context(retrieved_chunks)

    user_prompt = f"""
Machine: {machine}
Machine ID: {machine_id}
Problem: {problem}
Error Code: {error_code or "Not provided"}
Safety documentation available:
{str(safety_evidence_available).lower()}

APPROVED DOCUMENTATION:

{context}

Based ONLY on the approved documentation above,
provide the troubleshooting response.
"""

    response = _get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "The model returned an empty response."
        )

    try:
        return json.loads(content)

    except json.JSONDecodeError as error:
        raise ValueError(
            "The model returned invalid JSON."
        ) from error