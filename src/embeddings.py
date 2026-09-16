import time

from openai import OpenAI, RateLimitError

from src.config import (
    OPENAI_BASE_URL,
    EMBED_MODEL,
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


def create_embedding(text: str) -> list[float]:
    """
    Create an embedding for a single piece of text.
    """

    if not text.strip():
        raise ValueError(
            "Cannot create an embedding for empty text."
        )

    max_retries = 5

    for attempt in range(max_retries):

        try:
            response = _get_client().embeddings.create(
                model=EMBED_MODEL,
                input=text,
            )

            return response.data[0].embedding

        except RateLimitError:

            if attempt == max_retries - 1:
                raise

            wait_time = 60

            print(
                f"Embedding rate limit reached. "
                f"Waiting {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Could not create embedding."
    )


def create_embeddings(
    texts: list[str],
    batch_size: int = 20,
) -> list[list[float]]:
    """
    Create embeddings in small batches.

    Smaller batches help avoid hitting Gemini's
    free-tier embedding quota too quickly.
    """

    if not texts:
        return []

    embeddings = []

    total = len(texts)

    for start in range(
        0,
        total,
        batch_size,
    ):

        batch = texts[
            start:start + batch_size
        ]

        print(
            f"Creating embeddings "
            f"{start + 1}-{min(start + len(batch), total)} "
            f"of {total}..."
        )

        max_retries = 5

        for attempt in range(max_retries):

            try:
                response = _get_client().embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                )

                batch_embeddings = [
                    item.embedding
                    for item in response.data
                ]

                embeddings.extend(
                    batch_embeddings
                )

                break

            except RateLimitError:

                if attempt == max_retries - 1:
                    raise

                wait_time = 60

                print(
                    "Embedding rate limit reached. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

        # Small pause between batches.
        if start + batch_size < total:
            time.sleep(2)

    return embeddings