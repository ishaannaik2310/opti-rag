"""
Centralized prompt formatting for OptiRAG-lite.
Enforces strict concise QA instructions across all scripts and pipeline stages.
"""

from typing import List, Tuple, Dict, Any, Union


SHARED_INSTRUCTION = (
    "Answer the question using only the context. Reply with ONLY the answer: "
    "a few words, a name, a number, a date, or yes/no. No explanation and no full sentences. "
    "If the context does not contain the answer, reply exactly: I don't know."
)


def build_rag_prompt(
    question: str,
    chunks: Union[List[Tuple[str, str, float]], List[Dict[str, Any]]]
) -> str:
    """
    Builds the standardized RAG prompt for HotpotQA questions.

    Args:
        question: HotpotQA query string.
        chunks: List of retrieved chunks either as (text, title, score) tuples
                or dictionaries with 'text' and 'title' keys.

    Returns:
        Formatted prompt string.
    """
    formatted_chunks = []
    for i, chunk in enumerate(chunks, 1):
        if isinstance(chunk, tuple):
            text, title = chunk[0], chunk[1]
        elif isinstance(chunk, dict):
            text, title = chunk.get("text", ""), chunk.get("title", "")
        else:
            raise ValueError(f"Unsupported chunk type: {type(chunk)}")
        formatted_chunks.append(f"[{i}] Title: {title}\n{text}")

    context_str = "\n\n".join(formatted_chunks)
    return (
        f"{SHARED_INSTRUCTION}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
