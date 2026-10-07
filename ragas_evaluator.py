"""
RAGAS evaluator for the NASA mission chat system.

Scores a (question, retrieved contexts, answer) triple in real time and
returns a dictionary of metric names and values.

Metrics computed for every answer (LLM-based, need an OpenAI key):
    response_relevancy  - does the answer address the question?
    faithfulness        - are the answer's claims supported by the retrieved context?
    context_precision   - are the retrieved chunks useful for the answer?

Additional metrics computed when a reference (expected) answer is supplied,
for example in batch evaluation with evaluation_dataset.txt:
    bleu_score, rouge_score - overlap between the answer and the expected answer
    non_llm_context_precision - when reference contexts are also supplied
"""

import math
import os
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import (BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy,
                               Faithfulness, RougeScore, LLMContextPrecisionWithoutReference)
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    RAGAS_AVAILABLE = True
    RAGAS_IMPORT_ERROR = ""
except ImportError as import_error:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(import_error)

try:
    from llm_client import resolve_base_url
except ImportError:  # pragma: no cover
    def resolve_base_url(api_key):
        return os.getenv("OPENAI_BASE_URL")

EVALUATOR_MODEL = "gpt-3.5-turbo"
EVALUATOR_EMBEDDING_MODEL = "text-embedding-3-small"


def _validate_inputs(question, answer, contexts) -> Optional[str]:
    """Return an error message for empty or malformed inputs, otherwise None."""
    if not isinstance(question, str) or not question.strip():
        return "Question is empty or not text."
    if not isinstance(answer, str) or not answer.strip():
        return "Answer is empty or not text."
    if answer.startswith("Error generating response"):
        return "The answer is an error message, so it cannot be evaluated."
    if contexts is None or isinstance(contexts, str) or not isinstance(contexts, (list, tuple)):
        return "Contexts must be a list of retrieved text chunks."
    if not [c for c in contexts if isinstance(c, str) and c.strip()]:
        return "No retrieved context was provided, so faithfulness cannot be measured."
    return None


def _clean_score(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else round(value, 4)


def evaluate_response_quality(question: str, answer: str, contexts: List[str],
                              reference: Optional[str] = None,
                              reference_contexts: Optional[List[str]] = None,
                              openai_key: Optional[str] = None,
                              model: str = EVALUATOR_MODEL) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics

    Returns a dictionary such as
        {"response_relevancy": 0.91, "faithfulness": 0.8, "context_precision": 1.0}
    or {"error": "..."} when the inputs are invalid or RAGAS is unavailable.
    """
    if not RAGAS_AVAILABLE:
        return {"error": f"RAGAS not available ({RAGAS_IMPORT_ERROR}). "
                         "Install requirements and run: python fix_ragas_imports.py"}

    problem = _validate_inputs(question, answer, contexts)
    if problem:
        return {"error": problem}

    api_key = openai_key or os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
    if not api_key:
        return {"error": "An OpenAI API key is required for RAGAS evaluation."}
    base_url = resolve_base_url(api_key)

    clean_contexts = [c for c in contexts if isinstance(c, str) and c.strip()]

    # Create evaluator LLM and embeddings
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EVALUATOR_EMBEDDING_MODEL, api_key=api_key, base_url=base_url))

    sample_kwargs = dict(user_input=question.strip(), response=answer.strip(),
                         retrieved_contexts=clean_contexts)
    if reference and reference.strip():
        sample_kwargs["reference"] = reference.strip()
    if reference_contexts:
        sample_kwargs["reference_contexts"] = list(reference_contexts)
    sample = SingleTurnSample(**sample_kwargs)

    # Define an instance for each metric to evaluate
    metrics = {
        "response_relevancy": ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "context_precision": LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    }
    if "reference" in sample_kwargs:
        metrics["bleu_score"] = BleuScore()
        metrics["rouge_score"] = RougeScore()
    if "reference_contexts" in sample_kwargs:
        metrics["non_llm_context_precision"] = NonLLMContextPrecisionWithReference()

    # Evaluate the response with each metric. Each metric is scored on its
    # own so that one failing metric does not hide the others.
    results: Dict[str, float] = {}
    failures: List[str] = []
    for name, metric in metrics.items():
        try:
            score = _clean_score(metric.single_turn_score(sample))
            if score is None:
                failures.append(f"{name}: no score returned")
            else:
                results[name] = score
        except Exception as e:  # network, rate limit or parsing problems
            failures.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")

    if not results:
        return {"error": "All metrics failed. " + "; ".join(failures)}
    if failures:
        results["warnings"] = "; ".join(failures)  # not numeric, so not shown as a score

    # Return the evaluation results
    return results
