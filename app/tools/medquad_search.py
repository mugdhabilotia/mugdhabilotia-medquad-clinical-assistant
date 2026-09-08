import logging
import os
import random
import time
from typing import Any

from dotenv import load_dotenv

from app.tools.static_index import static_medical_index

load_dotenv()

logger = logging.getLogger("medquad_search")


def search_medquad_corpus(
    query: str, top_k: int = 3, category: str | None = None
) -> dict[str, Any]:
    """Performs semantic retrieval over the MedQuAD clinical corpus using Vertex AI Search.

    Implements exponential backoff with randomized jitter, and automatically falls back
    to a static medical knowledge index if the primary search API fails or is unavailable.

    Args:
        query: The clinical inquiry or medical search terms.
        top_k: The maximum number of relevant citations to return (default 3).
        category: Optional medical domain filter (e.g. 'oncology', 'cardiology', 'endocrinology').

    Returns:
        A dictionary containing the search results, source used (live vs fallback),
        relevance scores, evidence levels, and retry telemetry.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID")
    datastore_id = os.getenv("VERTEX_SEARCH_DATASTORE_ID")
    location = os.getenv("GCP_LOCATION", "global")

    max_retries = 3
    base_delay = 0.5
    backoff_attempts = 0
    last_error = None

    # Attempt live Vertex AI Search only if both project and datastore are explicitly configured
    if project_id and project_id != "YOUR_PROJECT_ID" and datastore_id:
        for attempt in range(max_retries):
            backoff_attempts = attempt + 1
            try:
                from google.cloud import discoveryengine_v1 as discoveryengine

                client = discoveryengine.SearchServiceClient()
                serving_config = client.serving_config_path(
                    project=project_id,
                    location=location,
                    data_store=datastore_id,
                    serving_config="default_config",
                )
                request = discoveryengine.SearchRequest(
                    serving_config=serving_config,
                    query=query,
                    page_size=top_k,
                )
                response = client.search(request)

                results = []
                for res in response.results:
                    doc = res.document
                    doc_data = doc.struct_data if doc.struct_data else doc.derived_struct_data
                    answer = doc_data.get("answer", "") if doc_data else ""
                    focus = doc_data.get("focus", query) if doc_data else query
                    question = doc_data.get("question", query) if doc_data else query
                    source = doc_data.get("source", "Vertex AI Search / MedQuAD") if doc_data else "Vertex AI Search / MedQuAD"
                    evidence_level = doc_data.get("evidence_level", "Clinical Evidence") if doc_data else "Clinical Evidence"

                    url = doc_data.get("url", "") if doc_data else ""
                    if not url:
                        url = f"https://medlineplus.gov/search?query={query.replace(' ', '+')}"

                    results.append(
                        {
                            "id": doc.id,
                            "focus": focus,
                            "question": question,
                            "answer_snippet": (answer[:280] + "...") if len(answer) > 280 else answer,
                            "full_answer": answer,
                            "source": source,
                            "url": url,
                            "evidence_level": evidence_level,
                            "relevance_score": 0.95,
                        }
                    )

                if results:
                    return {
                        "status": "SUCCESS",
                        "source": "Vertex AI Search (Live MedQuAD Index)",
                        "query": query,
                        "backoff_attempts": backoff_attempts,
                        "results_count": len(results),
                        "citations": results,
                    }
            except Exception as exc:
                last_error = str(exc)
                err_str = str(exc)
                # Fast-fail on non-retryable errors (disabled API, missing datastore, permission denied)
                if any(k in err_str for k in ("SERVICE_DISABLED", "NOT_FOUND", "PERMISSION_DENIED", "403", "404")):
                    logger.warning(
                        f"Vertex AI Search non-retryable error: {exc}. Fast-failing to static index without latency."
                    )
                    break

                jitter = random.uniform(0.1, 0.4)
                sleep_sec = min((base_delay * (2**attempt)) + jitter, 3.0)
                logger.warning(
                    f"Vertex AI Search attempt {attempt + 1} failed: {exc}. "
                    f"Backing off with jitter for {sleep_sec:.2f}s before retry..."
                )
                time.sleep(sleep_sec)

    # Graceful fallback to static medical index
    logger.info(
        f"Engaging resilient static medical index fallback for query: '{query}' "
        f"(Reason: {last_error or 'Vertex AI Search not directly reachable'})."
    )
    fallback_results = static_medical_index.search(
        query=query, top_k=top_k, category_filter=category
    )

    return {
        "status": "FALLBACK_SUCCESS",
        "source": "Static MedQuAD Medical Index (Resilient Fallback)",
        "query": query,
        "backoff_attempts": backoff_attempts,
        "fallback_reason": last_error or "Vertex AI Search API fallback engaged",
        "results_count": len(fallback_results),
        "citations": fallback_results,
    }


if __name__ == "__main__":
    from app.mcp.server import mcp_server

    mcp_server.run(transport="stdio")

