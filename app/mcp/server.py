# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette

from app.mcp.secrets import get_runtime_api_token
from app.tools.clinical_db import query_mock_clinical_db as _base_query_clinical_db
from app.tools.exception_handlers import resilient_function_call
from app.tools.medquad_search import search_medquad_corpus as _base_search_medquad
from app.tools.schemas import ClinicalDBQueryInput, MedQuADSearchInput

logger = logging.getLogger("medquad.mcp.server")

# Initialize isolated FastMCP server
mcp_server = FastMCP(
    name="medquad-clinical-mcp-server",
    instructions=(
        "Isolated Clinical & Biomedical Research Tools MCP Server. "
        "Provides semantic literature retrieval over MedQuAD and simulated EHR database queries. "
        "All parameter schemas are strictly bounded with Pydantic types."
    ),
)


@mcp_server.tool(
    name="search_medquad_corpus",
    description=(
        "Search peer-reviewed MedQuAD biomedical literature, clinical trial evidence, and MeSH terms. "
        "Enforces strict parameter bounds: query length 2-500 chars, top_k 1-10 citations."
    ),
)
@resilient_function_call(max_retries=3, initial_backoff_sec=1.0)
def search_medquad_corpus(
    query: str,
    top_k: int = 3,
    domain_filter: str | None = None,
) -> dict[str, Any]:
    """MCP tool implementation with Pydantic parameter boundary validation and Secret Manager token resolution."""
    # 1. Enforce strict parameter bounds via Pydantic schema
    validated_args = MedQuADSearchInput(
        query=query,
        top_k=top_k,
        domain_filter=domain_filter,
    )

    # 2. Retrieve API credentials at runtime from Secret Manager using service account identity
    api_token = get_runtime_api_token("VERTEX_AI_SEARCH_API_KEY", default="")
    if api_token:
        logger.debug(
            "Successfully loaded Vertex AI Search API token from Secret Manager."
        )

    # 3. Execute underlying search logic
    result = _base_search_medquad(
        query=validated_args.query,
        top_k=validated_args.top_k,
        category=validated_args.domain_filter,
    )
    return result


@mcp_server.tool(
    name="query_mock_clinical_db",
    description=(
        "Query simulated PostgreSQL Electronic Health Records (EHR) for patient clinical profiles or execute parameterized read-only SQL queries. "
        "Accepts patient identifiers (PT-XXXXX or MRN-XXXXXX) or a parameterized read-only SELECT query against PostgreSQL."
    ),
)
@resilient_function_call(max_retries=3, initial_backoff_sec=1.0)
def query_mock_clinical_db(
    patient_id_or_mrn: str | None = None,
    sql_query: str | None = None,
    params: list[Any] | None = None,
    domains: list[str] | None = None,
) -> dict[str, Any]:
    """MCP tool querying patient EHR records or executing parameterized read-only SQL against PostgreSQL."""
    # 1. Enforce strict parameter bounds via Pydantic schema
    input_kwargs: dict[str, Any] = {}
    if patient_id_or_mrn is not None:
        input_kwargs["patient_id_or_mrn"] = patient_id_or_mrn
    if sql_query is not None:
        input_kwargs["sql_query"] = sql_query
    if params is not None:
        input_kwargs["params"] = params
    if domains is not None:
        input_kwargs["domains"] = domains

    validated_args = ClinicalDBQueryInput(**input_kwargs)

    # 2. Retrieve DB credentials at runtime from Secret Manager using service account identity
    db_token = get_runtime_api_token("EHR_DB_PASSWORD", default="")
    if db_token:
        logger.debug("Successfully loaded EHR DB password from Secret Manager.")

    # 3. Execute simulated clinical DB query / SQL
    result = _base_query_clinical_db(
        patient_id_or_mrn=validated_args.patient_id_or_mrn,
        sql_query=validated_args.sql_query,
        params=validated_args.params,
    )
    return result


def create_mcp_sse_app() -> Starlette:
    """Returns the Starlette ASGI application providing standard Server-Sent Events (SSE)

    transport for network-isolated communication on Cloud Run, with container health probes.
    """
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    app = mcp_server.sse_app()

    async def health_check(request: Any) -> JSONResponse:
        return JSONResponse(
            {
                "status": "HEALTHY",
                "service": "medquad-clinical-mcp-server",
                "transport": "SSE / JSON-RPC",
                "tools": ["search_medquad_corpus", "query_mock_clinical_db"],
            }
        )

    app.routes.insert(0, Route("/healthz", health_check, methods=["GET"]))
    app.routes.insert(0, Route("/", health_check, methods=["GET"]))
    return app


if __name__ == "__main__":
    import os
    import sys

    port = int(os.getenv("PORT", "8080"))

    if "--sse" in sys.argv or os.getenv("MCP_TRANSPORT") == "sse":
        import uvicorn

        logger.info(
            f"Starting standalone MedQuAD MCP Server with SSE transport on Cloud Run (Port {port})..."
        )
        uvicorn.run(create_mcp_sse_app(), host="0.0.0.0", port=port)
    else:
        mcp_server.run(transport="stdio")
