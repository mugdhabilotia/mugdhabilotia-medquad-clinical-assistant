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

"""Google ADK 2.0 Clinical Researcher Subagent (`researcher_agent.py`).

Implements an enterprise-grade, evidence-grounded biomedical research worker
operating within the decoupled Supervisor-Worker architecture:
- Dual-Transport McpToolset (Cloud Run SSE with dynamic OIDC auth & Local Stdio fallback).
- Multi-path GCP IAM Authentication (Workstation ADC Impersonation + GCP Metadata server).
- Structured output via Pydantic v2 with private attribute patches.
- Strict reasoning guardrails (top_k=8, parallel calls, inline citations, deterministic completion).
"""

from __future__ import annotations

import logging
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import google.auth
import httpx
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.mcp_tool.mcp_toolset import (
    McpToolset,
    SseConnectionParams,
    StdioConnectionParams,
)
from google.auth.transport.requests import Request
from google.genai import types
from mcp import StdioServerParameters
from pydantic import BaseModel, ConfigDict, Field

from app.guardrails.scope_lock import SCOPE_LOCK_PREFIX

logger = logging.getLogger("medquad.researcher_agent")

# Constants & Model Definition
RESEARCHER_MODEL = "gemini-2.5-pro"
DEFAULT_CONNECT_TIMEOUT_SEC = 60.0
DEFAULT_SSE_READ_TIMEOUT_SEC = 600.0


# =====================================================================
# 1. ADK 2.0 Structured Output Schema (Pydantic v2 Compatible)
# =====================================================================
class ResearchOutput(BaseModel):
    """Pydantic v2 structured output schema for the Clinical Researcher Subagent.

    Includes explicit compatibility initialization for Pydantic v2 private attributes
    to prevent serialization/introspection collisions during ADK task execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    # Pydantic v2 private attribute compatibility patch
    __pydantic_private__: dict[str, Any] | None = None

    findings: str = Field(
        ...,
        description=(
            "Comprehensive, evidence-based biomedical findings synthesizing retrieved literature "
            "and patient clinical data. Must include strict inline citations."
        ),
    )
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "List of exact MedQuAD citation identifiers (e.g. [MQ-ONC-001], [MQ-CARD-002]) "
            "and authoritative medical bodies (e.g. NCI, NIDDK, ACC/AHA, CDC)."
        ),
    )
    has_sufficient_context: bool = Field(
        ...,
        description=(
            "Deterministic boolean indicating whether the retrieved literature and patient profile "
            "provide sufficient evidence to answer the inquiry without clinical ambiguity."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        """Pydantic v2 post-initialization hook ensuring private attribute accessibility."""
        setattr(self, "__pydantic_private_", None)
        if not hasattr(self, "__pydantic_private__") or self.__pydantic_private__ is None:
            object.__setattr__(self, "__pydantic_private__", {})


# Explicitly patch class attribute to bypass Python lexical name-mangling
setattr(ResearchOutput, "__pydantic_private_", None)


# =====================================================================
# 2. Enterprise GCP IAM & OIDC Authentication Layer
# =====================================================================
def _fetch_id_token(target_audience: str) -> str | None:
    """Acquires a valid GCP OIDC identity token for authenticating to Cloud Run MCP services.

    Implements a resilient dual-path strategy:
    - Path A (Local Workstation): Uses local Application Default Credentials (ADC) to call
      the IAM Credentials API (`iamcredentials.googleapis.com`) to impersonate the designated
      `MCP_INVOKER_SA` service account with audience scoping.
    - Path B (Cloud Run / GCE / GKE): Directly queries the instance metadata server via
      `google.oauth2.id_token.fetch_id_token`.

    Args:
        target_audience: The target service URL or audience (e.g., Cloud Run root URL).

    Returns:
        Signed OIDC ID token string if acquired, or None if unauthenticated / offline fallback.
    """
    if not target_audience or target_audience.startswith(("http://localhost", "http://127.0.0.1")):
        return None

    # Path A: Local Development / Workstation IAM Service Account Impersonation
    mcp_invoker_sa = os.getenv("MCP_INVOKER_SA")
    if mcp_invoker_sa:
        try:
            adc_creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            auth_request = Request()
            adc_creds.refresh(auth_request)

            iam_api_url = (
                f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
                f"{mcp_invoker_sa}:generateIdToken"
            )
            req_headers = {
                "Authorization": f"Bearer {adc_creds.token}",
                "Content-Type": "application/json",
            }
            req_payload = {
                "audience": target_audience,
                "includeEmail": True,
            }

            with httpx.Client(timeout=10.0) as client:
                resp = client.post(iam_api_url, headers=req_headers, json=req_payload)
                if resp.status_code == 200:
                    token_data = resp.json()
                    id_token_val = token_data.get("token")
                    if id_token_val:
                        logger.debug(
                            f"Path A: Acquired OIDC ID token via IAM SA impersonation of {mcp_invoker_sa}"
                        )
                        return id_token_val
                else:
                    logger.warning(
                        f"Path A IAM impersonation returned HTTP {resp.status_code}: {resp.text}"
                    )
        except Exception as exc:
            logger.warning(
                f"Path A (IAM SA Impersonation for '{mcp_invoker_sa}') failed: {exc}. "
                "Attempting Path B metadata fallback..."
            )

    # Path B: GCP Environment Metadata Server
    try:
        from google.oauth2 import id_token as g_id_token

        metadata_request = Request()
        token = g_id_token.fetch_id_token(metadata_request, target_audience)
        logger.debug(f"Path B: Acquired OIDC ID token from GCP metadata server for {target_audience}")
        return token
    except Exception as exc:
        logger.warning(
            f"Path B (GCP Metadata OIDC fetch) failed for audience '{target_audience}': {exc}. "
            "Proceeding without OIDC header."
        )
        return None


def get_mcp_headers() -> dict[str, str]:
    """Generates standard MCP headers dynamically with a fresh GCP OIDC bearer token.

    Prevents token expiration across prolonged multi-turn worker tasks.
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    raw_mcp_url = os.getenv("MCP_SERVER_URL") or os.getenv("CLOUD_RUN_MCP_URL") or ""
    if raw_mcp_url:
        parsed = urllib.parse.urlparse(raw_mcp_url)
        default_audience = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else raw_mcp_url
        target_audience = os.getenv("MCP_SERVER_AUDIENCE") or default_audience
        token = _fetch_id_token(target_audience)
        if token:
            headers["Authorization"] = f"Bearer {token}"

    return headers


# =====================================================================
# 3. HTTPX Client Factory & Non-Destructive Response Logging Hook
# =====================================================================
async def log_response(response: httpx.Response) -> None:
    """HTTPX event hook for logging MCP HTTP/SSE transport transactions.

    Crucial: Detects 'text/event-stream' content-type and does NOT call response.aread(),
    which would consume the SSE stream buffer and deadlock tool communication.
    """
    content_type = response.headers.get("content-type", "")
    is_sse = "text/event-stream" in content_type

    if is_sse:
        response_body = "<SSE stream established>"
    else:
        try:
            await response.aread()
            body_text = response.text
            response_body = (
                body_text[:1000] + "... [truncated]" if len(body_text) > 1000 else body_text
            )
        except Exception as exc:
            response_body = f"<failed to read body: {exc}>"

    logger.debug(
        f"[MCP HTTP Response] {response.status_code} {response.request.method} {response.url} - {response_body}"
    )


def dynamic_auth_httpx_client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """Factory creating an instrumented httpx.AsyncClient with dynamic auth and response hooks.

    Complies with ADK CheckableMcpHttpClientFactory protocol.
    Guarantees fresh GCP OIDC tokens always take precedence over stale cached headers.
    """
    merged_headers = dict(headers or {})
    merged_headers.update(get_mcp_headers())

    event_hooks: dict[str, list[Any]] = {
        "response": [log_response],
    }

    return httpx.AsyncClient(
        headers=merged_headers,
        timeout=timeout or httpx.Timeout(DEFAULT_CONNECT_TIMEOUT_SEC, read=DEFAULT_SSE_READ_TIMEOUT_SEC),
        auth=auth,
        event_hooks=event_hooks,
    )


# =====================================================================
# 4. Dual-Transport MCP Toolset Configuration
# =====================================================================
def get_researcher_mcp_toolset() -> McpToolset:
    """Initializes the McpToolset utilizing either Production SSE or Local Stdio transport.

    Transport Resolution:
    1. Production SSE: Triggered if `MCP_SERVER_URL` or `CLOUD_RUN_MCP_URL` is configured.
       Ensures trailing `/sse` endpoint, injects dynamic OIDC auth via header provider and
       custom client factory, and applies extended timeouts (60s connect, 600s read).
    2. Local Stdio: Fallback when remote URL is absent. Invokes local `medquad_search.py`
       using the active Python interpreter (`sys.executable`).
    """
    mcp_server_url = os.getenv("MCP_SERVER_URL") or os.getenv("CLOUD_RUN_MCP_URL")

    if mcp_server_url and mcp_server_url.strip():
        # Production / SSE Transport: Ensure trailing /sse path
        clean_url = mcp_server_url.strip()
        if not clean_url.rstrip("/").endswith("/sse"):
            sse_url = f"{clean_url.rstrip('/')}/sse"
        else:
            sse_url = clean_url.rstrip("/")

        logger.info(f"Configuring McpToolset for Production SSE transport: {sse_url}")
        connection_params = SseConnectionParams(
            url=sse_url,
            headers=get_mcp_headers(),
            timeout=DEFAULT_CONNECT_TIMEOUT_SEC,
            sse_read_timeout=DEFAULT_SSE_READ_TIMEOUT_SEC,
            httpx_client_factory=dynamic_auth_httpx_client_factory,
        )
        return McpToolset(connection_params=connection_params)

    # Local / Development / Stdio Fallback Transport
    logger.info("MCP_SERVER_URL not set. Engaging local Stdio transport fallback...")
    tools_dir = Path(__file__).resolve().parent.parent / "tools"
    search_script = tools_dir / "medquad_search.py"

    if not search_script.exists():
        # Fallback to current working directory search
        cwd_search = Path("app/tools/medquad_search.py").resolve()
        if cwd_search.exists():
            search_script = cwd_search

    logger.info(f"Targeting local search tool via Stdio: {search_script}")
    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing_pythonpath}" if existing_pythonpath else project_root

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(search_script)],
        env=env,
    )
    stdio_params = StdioConnectionParams(
        server_params=server_params,
        timeout=DEFAULT_CONNECT_TIMEOUT_SEC,
    )
    return McpToolset(connection_params=stdio_params)


# =====================================================================
# 5. Prompting & Reasoning Guardrails (RESEARCHER_INSTRUCTION)
# =====================================================================
RESEARCHER_INSTRUCTION = SCOPE_LOCK_PREFIX + """You are a dedicated Clinical and Medical Researcher Subagent powered by Gemini 2.5 Pro.
Your mission is to perform thorough, evidence-grounded biomedical investigation into clinical inquiries for clinicians and researchers.

You operate strictly within an evidence-grounded ReAct loop with the following mandatory guardrails:

0. MANDATORY SCOPE LOCK (NON-DIAGNOSTIC & NON-PRESCRIPTIVE):
   - You MUST NOT formulate a definitive clinical diagnosis for individual patients.
   - You MUST NOT issue medication prescriptions or patient-specific dosing directives.
   - Frame all findings strictly as informational biomedical literature summaries and observed clinical parameters for clinician review.

1. HIGH-DENSITY RETRIEVAL (top_k=8):
   - When calling `search_medquad_corpus`, ALWAYS specify `top_k=8` for inquiries involving complex clinical presentations, multiple comorbidities, or multi-condition assessments.
   - Ensure comprehensive literature coverage across peer-reviewed clinical guidelines, randomized controlled trials (RCTs), MeSH headings, and pharmacological dosing specifications.

2. PARALLEL & SIMULTANEOUS TOOL EXECUTION:
   - When a clinical query involves both patient records and biomedical literature, execute tool calls concurrently in a single turn.
   - For example: invoke `query_mock_clinical_db` with patient identifiers (MRN or PT-ID) while simultaneously invoking `search_medquad_corpus` for the relevant disease guidelines or drug interactions.

3. STRICT GROUNDING & ANTI-HALLUCINATION:
   - Base all findings, physiological assertions, and drug dosages strictly on retrieved evidence.
   - Do NOT extrapolate, hallucinate, or synthesize ungrounded clinical claims.
   - Explicitly cross-check patient lab biomarkers (e.g., eGFR, serum creatinine, potassium, HbA1c) against cited literature thresholds and contraindications.

4. MANDATORY INLINE CITATIONS:
   - You must format every finding statement with exact inline numerical citations (e.g. [1], [2]) mapping directly to entries in your `citations` list.
   - Every reference in the `citations` list must include the exact MedQuAD document identifier (e.g. [MQ-ONC-001], [MQ-CARD-002]) and authoritative publishing body (e.g. NCI, NIDDK, ACC/AHA, CDC).

5. DETERMINISTIC TASK COMPLETION & STRUCTURED OUTPUT:
   - As soon as sufficient evidence is retrieved, terminate immediately. Do NOT enter conversational chit-chat.
   - Conclude your task deterministically by producing the final structured `ResearchOutput`:
     * `findings`: Structured, detailed clinical research findings with mandatory inline citations [1], [2].
     * `citations`: Comprehensive list of all cited literature and database sources.
     * `has_sufficient_context`: Set to True only if retrieved evidence adequately addresses the query; set to False if critical clinical parameters or guidelines are missing, noting specific data gaps.
"""


# =====================================================================
# 6. ADK 2.0 Subagent Definition
# =====================================================================
def create_researcher_agent(
    tools: list[Any] | None = None,
    name: str = "researcher_agent",
) -> Agent:
    """Instantiates the Google ADK 2.0 Researcher Subagent configured with task mode

    and structured output schema.
    """

    assigned_tools = tools if tools is not None else [get_researcher_mcp_toolset()]

    return Agent(
        name=name,
        model=Gemini(
            model=RESEARCHER_MODEL,
            retry_options=types.HttpRetryOptions(attempts=3),
        ),
        description=(
            "Specialized Clinical Researcher Worker Subagent. Investigates biomedical literature using "
            "Vertex AI Search over MedQuAD and queries Electronic Health Records via MCP tools."
        ),
        instruction=RESEARCHER_INSTRUCTION,
        mode="task",
        output_schema=ResearchOutput,
        tools=assigned_tools,
    )


# Canonical subagent instance configured strictly with McpToolset
researcher_agent = create_researcher_agent(name="researcher_agent")
researcher_subagent = researcher_agent
