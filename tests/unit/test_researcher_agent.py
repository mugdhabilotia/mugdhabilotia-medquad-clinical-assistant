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

"""Unit test suite verifying the Google ADK 2.0 Clinical Researcher Subagent (`researcher_agent.py`).

Covers all 5 verification checklist categories:
1. Code completeness & syntax integrity.
2. Dual-transport MCP Toolset configuration (SSE + Stdio fallback).
3. Enterprise GCP IAM & OIDC authentication (_fetch_id_token dual-path).
4. ADK 2.0 agent definition & structured output schema (Pydantic v2 patch).
5. Prompting & reasoning guardrails.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from google.adk.tools.mcp_tool.mcp_toolset import (
    McpToolset,
    SseConnectionParams,
    StdioConnectionParams,
)

from app.agents.researcher_agent import (
    DEFAULT_CONNECT_TIMEOUT_SEC,
    DEFAULT_SSE_READ_TIMEOUT_SEC,
    RESEARCHER_INSTRUCTION,
    RESEARCHER_MODEL,
    ResearchOutput,
    _fetch_id_token,
    dynamic_auth_httpx_client_factory,
    get_mcp_headers,
    get_researcher_mcp_toolset,
    log_response,
    researcher_agent,
)


# =====================================================================
# 1. Code Completeness & Syntax Integrity Tests
# =====================================================================
def test_get_mcp_headers_structure_and_scoping():
    """Verify get_mcp_headers returns complete headers with dynamic OIDC bearer token."""
    with patch("app.agents.researcher_agent._fetch_id_token", return_value="mock-oidc-jwt-token"):
        with patch.dict(
            "os.environ",
            {
                "MCP_SERVER_URL": "https://mcp-clinical-server-xyz.a.run.app",
                "MCP_SERVER_AUDIENCE": "https://mcp-clinical-server-xyz.a.run.app",
            },
        ):
            headers = get_mcp_headers()
            assert headers["Content-Type"] == "application/json"
            assert headers["Accept"] == "text/event-stream"
            assert headers["Authorization"] == "Bearer mock-oidc-jwt-token"


@pytest.mark.asyncio
async def test_log_response_sse_non_destructive():
    """Verify log_response safely handles SSE event streams without calling aread()."""
    mock_request = httpx.Request("GET", "https://mcp-server.a.run.app/sse")
    mock_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        request=mock_request,
    )
    # Mock aread to ensure it is NEVER awaited on an active SSE stream
    mock_response.aread = AsyncMock()

    await log_response(mock_response)
    mock_response.aread.assert_not_called()


@pytest.mark.asyncio
async def test_log_response_json_safe_read():
    """Verify log_response reads and safely logs non-SSE responses."""
    mock_request = httpx.Request("POST", "https://mcp-server.a.run.app/tools/call")
    mock_response = httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=b'{"status": "OK", "result": "sample"}',
        request=mock_request,
    )

    await log_response(mock_response)
    assert mock_response.is_success


def test_dynamic_auth_httpx_client_factory_binding():
    """Verify client factory constructs AsyncClient with logging hooks and merged headers."""
    with patch("app.agents.researcher_agent.get_mcp_headers", return_value={"X-Custom": "val"}):
        client = dynamic_auth_httpx_client_factory(headers={"X-Extra": "extra-val"})
        assert isinstance(client, httpx.AsyncClient)
        assert client.headers.get("X-Custom") == "val"
        assert client.headers.get("X-Extra") == "extra-val"
        assert "response" in client.event_hooks
        assert log_response in client.event_hooks["response"]


# =====================================================================
# 2. Dual-Transport MCP Toolset Configuration Tests
# =====================================================================
def test_mcp_toolset_production_sse_transport():
    """Verify SSE transport normalizes trailing /sse, configures timeouts and client factory."""
    with patch.dict(
        "os.environ",
        {"MCP_SERVER_URL": "https://mcp-clinical-service-test.a.run.app"},
        clear=False,
    ):
        toolset = get_researcher_mcp_toolset()
        assert isinstance(toolset, McpToolset)
        conn_params = toolset._connection_params
        assert isinstance(conn_params, SseConnectionParams)
        # Verify trailing /sse path normalization
        assert conn_params.url == "https://mcp-clinical-service-test.a.run.app/sse"
        # Verify extended clinical streaming timeouts
        assert conn_params.timeout == DEFAULT_CONNECT_TIMEOUT_SEC  # 60.0
        assert conn_params.sse_read_timeout == DEFAULT_SSE_READ_TIMEOUT_SEC  # 600.0
        # Verify custom HTTPX client factory
        assert conn_params.httpx_client_factory == dynamic_auth_httpx_client_factory


def test_mcp_toolset_local_stdio_fallback():
    """Verify local Stdio fallback targets medquad_search.py via sys.executable when remote URL absent."""
    with patch.dict("os.environ", {"MCP_SERVER_URL": "", "CLOUD_RUN_MCP_URL": ""}, clear=True):
        toolset = get_researcher_mcp_toolset()
        assert isinstance(toolset, McpToolset)
        conn_params = toolset._connection_params
        assert isinstance(conn_params, StdioConnectionParams)
        server_params = conn_params.server_params
        import sys

        assert server_params.command == sys.executable
        assert any("medquad_search.py" in arg for arg in server_params.args)
        assert conn_params.timeout == 60.0


# =====================================================================
# 3. Enterprise GCP IAM Authentication Layer Tests
# =====================================================================
def test_fetch_id_token_path_a_iam_sa_impersonation():
    """Verify Path A invokes IAM Credentials API with audience scoping for local impersonation."""
    mock_adc = MagicMock()
    mock_adc.token = "mock-adc-access-token"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"token": "signed-impersonated-oidc-jwt"}

    target_aud = "https://mcp-clinical-service-test.a.run.app"
    sa_email = "mcp-invoker@test-project.iam.gserviceaccount.com"

    with patch.dict("os.environ", {"MCP_INVOKER_SA": sa_email}):
        with patch("google.auth.default", return_value=(mock_adc, "test-project")):
            with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
                token = _fetch_id_token(target_aud)
                assert token == "signed-impersonated-oidc-jwt"
                mock_post.assert_called_once()
                call_args, call_kwargs = mock_post.call_args
                assert sa_email in call_args[0]
                assert call_kwargs["json"] == {"audience": target_aud, "includeEmail": True}
                assert call_kwargs["headers"]["Authorization"] == "Bearer mock-adc-access-token"


def test_fetch_id_token_path_b_metadata_fallback():
    """Verify Path B retrieves OIDC token from GCP instance metadata server when SA not set."""
    target_aud = "https://mcp-clinical-service-test.a.run.app"

    with patch.dict("os.environ", {"MCP_INVOKER_SA": ""}, clear=True):
        with patch("google.oauth2.id_token.fetch_id_token", return_value="metadata-oidc-jwt") as mock_fetch:
            token = _fetch_id_token(target_aud)
            assert token == "metadata-oidc-jwt"
            mock_fetch.assert_called_once()


def test_fetch_id_token_localhost_graceful_bypass():
    """Verify localhost targets bypass OIDC token acquisition cleanly."""
    assert _fetch_id_token("http://localhost:8080") is None
    assert _fetch_id_token("http://127.0.0.1:8000") is None
    assert _fetch_id_token("") is None


# =====================================================================
# 4. ADK 2.0 Agent Definition & Structured Output Tests
# =====================================================================
def test_research_output_pydantic_v2_schema_and_private_patch():
    """Verify ResearchOutput schema validation and Pydantic v2 private attribute compatibility."""
    output = ResearchOutput(
        findings="Frontline osimertinib demonstrated prolonged progression-free survival [1].",
        citations=["[MQ-ONC-001] NCI Clinical Guidelines"],
        has_sufficient_context=True,
    )
    assert output.findings.startswith("Frontline osimertinib")
    assert len(output.citations) == 1
    assert output.has_sufficient_context is True

    # Validate Pydantic v2 private attribute patch
    assert hasattr(output, "__pydantic_private_")
    assert output.__pydantic_private_ is None

    # Serialization roundtrip
    payload = output.model_dump()
    assert payload["has_sufficient_context"] is True
    reconstructed = ResearchOutput.model_validate(payload)
    assert reconstructed.findings == output.findings


def test_researcher_agent_definition():
    """Verify researcher_agent ADK 2.0 configuration with mode='task' and output_schema."""
    assert researcher_agent.name == "researcher_agent"
    assert researcher_agent.mode == "task"
    assert researcher_agent.output_schema == ResearchOutput
    assert researcher_agent.model.model == RESEARCHER_MODEL
    assert RESEARCHER_MODEL == "gemini-2.5-pro"

    # ADK 2.0 task mode binds the configured McpToolset and automatically adds FinishTaskTool
    tool_types = [t.__class__.__name__ for t in researcher_agent.tools]
    assert "McpToolset" in tool_types
    assert "FinishTaskTool" in tool_types
    assert any(isinstance(t, McpToolset) for t in researcher_agent.tools)


# =====================================================================
# 5. Prompting & Reasoning Guardrails Tests
# =====================================================================
def test_researcher_instruction_guardrails():
    """Verify RESEARCHER_INSTRUCTION strictly enforces all 5 required reasoning rules."""
    instruction = RESEARCHER_INSTRUCTION

    # Rule 1: High-density retrieval (top_k=8)
    assert "top_k=8" in instruction
    assert "HIGH-DENSITY RETRIEVAL" in instruction

    # Rule 2: Parallel execution
    assert "PARALLEL & SIMULTANEOUS TOOL EXECUTION" in instruction
    assert "concurrently in a single turn" in instruction

    # Rule 3: Anti-hallucination & Grounding
    assert "STRICT GROUNDING & ANTI-HALLUCINATION" in instruction
    assert "eGFR" in instruction
    assert "creatinine" in instruction

    # Rule 4: Mandatory inline citations
    assert "MANDATORY INLINE CITATIONS" in instruction
    assert "[1]" in instruction
    assert "[2]" in instruction
    assert "MedQuAD" in instruction

    # Rule 5: Deterministic completion & structured output
    assert "DETERMINISTIC TASK COMPLETION" in instruction
    assert "ResearchOutput" in instruction
    assert "has_sufficient_context" in instruction
