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
import os
from typing import Any

from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, SseConnectionParams

from app.mcp.secrets import get_runtime_api_token
from app.mcp.server import query_mock_clinical_db, search_medquad_corpus

logger = logging.getLogger("medquad.mcp.client")


def create_remote_mcp_toolset(
    sse_endpoint_url: str,
    auth_token: str | None = None,
    timeout_sec: float = 10.0,
    sse_read_timeout_sec: float = 300.0,
) -> McpToolset:
    """Creates an ADK McpToolset connecting over Server-Sent Events (SSE).

    Dynamic discovery of schema elements is handled via standard JSON-RPC capabilities
    exposed by the MCP client.

    Args:
        sse_endpoint_url: Secure SSE endpoint URL of the Cloud Run MCP server.
        auth_token: Optional bearer token for secure network-isolated communication.
        timeout_sec: Connection timeout in seconds.
        sse_read_timeout_sec: Maximum SSE stream idle timeout.

    Returns:
        Configured McpToolset ready for ADK Agent assignment.
    """
    token = auth_token or get_runtime_api_token("MCP_AUTH_TOKEN", default="")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    logger.info(
        f"Initializing ADK McpToolset targeting Cloud Run MCP SSE endpoint: {sse_endpoint_url}"
    )
    connection_params = SseConnectionParams(
        url=sse_endpoint_url,
        headers=headers if headers else None,
        timeout=timeout_sec,
        sse_read_timeout=sse_read_timeout_sec,
    )
    return McpToolset(connection_params=connection_params)


def get_native_adk_clinical_tools() -> list[Any]:
    """Returns natively registered ADK tools enforcing explicit Pydantic type schemas

    and standardized HTTP 429/500 exception handling.
    """
    search_tool = FunctionTool(func=search_medquad_corpus)
    db_tool = FunctionTool(func=query_mock_clinical_db)
    return [search_tool, db_tool]


def get_clinical_tools() -> list[Any]:
    """Resolves tool definitions for the Researcher Subagent.

    If `MCP_SERVER_URL` or `CLOUD_RUN_MCP_URL` is set in environment, utilizes
    dynamic JSON-RPC discovery via ADK McpToolset over SSE; otherwise registers
    native ADK Pydantic-bounded function tools.
    """
    mcp_url = os.environ.get("MCP_SERVER_URL") or os.environ.get("CLOUD_RUN_MCP_URL")
    if mcp_url:
        logger.info(f"Using remote Cloud Run MCP server at {mcp_url}")
        return [create_remote_mcp_toolset(sse_endpoint_url=mcp_url)]

    logger.info(
        "Using native ADK clinical tools with explicit Pydantic type schemas and resilient function calling."
    )
    return get_native_adk_clinical_tools()
