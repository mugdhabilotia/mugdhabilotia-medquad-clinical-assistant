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

"""Multi-Agent Clinical Assistant Subagents and Orchestrator Module.

Decoupled Supervisor-Worker Topology:
1. `researcher_agent`: Grounded biomedical investigator (Gemini 2.5 Pro) with dual-transport MCP.
2. `reviewer_agent`: Independent clinical quality and safety auditor (Gemini 3.5 Flash).
3. `supervisor_agent`: Root clinical orchestrator (Gemini 2.5 Flash) managing the inquiry lifecycle.
"""

from app.agents.researcher_agent import (
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
from app.agents.reviewer_agent import (
    REVIEWER_INSTRUCTION,
    REVIEWER_MODEL,
    AuditVerdict,
    ReviewOutput,
    reviewer_agent,
)
from app.agents.supervisor_agent import (
    SUPERVISOR_INSTRUCTION,
    SUPERVISOR_MODEL,
    orchestrator_agent,
)

__all__ = [
    "RESEARCHER_INSTRUCTION",
    "RESEARCHER_MODEL",
    "REVIEWER_INSTRUCTION",
    "REVIEWER_MODEL",
    "SUPERVISOR_INSTRUCTION",
    "SUPERVISOR_MODEL",
    "AuditVerdict",
    "ResearchOutput",
    "ReviewOutput",
    "_fetch_id_token",
    "dynamic_auth_httpx_client_factory",
    "get_mcp_headers",
    "get_researcher_mcp_toolset",
    "log_response",
    "orchestrator_agent",
    "researcher_agent",
    "reviewer_agent",
]
