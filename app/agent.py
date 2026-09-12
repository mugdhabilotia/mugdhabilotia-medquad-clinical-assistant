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

"""Google ADK 2.0 Agent Application Entrypoint (`app/agent.py`).

Coordinates the decoupled Supervisor-Worker topology by importing modular subagents:
1. `researcher_subagent` from `app.agents.researcher_agent` (Gemini 2.5 Pro)
2. `reviewer_subagent` from `app.agents.reviewer_agent` (Gemini 3.5 Flash)
3. `root_agent` from `app.agents.supervisor_agent` (Gemini 2.5 Flash)
"""

from __future__ import annotations
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App

from app.agents.researcher_agent import (
    RESEARCHER_MODEL,
    researcher_agent,
)
from app.agents.reviewer_agent import (
    REVIEWER_MODEL,
    reviewer_agent,
)
from app.agents.supervisor_agent import (
    SUPERVISOR_MODEL,
    orchestrator_agent,
)

# =====================================================================
# ADK Application Entrypoint
# =====================================================================
app = App(
    root_agent=orchestrator_agent,
    name="medquad-agent",
    context_cache_config=ContextCacheConfig(
        ttl_seconds=3600,     # 1 hour TTL
        cache_intervals=20,   # Keep cache across 20 turns
        min_tokens=2048,
    ),
)
