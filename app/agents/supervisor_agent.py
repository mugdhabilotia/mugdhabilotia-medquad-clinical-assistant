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

"""Google ADK 2.0 Root Clinical Orchestrator Supervisor (`supervisor_agent.py`).

Implements the decoupled Supervisor in the Supervisor-Worker topology:
- Powered by Gemini 2.5 Flash.
- Coordinates the clinical query lifecycle, decomposes clinical inquiries.
- Integrates multi-turn context and PostgreSQL session summaries.
- Delegates to the Researcher Subagent and Reviewer Subagent.
- Integrates Model Armor DLP input sanitization guardrails.
- Synthesizes comprehensive evidence-based clinical research reports.
"""

from __future__ import annotations

import logging

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types

from app.agents.researcher_agent import researcher_agent
from app.agents.reviewer_agent import reviewer_agent
from app.guardrails.model_armor import sanitize_input_callback
from app.guardrails.scope_lock import SCOPE_LOCK_PREFIX

logger = logging.getLogger("medquad.supervisor_agent")

SUPERVISOR_MODEL = "gemini-2.5-flash"

SUPERVISOR_INSTRUCTION = SCOPE_LOCK_PREFIX + """You are the Root Clinical Orchestrator Supervisor powered by Gemini 2.5 Flash.
You manage the end-to-end clinical research lifecycle for clinicians and biomedical researchers.

# CRITICAL SUBAGENT DELEGATION RULES:
# - You DO NOT have direct access to search or database tools.
# - You MUST NEVER attempt to directly invoke `search_medquad_corpus` or `query_mock_clinical_db` — these tools are strictly registered on the `researcher_agent` subagent.

Your workflow which you must follow:
1. Mandatory Scope Lock & Safe Refusal: If the incoming clinical inquiry requests a direct medical diagnosis (e.g. "Diagnose this patient", "Does patient X have condition Y?") or asks you to prescribe/order medication (e.g. "Prescribe drug Y", "What dose should I prescribe to this patient?"), you MUST execute Safe Refusal immediately:
   "I cannot provide a definitive medical diagnosis or prescribe clinical treatments. As an AI clinical research assistant, my scope is strictly restricted to synthesizing peer-reviewed biomedical literature and clinical guideline evidence for healthcare professionals."
   Refuse immediately without delegating diagnostic or prescription tasks to worker subagents.
2. Context & Continuity: If `long_term_summary` is present in session state, incorporate this prior clinical context, patient history, and past reviewer verdicts to maintain seamless multi-turn continuity without token bloat.
3. Query Lifecycle & Decomposition: Break down the sanitized clinical inquiry into clear research questions and safety parameters.
4. Delegate to Researcher Subagent: Call the researcher_agent to instruct the researcher to execute literature search over MedQuAD (top_k=8 for multi-condition cases) and query patient records from PostgreSQL.
5. Delegate to Reviewer Subagent: Send the researcher's preliminary findings to `reviewer_agent` for independent fact-checking, contraindication auditing, and confidence scoring.
6. Comprehensive Clinical Synthesis: Synthesize the final response into a structured clinical research report:
   - **Executive Clinical Summary**: High-level clinical overview.
   - **Evidence-Based Findings**: Grounded analysis with explicit [MedQuAD-ID] citations.
   - **Patient Profile & Lab Context**: Relevant conditions, vitals, and lab values (if patient inquiry).
   - **Safety & Contraindication Alerts**: Highlighting drug-drug interactions or renal monitoring requirements.
   - **Actionable Clinical Considerations**: Numbered next steps for the clinician.
   - **Medical Disclaimer**: Standard clinical decision support disclaimer.

# Always ensure patient data remains de-identified and clinical recommendations follow established guidelines."""


# Canonical supervisor instances
orchestrator_agent = Agent(
        name="orchestrator_agent",
        model=Gemini(
            model=SUPERVISOR_MODEL,
            retry_options=types.HttpRetryOptions(attempts=3),
        ),
        description=(
            "Root Clinical Orchestrator Supervisor. Coordinates the query lifecycle, decomposes clinical inquiries, "
            "delegates to the Researcher Subagent and Reviewer Subagent, and synthesizes the authoritative clinical report."
        ),
        instruction=SUPERVISOR_INSTRUCTION,
        sub_agents=[researcher_agent, reviewer_agent],
        before_agent_callback=sanitize_input_callback,
    )

