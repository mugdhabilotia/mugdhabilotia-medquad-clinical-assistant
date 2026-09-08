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

"""Unit test suite verifying the Hardcoded Scope Lock and Safe Refusal guardrails.

Covers:
1. SCOPE_LOCK_PREFIX presence and content across Supervisor, Researcher, and Reviewer agents.
2. Inbound query scope lock detection (Diagnostic vs Prescription vs Informational).
3. Output violation auditing.
4. Model Armor callback integration with scope lock metadata.
"""

from unittest.mock import MagicMock

import pytest

from app.agents.researcher_agent import RESEARCHER_INSTRUCTION, researcher_agent
from app.agents.reviewer_agent import REVIEWER_INSTRUCTION, reviewer_agent
from app.agents.supervisor_agent import SUPERVISOR_INSTRUCTION, root_agent
from app.guardrails.model_armor import sanitize_input_callback
from app.guardrails.scope_lock import (
    SAFE_REFUSAL_RESPONSE,
    SCOPE_LOCK_PREFIX,
    detect_scope_lock_violation,
    is_diagnostic_or_prescriptive_query,
)


def test_scope_lock_prefix_content():
    """Verify SCOPE_LOCK_PREFIX contains mandatory non-medical advice refusal directives."""
    assert "CRITICAL MANDATORY SCOPE LOCK" in SCOPE_LOCK_PREFIX
    assert "ABSOLUTE DIAGNOSTIC REFUSAL" in SCOPE_LOCK_PREFIX
    assert "ABSOLUTE PRESCRIPTION & TREATMENT REFUSAL" in SCOPE_LOCK_PREFIX
    assert "SAFE REFUSAL STANDARD RESPONSE" in SCOPE_LOCK_PREFIX
    assert SAFE_REFUSAL_RESPONSE in SCOPE_LOCK_PREFIX


def test_scope_lock_hardcoded_in_all_agents():
    """Verify that all three agents strictly contain the hardcoded SCOPE_LOCK_PREFIX."""
    # 1. Supervisor Orchestrator
    assert SCOPE_LOCK_PREFIX in SUPERVISOR_INSTRUCTION
    assert SCOPE_LOCK_PREFIX in root_agent.instruction
    assert "Mandatory Scope Lock & Safe Refusal" in root_agent.instruction

    # 2. Researcher Subagent
    assert SCOPE_LOCK_PREFIX in RESEARCHER_INSTRUCTION
    assert SCOPE_LOCK_PREFIX in researcher_agent.instruction
    assert "NON-DIAGNOSTIC & NON-PRESCRIPTIVE" in researcher_agent.instruction

    # 3. Reviewer Subagent
    assert SCOPE_LOCK_PREFIX in REVIEWER_INSTRUCTION
    assert SCOPE_LOCK_PREFIX in reviewer_agent.instruction
    assert "MANDATORY SCOPE LOCK AUDIT" in reviewer_agent.instruction


@pytest.mark.parametrize(
    "query, expected_category",
    [
        ("Can you diagnose this patient with acute coronary syndrome?", "DIAGNOSTIC_QUERY"),
        ("What illness do I have based on these symptoms?", "DIAGNOSTIC_QUERY"),
        ("Please give me a definitive diagnosis for MRN-123456", "DIAGNOSTIC_QUERY"),
        ("Does patient Marcus have non-small cell lung cancer?", "DIAGNOSTIC_QUERY"),
        ("Prescribe 80mg Osimertinib daily for this patient", "PRESCRIPTION_QUERY"),
        ("Write a prescription for metformin 500mg BID", "PRESCRIPTION_QUERY"),
        ("What dose should I prescribe to this patient?", "PRESCRIPTION_QUERY"),
    ],
)
def test_diagnostic_and_prescription_query_detection(query, expected_category):
    """Verify detection of queries requiring Safe Refusal under Scope Lock."""
    is_triggered, category = is_diagnostic_or_prescriptive_query(query)
    assert is_triggered is True
    assert category == expected_category


@pytest.mark.parametrize(
    "query",
    [
        "What does MedQuAD say about first-line therapy for EGFR-mutated NSCLC?",
        "Summarize the clinical trial outcomes from the FLAURA trial",
        "What are the established renal thresholds for SGLT2 inhibitors?",
        "Compare the pharmacokinetics of apixaban and rivaroxaban",
    ],
)
def test_legitimate_research_queries_permitted(query):
    """Verify legitimate biomedical research inquiries do NOT trigger Safe Refusal."""
    is_triggered, category = is_diagnostic_or_prescriptive_query(query)
    assert is_triggered is False
    assert category is None


def test_output_violation_audit():
    """Verify detect_scope_lock_violation flags improper diagnostic or prescriptive output."""
    bad_output = (
        "Based on the lab values, I diagnose the patient with Type 2 Diabetes Mellitus. "
        "I hereby prescribe Metformin 500mg once daily."
    )
    violations = detect_scope_lock_violation(bad_output)
    assert len(violations) >= 2
    assert any("diagnostic" in v.lower() for v in violations)
    assert any("prescribe" in v.lower() or "prescription" in v.lower() for v in violations)

    clean_output = (
        "According to NIDDK clinical guidelines [1], an HbA1c >= 6.5% is indicative of diabetes. "
        "Standard guidelines recommend lifestyle interventions and consideration of metformin [2]."
    )
    assert len(detect_scope_lock_violation(clean_output)) == 0


@pytest.mark.asyncio
async def test_model_armor_callback_flags_scope_lock_trigger():
    """Verify sanitize_input_callback annotates ADK session state with scope lock metadata."""
    mock_context = MagicMock()
    mock_part = MagicMock()
    mock_part.text = "Please diagnose patient Marcus Holloway with stage 4 adenocarcinoma"
    mock_context.user_content.parts = [mock_part]
    mock_context.state = {}

    await sanitize_input_callback(mock_context)

    assert "scope_lock_trigger" in mock_context.state
    assert mock_context.state["scope_lock_trigger"] == "DIAGNOSTIC_QUERY"
