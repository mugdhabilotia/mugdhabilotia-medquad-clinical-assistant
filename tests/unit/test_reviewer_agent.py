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

"""Unit test suite verifying the Google ADK 2.0 Clinical Reviewer Subagent (`reviewer_agent.py`).

Covers:
1. ReviewOutput Pydantic v2 structured output schema & private attribute patch.
2. ADK 2.0 Task-mode agent definition & model configurations (Gemini 3.5 Flash).
3. 5-point clinical verification protocol & guardrails enforcement.
"""

import pytest
from pydantic import ValidationError

from app.agents.reviewer_agent import (
    REVIEWER_INSTRUCTION,
    REVIEWER_MODEL,
    AuditVerdict,
    ReviewOutput,
    create_reviewer_agent,
    reviewer_agent,
    reviewer_subagent,
)


def test_review_output_schema_and_verdicts():
    """Verify ReviewOutput valid instantiation, verdict enum, and boundary validation."""
    # 1. Successful creation
    valid_review = ReviewOutput(
        verdict=AuditVerdict.APPROVED,
        evidence_confidence_score=94.5,
        citation_accuracy_score=98.0,
        safety_alerts=[],
        has_hallucinations=False,
        auditor_rationale="Evidence aligns directly with NCI clinical guidelines; no renal contraindications detected.",
        recommended_adjustments=["Routine monitoring of CBC and metabolic panel."],
    )
    assert valid_review.verdict == AuditVerdict.APPROVED
    assert valid_review.evidence_confidence_score == 94.5
    assert valid_review.citation_accuracy_score == 98.0
    assert not valid_review.has_hallucinations
    assert len(valid_review.recommended_adjustments) == 1

    # 2. Pydantic v2 private attribute compatibility patch
    assert hasattr(valid_review, "__pydantic_private_")
    assert valid_review.__pydantic_private_ is None

    # 3. Serialization roundtrip
    dumped = valid_review.model_dump()
    assert dumped["verdict"] == "APPROVED"
    reconstructed = ReviewOutput.model_validate(dumped)
    assert reconstructed.verdict == AuditVerdict.APPROVED


def test_review_output_score_bounds_validation():
    """Verify ReviewOutput enforces 0.0 to 100.0 score bounds."""
    # Out-of-bounds evidence score (>100)
    with pytest.raises(ValidationError):
        ReviewOutput(
            verdict=AuditVerdict.APPROVED,
            evidence_confidence_score=105.0,  # Invalid
            citation_accuracy_score=90.0,
            safety_alerts=[],
            has_hallucinations=False,
            auditor_rationale="Out of bounds test",
        )

    # Negative citation score (<0)
    with pytest.raises(ValidationError):
        ReviewOutput(
            verdict=AuditVerdict.REJECTED,
            evidence_confidence_score=50.0,
            citation_accuracy_score=-5.0,  # Invalid
            safety_alerts=["Direct contraindication"],
            has_hallucinations=True,
            auditor_rationale="Out of bounds negative score test",
        )


def test_reviewer_agent_definition():
    """Verify reviewer subagent configuration in ADK 2.0 with mode='task' and output_schema."""
    assert reviewer_agent.name == "reviewer_agent"
    assert reviewer_agent.mode == "task"
    assert reviewer_agent.output_schema == ReviewOutput
    assert reviewer_agent.model.model == REVIEWER_MODEL
    assert REVIEWER_MODEL in ["gemini-2.5-flash", "gemini-2.5-flash"]
    assert reviewer_subagent is reviewer_agent

    # Verify task-mode automatically equips FinishTaskTool
    tool_types = [t.__class__.__name__ for t in reviewer_agent.tools]
    assert "FinishTaskTool" in tool_types

    # Verify custom instance creation
    custom_reviewer = create_reviewer_agent(name="custom_auditor")
    assert custom_reviewer.name == "custom_auditor"
    assert custom_reviewer.mode == "task"
    assert custom_reviewer.output_schema == ReviewOutput


def test_reviewer_instruction_guardrails():
    """Verify REVIEWER_INSTRUCTION enforces the 5-point clinical verification protocol."""
    instruction = REVIEWER_INSTRUCTION

    # 1. Evidence & citation audit
    assert "EVIDENCE & CITATION AUDIT" in instruction
    assert "MedQuAD" in instruction
    assert "has_hallucinations=True" in instruction

    # 2. Safety & contraindications
    assert "SAFETY & CONTRAINDICATION CHECK" in instruction
    assert "eGFR" in instruction
    assert "creatinine" in instruction
    assert "potassium" in instruction

    # 3. Drug-Drug Interactions
    assert "DRUG-DRUG INTERACTIONS" in instruction
    assert "Warfarin" in instruction

    # 4. Quantitative scoring
    assert "evidence_confidence_score" in instruction
    assert "citation_accuracy_score" in instruction

    # 5. Deterministic verdict & finish_task
    assert "DETERMINISTIC AUDIT VERDICT" in instruction
    assert "APPROVED" in instruction
    assert "APPROVED_WITH_CAUTIONS" in instruction
    assert "REJECTED" in instruction
    assert "finish_task" in instruction
