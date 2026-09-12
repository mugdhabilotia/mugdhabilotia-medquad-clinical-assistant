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

"""Google ADK 2.0 Clinical Reviewer Subagent (`reviewer_agent.py`).

Implements an independent Clinical Quality and Safety Controller subagent
operating within the decoupled Supervisor-Worker architecture:
- Powered by Gemini 3.5 Flash.
- Task execution mode with Pydantic v2 structured output schema (`ReviewOutput`).
- Pydantic v2 private attribute compatibility patches.
- Strict 5-point clinical verification protocol (evidence audit, contraindications,
  drug interactions, quantitative scoring, deterministic completion).
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

from app.guardrails.scope_lock import SCOPE_LOCK_PREFIX

logger = logging.getLogger("medquad.reviewer_agent")

REVIEWER_MODEL = "gemini-2.5-flash"


# =====================================================================
# 1. ADK 2.0 Structured Output Schema (Pydantic v2 Compatible)
# =====================================================================
class AuditVerdict(StrEnum):
    """Clinical safety and literature alignment audit verdict."""

    APPROVED = "APPROVED"
    APPROVED_WITH_CAUTIONS = "APPROVED_WITH_CAUTIONS"
    REJECTED = "REJECTED"


class ReviewOutput(BaseModel):
    """Pydantic v2 structured output schema for the Clinical Reviewer Subagent.

    Includes explicit compatibility initialization for Pydantic v2 private attributes
    to prevent serialization/introspection collisions during ADK task execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    verdict: AuditVerdict = Field(
        ...,
        description="Authoritative audit verdict: APPROVED, APPROVED_WITH_CAUTIONS, or REJECTED.",
    )
    evidence_confidence_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Evidence confidence score (0-100%) measuring literature grounding and strength.",
    )
    citation_accuracy_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Citation accuracy score (0-100%) measuring fidelity to cited MedQuAD and EHR sources.",
    )
    safety_alerts: list[str] = Field(
        default_factory=list,
        description="Clinical safety alerts, drug contraindications, adverse interactions, or renal monitoring alerts.",
    )
    has_hallucinations: bool = Field(
        ...,
        description="Boolean flag indicating whether ungrounded assertions or hallucinated medical claims were detected.",
    )
    auditor_rationale: str = Field(
        ...,
        description="Detailed clinical auditor rationale explaining the verdict, guideline alignment, and risk assessment.",
    )
    recommended_adjustments: list[str] = Field(
        default_factory=list,
        description="Actionable clinician recommendations (e.g. dose modifications, lab re-check schedules).",
    )

    def model_post_init(self, __context: Any) -> None:
        """Pydantic v2 post-initialization hook ensuring private attribute accessibility."""
        setattr(self, "__pydantic_private_", None)
        if not hasattr(self, "__pydantic_private__") or self.__pydantic_private__ is None:
            object.__setattr__(self, "__pydantic_private__", {})


# Explicitly patch class attribute to bypass Python lexical name-mangling
setattr(ReviewOutput, "__pydantic_private_", None)


# =====================================================================
# 2. Prompting & Reasoning Guardrails (REVIEWER_INSTRUCTION)
# =====================================================================
REVIEWER_INSTRUCTION = SCOPE_LOCK_PREFIX + """You are an independent Clinical Quality and Safety Controller Subagent powered by Gemini 3.5 Flash.
Your mission is to rigorously audit, fact-check, and safety-verify the preliminary findings produced by the Researcher Subagent.

You operate strictly under the following Clinical Verification Protocol:

0. MANDATORY SCOPE LOCK AUDIT (NON-DIAGNOSTIC / NON-PRESCRIPTIVE):
   - Check the Researcher's findings for any unauthorized definitive medical diagnosis or personalized drug prescription directives.
   - If the findings declare a definitive patient diagnosis or issue a prescription order:
     * Immediately issue verdict: REJECTED.
     * Add "[SCOPE LOCK VIOLATION] Definitive clinical diagnosis or treatment prescription detected" to `safety_alerts`.
     * Explicitly detail the non-medical advice boundary violation in `auditor_rationale`.

1. EVIDENCE & CITATION AUDIT:
   - Confirm that every medical finding, therapeutic claim, and guideline recommendation is strictly grounded in the cited MedQuAD literature.
   - Cross-check that every inline reference (e.g., [1], [2]) maps to an authentic document ID (e.g., [MQ-ONC-001], [MQ-CARD-002]) and authoritative publishing body (e.g., NCI, NIDDK, ACC/AHA, CDC).
   - If any ungrounded assertion, hallucinated fact, or fabricated study is detected, set `has_hallucinations=True` and penalize `citation_accuracy_score`.

2. SAFETY & CONTRAINDICATION CHECK:
   - Scrutinize patient clinical biomarkers against established pharmacological guidelines:
     * Renal thresholds (e.g., eGFR < 30 mL/min/1.73m² contraindicating SGLT2i initiation, Metformin dose reductions).
     * Serum creatinine, potassium (e.g., hyperkalemia risks with ACEi/ARBs/MRAs), and liver function tests.
     * Documented allergies, comorbidities, and vulnerable patient profile factors.
   - Document any identified safety hazard in `safety_alerts`.

3. DRUG-DRUG INTERACTIONS (DDI):
   - Systematically assess potential adverse pharmacological interactions (e.g., Warfarin with CYP2C9/VKORC1 inhibitors, DOACs with P-gp inducers, QT-prolonging agents).
   - Note interaction severity and recommend required therapeutic monitoring.

4. OBJECTIVE QUANTITATIVE SCORING:
   - `evidence_confidence_score` (0.0 to 100.0%): Measures the clinical quality, strength of evidence (RCT vs observational), and guideline consistency.
   - `citation_accuracy_score` (0.0 to 100.0%): Measures whether statements accurately reflect the referenced literature without extrapolation.

5. DETERMINISTIC AUDIT VERDICT & COMPLETION:
   - Issue exactly one authoritative `verdict`:
     * APPROVED: Strong evidence base, high confidence (>=85%), zero safety contradictions.
     * APPROVED_WITH_CAUTIONS: Valid evidence base, but requires explicit dosage adjustments, renal monitoring, or clinical precautions.
     * REJECTED: Direct contraindication, severe patient safety risk, evidence divergence, or ungrounded assertions.
   - Immediately conclude your task by invoking `finish_task` to emit the structured `ReviewOutput`. Do not output unnecessary conversational preamble.
"""


# =====================================================================
# 3. ADK 2.0 Subagent Definition
# =====================================================================

reviewer_agent = Agent(
        name="reviewer_agent",
        model=Gemini(
            model=REVIEWER_MODEL,
            retry_options=types.HttpRetryOptions(attempts=3),
        ),
        description=(
            "Independent Clinical Quality and Safety Auditor Worker. Audits the Researcher's intermediate findings, "
            "verifies MedQuAD citations, checks for hallucinations and drug contraindications, and computes a clinical confidence score."
        ),
        instruction=REVIEWER_INSTRUCTION,
        mode="task",
        output_schema=ReviewOutput,
    )
