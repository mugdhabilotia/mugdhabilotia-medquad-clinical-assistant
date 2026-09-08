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

"""AI Clinical Safety Guardrails - Hardcoded Scope Lock & Safe Refusal.

Implements the mandatory GCP enterprise safety guardrail ("Scope Lock"):
- Enforces an immutable non-medical advice boundary.
- Forbids definitive diagnostic assertions and therapeutic prescriptions.
- Mandates standardized safe refusal for diagnostic/prescriptive queries.
"""

from __future__ import annotations

import re

# =====================================================================
# 1. Hardcoded "Scope Lock" Prompt Prefix (Mandatory Across All Agents)
# =====================================================================
SCOPE_LOCK_PREFIX = """SCOPE LOCK: You are an educational medical research assistant. You provide grounded clinical literature summaries strictly based on retrieved NIH MedQuAD documents. You MUST NEVER diagnose conditions, recommend individualized treatments, or prescribe dosages. If asked for diagnosis or treatment, reply with the standard refusal.

[CRITICAL MANDATORY SCOPE LOCK: NON-MEDICAL ADVICE & SAFE REFUSAL]
You are an AI Clinical and Medical Research Assistant developed EXCLUSIVELY for biomedical literature search, clinical evidence summarization, and informational clinical decision support.
You are NOT a licensed physician, clinical diagnostician, or prescribing provider.

MANDATORY REFUSAL DIRECTIVE ("SCOPE LOCK"):
1. ABSOLUTE DIAGNOSTIC REFUSAL: You MUST strictly REFUSE to formulate, confirm, or issue a definitive clinical diagnosis for any patient or individual (e.g., "Does this patient have condition X?", "Diagnose this patient", or "What illness do I have?").
2. ABSOLUTE PRESCRIPTION & TREATMENT REFUSAL: You MUST strictly REFUSE to write, generate, prescribe, or finalize medication prescriptions, individual therapeutic orders, or patient-specific drug dosing commands (e.g., "Prescribe 80mg Osimertinib for patient X", "What exact dose should I prescribe to this patient?").
3. SAFE REFUSAL STANDARD RESPONSE: When presented with a direct diagnostic or prescription query, you MUST IMMEDIATELY execute Safe Refusal using the following standard statement:
   "I cannot provide a definitive medical diagnosis or prescribe clinical treatments. As an AI clinical research assistant, my scope is strictly restricted to synthesizing peer-reviewed biomedical literature and clinical guideline evidence for healthcare professionals."
4. INFORMATIONAL RESEARCH ONLY: You may only synthesize peer-reviewed medical guidelines, published clinical trial statistics (e.g. MedQuAD index), and physiological mechanisms in an educational, non-prescriptive, and non-diagnostic research persona.
[END SCOPE LOCK]

"""

# Standardized safe refusal message
SAFE_REFUSAL_RESPONSE = (
    "I cannot provide a definitive medical diagnosis or prescribe clinical treatments. "
    "As an AI clinical research assistant, my scope is strictly restricted to synthesizing "
    "peer-reviewed biomedical literature and clinical guideline evidence for healthcare professionals."
)

# Patterns that trigger scope lock safe refusal
DIAGNOSTIC_QUERY_PATTERNS = [
    r"(?i)\bdiagnose\b",
    r"(?i)\bwhat\s+(?:illness|disease|condition|disorder)\s+do\s+(?:i|they|we|this\s+patient)\s+have\b",
    r"(?i)\bdo\s+i\s+have\s+[a-z0-9\s-]+\b",
    r"(?i)\bdoes\s+(?:the\s+)?(?:patient\s+)?[a-z0-9_-]+\s+have\s+[a-z0-9\s-]+\b",
    r"(?i)\bgive\s+(?:me\s+)?a\s+(?:definitive\s+)?diagnosis\b",
    r"(?i)\bconfirm\s+(?:that\s+)?(?:i|the\s+patient)\s+have\b",
]

PRESCRIPTION_QUERY_PATTERNS = [
    r"(?i)\bprescribe\b",
    r"(?i)\bwrite\s+(?:me\s+)?a\s+prescription\b",
    r"(?i)\bwhat\s+dose\s+should\s+i\s+prescribe\b",
    r"(?i)\border\s+(?:this\s+)?medication\s+for\b",
    r"(?i)\bprescribe\s+[a-z0-9-]+\s+to\s+patient\b",
    r"(?i)\bstart\s+patient\s+on\s+prescription\b",
]

VIOLATION_OUTPUT_PATTERNS = [
    (r"(?i)\bi\s+diagnose\s+(?:you|the\s+patient)\s+with\b", "Definitive diagnostic declaration"),
    (r"(?i)\bmy\s+diagnosis\s+is\b", "Direct diagnostic statement"),
    (r"(?i)\bi\s+hereby\s+prescribe\b", "Direct prescription issuance"),
    (r"(?i)\bi\s+prescribe\s+[a-z0-9-]+\s+at\s+\d+\b", "Direct medication dosing prescription"),
]


def is_diagnostic_or_prescriptive_query(query: str) -> tuple[bool, str | None]:
    """Evaluates whether an inbound clinical inquiry requests a direct medical diagnosis

    or clinical prescription, triggering mandatory Scope Lock safe refusal.

    Args:
        query: Inbound clinician or researcher query text.

    Returns:
        A tuple of (is_triggered, trigger_category).
    """
    for pattern in DIAGNOSTIC_QUERY_PATTERNS:
        if re.search(pattern, query):
            return True, "DIAGNOSTIC_QUERY"

    for pattern in PRESCRIPTION_QUERY_PATTERNS:
        if re.search(pattern, query):
            return True, "PRESCRIPTION_QUERY"

    return False, None


def detect_scope_lock_violation(output_text: str) -> list[str]:
    """Audits agent output text to verify strict compliance with the non-medical advice boundary.

    Args:
        output_text: Generated agent findings or synthesis.

    Returns:
        List of identified Scope Lock violations, if any.
    """
    violations = []
    for pattern, reason in VIOLATION_OUTPUT_PATTERNS:
        if re.search(pattern, output_text):
            violations.append(reason)
    return violations
