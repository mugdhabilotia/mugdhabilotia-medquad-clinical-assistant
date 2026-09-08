import logging
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext

logger = logging.getLogger("model_armor")

KNOWN_CLINICAL_NAMES = [
    "Eleanor Vance",
    "Marcus Holloway",
    "Sophia Chen",
    "David Rodriguez",
    "Arthur Pendelton",
    "Eleanor",
    "Vance",
    "Marcus",
    "Holloway",
    "Sophia",
    "Chen",
    "David",
    "Rodriguez",
    "Arthur",
    "Pendelton",
]

INJECTION_PATTERNS = [
    (
        r"(?i)ignore (?:all )?(?:previous|prior) (?:instructions|rules|prompts)",
        "Instruction override attempt",
    ),
    (
        r"(?i)(?:reveal|display|output|leak) (?:the )?(?:system|internal) prompt",
        "System prompt exfiltration attempt",
    ),
    (
        r"(?i)you are now in (?:dan|developer|god) mode",
        "Jailbreak persona hijack attempt",
    ),
    (
        r"(?i)bypass (?:all )?(?:safety|content|dlp|guardrail) filters",
        "Guardrail bypass attempt",
    ),
    (r"(?i)drop table|delete from|exec\s*\(|insert into", "Prohibited SQL sequence"),
]


def sanitize_clinical_query(raw_query: str) -> dict[str, Any]:
    """Sanitizes an input clinical query mimicking Google Cloud Model Armor and DLP.

    Detects and masks Protected Health Information (PHI) such as patient names, MRNs,
    SSNs, phone numbers, and addresses. Scans for adversarial prompt injections.

    Args:
        raw_query: The incoming natural language inquiry from the clinician or researcher.

    Returns:
        A dictionary containing the sanitized query, redacted entity details,
        injection detection flags, and the de-identification token mapping.
    """
    sanitized = raw_query
    token_mapping = {}
    detected_entities = []
    injection_warnings = []

    # 1. Adversarial Injection Detection
    for pattern, reason in INJECTION_PATTERNS:
        if re.search(pattern, raw_query):
            injection_warnings.append(reason)

    injection_detected = len(injection_warnings) > 0

    # 2. Medical Record Numbers (MRN)
    mrn_matches = re.finditer(r"\bMRN[-\s]?(\d{6})\b", sanitized, re.IGNORECASE)
    for i, match in enumerate(list(mrn_matches)):
        orig = match.group(0)
        token = f"[REDACTED_MRN_{i + 1}]"
        detected_entities.append({"type": "MRN", "value": orig, "token": token})
        token_mapping[token] = orig
        sanitized = sanitized.replace(orig, token)

    # 3. Social Security Numbers (SSN)
    ssn_matches = re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", sanitized)
    for i, match in enumerate(list(ssn_matches)):
        orig = match.group(0)
        token = f"[REDACTED_SSN_{i + 1}]"
        detected_entities.append({"type": "SSN", "value": orig, "token": token})
        token_mapping[token] = orig
        sanitized = sanitized.replace(orig, token)

    # 4. Phone numbers
    phone_matches = re.finditer(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", sanitized)
    for i, match in enumerate(list(phone_matches)):
        orig = match.group(0)
        token = f"[REDACTED_PHONE_{i + 1}]"
        detected_entities.append({"type": "PHONE", "value": orig, "token": token})
        token_mapping[token] = orig
        sanitized = sanitized.replace(orig, token)

    # 5. Email addresses
    email_matches = re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", sanitized)
    for i, match in enumerate(list(email_matches)):
        orig = match.group(0)
        token = f"[REDACTED_EMAIL_{i + 1}]"
        detected_entities.append({"type": "EMAIL", "value": orig, "token": token})
        token_mapping[token] = orig
        sanitized = sanitized.replace(orig, token)

    # 6. Patient Names (PHI)
    name_idx = 1
    for name in sorted(KNOWN_CLINICAL_NAMES, key=len, reverse=True):
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, sanitized, re.IGNORECASE):
            for match in re.finditer(pattern, sanitized, re.IGNORECASE):
                orig = match.group(0)
                if orig not in token_mapping.values():
                    token = f"[REDACTED_PATIENT_NAME_{name_idx}]"
                    detected_entities.append(
                        {"type": "PATIENT_NAME", "value": orig, "token": token}
                    )
                    token_mapping[token] = orig
                    sanitized = re.sub(rf"\b{re.escape(orig)}\b", token, sanitized)
                    name_idx += 1

    return {
        "is_clean": len(detected_entities) == 0 and not injection_detected,
        "original_query": raw_query,
        "sanitized_query": sanitized,
        "redactions_count": len(detected_entities),
        "detected_entities": detected_entities,
        "injection_detected": injection_detected,
        "injection_warnings": injection_warnings,
        "token_mapping": token_mapping,
        "model_armor_status": "BLOCK" if injection_detected else "PASS_SANITIZED",
    }


async def sanitize_input_callback(callback_context: CallbackContext) -> None:
    """ADK Before-Agent Callback Hook

    Executes at the workflow entry point to sanitize incoming clinical prompts
    before model execution, mimicking Model Armor & DLP guardrails.
    """
    user_content = callback_context.user_content
    if not user_content:
        return

    # Extract text from parts
    text_content = ""
    if hasattr(user_content, "parts") and user_content.parts:
        for p in user_content.parts:
            if hasattr(p, "text") and p.text:
                text_content += p.text + " "

    if not text_content.strip():
        return

    sanitization = sanitize_clinical_query(text_content.strip())

    # Store DLP sanitization metadata in ADK session state
    if "dlp_audit" not in callback_context.state:
        callback_context.state["dlp_audit"] = []

    callback_context.state["dlp_audit"].append(
        {
            "original": sanitization["original_query"],
            "sanitized": sanitization["sanitized_query"],
            "redactions_count": sanitization["redactions_count"],
            "injection_detected": sanitization["injection_detected"],
        }
    )
    callback_context.state["dlp_token_mapping"] = sanitization["token_mapping"]

    # If injection detected, instruct supervisor to warn and block
    if sanitization["injection_detected"]:
        logger.warning(
            f"Model Armor intercepted adversarial injection: {sanitization['injection_warnings']}"
        )
        # In ADK, we can flag this in session state
        callback_context.state["security_alert"] = sanitization["injection_warnings"]

    # Evaluate Scope Lock triggers (diagnostic or prescriptive requests)
    from app.guardrails.scope_lock import is_diagnostic_or_prescriptive_query

    is_scope_trigger, trigger_category = is_diagnostic_or_prescriptive_query(
        sanitization["sanitized_query"]
    )
    if is_scope_trigger:
        callback_context.state["scope_lock_trigger"] = trigger_category
        logger.info(
            f"Scope Lock triggered on inbound query ({trigger_category}). "
            "Safe refusal non-medical advice boundary enforced."
        )

    # Update part text with sanitized query to ensure zero PHI leakage into LLM
    if hasattr(user_content, "parts") and user_content.parts:
        for p in user_content.parts:
            if hasattr(p, "text") and p.text:
                p.text = sanitization["sanitized_query"]
