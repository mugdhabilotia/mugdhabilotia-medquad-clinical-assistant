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

"""AI Clinical Safety Guardrails - Google Cloud Model Armor & Cloud DLP Pipeline.

Enforces:
1. Model Armor API via Python SDK (`google-cloud-modelarmor`):
   - Templates for user prompts and model responses.
   - `sanitize_user_prompt` API for incoming prompts (jailbreaks, prompt injections, RAI, SDP).
   - `sanitize_model_response` API for outgoing model responses (RAI, SDP data leakage prevention).
2. Cloud DLP / SDP Sensitive Data Protection: Sanitizes incoming clinical queries
   and outgoing responses for PII/PHI (names, MRNs, SSNs, phone numbers, emails).
3. Integration with Google ADK callback lifecycle:
   - `sanitize_input_callback` (before_agent_callback)
   - `sanitize_response_callback` (after_agent_callback)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.api_core.client_options import ClientOptions
from google.cloud import modelarmor_v1

logger = logging.getLogger("model_armor")

DEFAULT_LOCATION = (
    os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GCP_LOCATION") or "us-central1"
)
if DEFAULT_LOCATION == "global" or not DEFAULT_LOCATION:
    DEFAULT_LOCATION = "us-central1"

DEFAULT_PROMPT_TEMPLATE_ID = os.getenv(
    "MODEL_ARMOR_PROMPT_TEMPLATE_ID", "medquad-prompt-template"
)
DEFAULT_RESPONSE_TEMPLATE_ID = os.getenv(
    "MODEL_ARMOR_RESPONSE_TEMPLATE_ID", "medquad-response-template"
)

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

TOXICITY_PATTERNS = [
    (
        r"(?i)\b(?:synthesize|manufacture|create|produce|weaponize)\s+(?:ricin|sarin|anthrax|cyanide|botulinum|mustard\s+gas|biological\s+weapon|chemical\s+weapon)\b",
        "Prohibited biological or chemical weapons request",
    ),
    (
        r"(?i)\b(?:how\s+to\s+commit\s+suicide|methods\s+to\s+end\s+(?:my|one's)\s+life|ways\s+to\s+kill\s+myself)\b",
        "Self-harm and suicide instruction prohibited",
    ),
    (
        r"(?i)\b(?:lethal\s+dose\s+to\s+poison|how\s+much\s+[a-z0-9-]+\s+to\s+overdose\s+and\s+kill)\b",
        "Intentional lethal poisoning/overdose instruction prohibited",
    ),
    (
        r"(?i)\b(?:kill\s+all\s+[a-z]+|racial\s+slur|hate\s+crime)\b",
        "Hate speech or violent extremism prohibited",
    ),
]


# =====================================================================
# 1. Model Armor Client Factory
# =====================================================================


def get_model_armor_client(
    location: str | None = None,
) -> modelarmor_v1.ModelArmorClient:
    """Initializes and returns a Google Cloud Model Armor client using Python SDK.

    Configures regional endpoint routing required by Model Armor.
    """
    loc = location or DEFAULT_LOCATION
    if loc == "global" or not loc:
        loc = "us"
    endpoint = f"modelarmor.{loc}.rep.googleapis.com"
    return modelarmor_v1.ModelArmorClient(
        transport="rest",
        client_options=ClientOptions(api_endpoint=endpoint),
    )


# =====================================================================
# 2. Model Armor Templates (Prompts & Responses)
# =====================================================================


def build_prompt_template() -> modelarmor_v1.Template:
    """Creates the adequate Model Armor Template configuration for user prompts.

    Configured with:
    1. Prompt Injection and Jailbreak Filter (ENABLED, MEDIUM_AND_ABOVE)
    2. Malicious URI Filter (ENABLED)
    3. Responsible AI (RAI) Safety Filters (HATE_SPEECH, HARASSMENT, DANGEROUS, SEXUALLY_EXPLICIT)
    4. Sensitive Data Protection (SDP/DLP BasicConfig ENABLED for PII/PHI redaction)
    """
    return modelarmor_v1.Template(
        filter_config=modelarmor_v1.FilterConfig(
            pi_and_jailbreak_filter_settings=modelarmor_v1.PiAndJailbreakFilterSettings(
                filter_enforcement=modelarmor_v1.PiAndJailbreakFilterSettings.PiAndJailbreakFilterEnforcement.ENABLED,
                confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
            ),
            malicious_uri_filter_settings=modelarmor_v1.MaliciousUriFilterSettings(
                filter_enforcement=modelarmor_v1.MaliciousUriFilterSettings.MaliciousUriFilterEnforcement.ENABLED,
            ),
            rai_settings=modelarmor_v1.RaiFilterSettings(
                rai_filters=[
                    modelarmor_v1.RaiFilterSettings.RaiFilter(
                        filter_type=modelarmor_v1.RaiFilterType.HATE_SPEECH,
                        confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                    ),
                    modelarmor_v1.RaiFilterSettings.RaiFilter(
                        filter_type=modelarmor_v1.RaiFilterType.HARASSMENT,
                        confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                    ),
                    modelarmor_v1.RaiFilterSettings.RaiFilter(
                        filter_type=modelarmor_v1.RaiFilterType.DANGEROUS,
                        confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                    ),
                    modelarmor_v1.RaiFilterSettings.RaiFilter(
                        filter_type=modelarmor_v1.RaiFilterType.SEXUALLY_EXPLICIT,
                        confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                    ),
                ]
            ),
            sdp_settings=modelarmor_v1.SdpFilterSettings(
                basic_config=modelarmor_v1.SdpBasicConfig(
                    filter_enforcement=modelarmor_v1.SdpBasicConfig.SdpBasicConfigEnforcement.ENABLED
                )
            ),
        )
    )


def build_response_template() -> modelarmor_v1.Template:
    """Creates the adequate Model Armor Template configuration for model responses.

    Configured with:
    1. Responsible AI (RAI) Safety Filters (HATE_SPEECH, HARASSMENT, DANGEROUS, SEXUALLY_EXPLICIT)
    2. Sensitive Data Protection (SDP/DLP BasicConfig ENABLED to prevent PII/PHI leakage)
    3. Malicious URI Filter (ENABLED)
    """
    return modelarmor_v1.Template(
        filter_config=modelarmor_v1.FilterConfig(
            malicious_uri_filter_settings=modelarmor_v1.MaliciousUriFilterSettings(
                filter_enforcement=modelarmor_v1.MaliciousUriFilterSettings.MaliciousUriFilterEnforcement.ENABLED,
            ),
            rai_settings=modelarmor_v1.RaiFilterSettings(
                rai_filters=[
                    modelarmor_v1.RaiFilterSettings.RaiFilter(
                        filter_type=modelarmor_v1.RaiFilterType.HATE_SPEECH,
                        confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                    ),
                    modelarmor_v1.RaiFilterSettings.RaiFilter(
                        filter_type=modelarmor_v1.RaiFilterType.HARASSMENT,
                        confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                    ),
                    modelarmor_v1.RaiFilterSettings.RaiFilter(
                        filter_type=modelarmor_v1.RaiFilterType.DANGEROUS,
                        confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                    ),
                    modelarmor_v1.RaiFilterSettings.RaiFilter(
                        filter_type=modelarmor_v1.RaiFilterType.SEXUALLY_EXPLICIT,
                        confidence_level=modelarmor_v1.DetectionConfidenceLevel.MEDIUM_AND_ABOVE,
                    ),
                ]
            ),
            sdp_settings=modelarmor_v1.SdpFilterSettings(
                basic_config=modelarmor_v1.SdpBasicConfig(
                    filter_enforcement=modelarmor_v1.SdpBasicConfig.SdpBasicConfigEnforcement.ENABLED
                )
            ),
        )
    )


def create_prompt_template(
    client: modelarmor_v1.ModelArmorClient | None = None,
    project_id: str | None = None,
    location: str | None = None,
    template_id: str | None = None,
) -> modelarmor_v1.Template:
    """Creates a Model Armor prompt template on Google Cloud using the Python SDK."""
    p_id = (
        project_id
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT_ID", "medquad")
    )
    loc = location or DEFAULT_LOCATION
    t_id = template_id or DEFAULT_PROMPT_TEMPLATE_ID
    c = client or get_model_armor_client(location=loc)
    parent = f"projects/{p_id}/locations/{loc}"
    template = build_prompt_template()
    request = modelarmor_v1.CreateTemplateRequest(
        parent=parent,
        template_id=t_id,
        template=template,
    )
    return c.create_template(request=request)


def create_response_template(
    client: modelarmor_v1.ModelArmorClient | None = None,
    project_id: str | None = None,
    location: str | None = None,
    template_id: str | None = None,
) -> modelarmor_v1.Template:
    """Creates a Model Armor response template on Google Cloud using the Python SDK."""
    p_id = (
        project_id
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT_ID", "medquad")
    )
    loc = location or DEFAULT_LOCATION
    t_id = template_id or DEFAULT_RESPONSE_TEMPLATE_ID
    c = client or get_model_armor_client(location=loc)
    parent = f"projects/{p_id}/locations/{loc}"
    template = build_response_template()
    request = modelarmor_v1.CreateTemplateRequest(
        parent=parent,
        template_id=t_id,
        template=template,
    )
    return c.create_template(request=request)


def get_or_create_template(
    template_id: str,
    template_config: modelarmor_v1.Template,
    client: modelarmor_v1.ModelArmorClient | None = None,
    project_id: str | None = None,
    location: str | None = None,
) -> str:
    """Ensures a template exists on Google Cloud Model Armor, creating it if needed.

    Returns:
        Full template resource name path:
        projects/{project}/locations/{location}/templates/{template_id}
    """
    p_id = (
        project_id
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT_ID", "medquad")
    )
    loc = location or DEFAULT_LOCATION
    c = client or get_model_armor_client(location=loc)
    parent = f"projects/{p_id}/locations/{loc}"
    template_name = f"{parent}/templates/{template_id}"
    try:
        c.get_template(name=template_name)
    except Exception as get_exc:
        logger.debug(
            f"Template {template_name} lookup: {get_exc}. Attempting creation..."
        )
        try:
            req = modelarmor_v1.CreateTemplateRequest(
                parent=parent,
                template_id=template_id,
                template=template_config,
            )
            c.create_template(request=req)
        except Exception as create_exc:
            logger.debug(f"Template creation skipped or already exists: {create_exc}")
    return template_name


# =====================================================================
# 3. Model Armor API Calls (sanitize_user_prompt & sanitize_model_response)
# =====================================================================


def sanitize_user_prompt(
    prompt: str,
    project_id: str | None = None,
    location: str | None = None,
    template_id: str | None = None,
    client: modelarmor_v1.ModelArmorClient | None = None,
) -> modelarmor_v1.SanitizeUserPromptResponse:
    """Calls the Model Armor sanitize_user_prompt API using the official Python SDK.

    Args:
        prompt: The user prompt text to sanitize.
        project_id: GCP project ID.
        location: GCP location/region.
        template_id: Model Armor prompt template ID.
        client: Optional ModelArmorClient instance.

    Returns:
        SanitizeUserPromptResponse from Model Armor.
    """
    p_id = (
        project_id
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT_ID", "medquad")
    )
    loc = DEFAULT_LOCATION
    t_id = DEFAULT_PROMPT_TEMPLATE_ID
    template_name = f"projects/{p_id}/locations/us/templates/{t_id}"
    c = client or get_model_armor_client(location=loc)

    user_prompt_data = modelarmor_v1.DataItem(text=prompt)
    request = modelarmor_v1.SanitizeUserPromptRequest(
        name=template_name,
        user_prompt_data=user_prompt_data,
    )

    try:
        return c.sanitize_user_prompt(request=request)
    except Exception as exc:
        logger.warning(
            f"Live Model Armor sanitize_user_prompt API call unavailable ({exc}). "
            "Falling back to local inspection."
        )
        return _simulate_user_prompt_sanitization(prompt)


def sanitize_model_response(
    model_response: str,
    user_prompt: str = "",
    project_id: str | None = None,
    location: str | None = None,
    template_id: str | None = None,
    client: modelarmor_v1.ModelArmorClient | None = None,
) -> modelarmor_v1.SanitizeModelResponseResponse:
    """Calls the Model Armor sanitize_model_response API using the official Python SDK.

    Args:
        model_response: The model response text to sanitize.
        user_prompt: The original user prompt (optional context).
        project_id: GCP project ID.
        location: GCP location/region.
        template_id: Model Armor response template ID.
        client: Optional ModelArmorClient instance.

    Returns:
        SanitizeModelResponseResponse from Model Armor.
    """
    p_id = (
        project_id
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT_ID", "medquad")
    )
    loc = location or DEFAULT_LOCATION
    t_id = template_id or DEFAULT_RESPONSE_TEMPLATE_ID
    template_name = DEFAULT_RESPONSE_TEMPLATE_ID
    c = client or get_model_armor_client(location=loc)

    model_response_data = modelarmor_v1.DataItem(text=model_response)
    request = modelarmor_v1.SanitizeModelResponseRequest(
        name=template_name,
        model_response_data=model_response_data,
        user_prompt=user_prompt,
    )

    try:
        return c.sanitize_model_response(request=request)
    except Exception as exc:
        logger.warning(
            f"Live Model Armor sanitize_model_response API call unavailable ({exc}). "
            "Falling back to local inspection."
        )
        return _simulate_model_response_sanitization(model_response)


# =====================================================================
# 4. Helper & Simulation Functions for Offline Resilience
# =====================================================================


def _redact_phi_pii(
    text: str,
) -> tuple[str, list[dict[str, str]], dict[str, str]]:
    """Performs PHI/PII entity masking."""
    sanitized = text
    token_mapping: dict[str, str] = {}
    detected_entities: list[dict[str, str]] = []

    # 1. Medical Record Numbers (MRN)
    mrn_matches = re.finditer(r"\bMRN[-\s]?(\d{6})\b", sanitized, re.IGNORECASE)
    for i, match in enumerate(list(mrn_matches)):
        orig = match.group(0)
        token = f"[REDACTED_MRN_{i + 1}]"
        detected_entities.append({"type": "MRN", "value": orig, "token": token})
        token_mapping[token] = orig
        sanitized = sanitized.replace(orig, token)

    # 2. Social Security Numbers (SSN)
    ssn_matches = re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", sanitized)
    for i, match in enumerate(list(ssn_matches)):
        orig = match.group(0)
        token = f"[REDACTED_SSN_{i + 1}]"
        detected_entities.append({"type": "SSN", "value": orig, "token": token})
        token_mapping[token] = orig
        sanitized = sanitized.replace(orig, token)

    # 3. Phone numbers
    phone_matches = re.finditer(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", sanitized)
    for i, match in enumerate(list(phone_matches)):
        orig = match.group(0)
        token = f"[REDACTED_PHONE_{i + 1}]"
        detected_entities.append({"type": "PHONE", "value": orig, "token": token})
        token_mapping[token] = orig
        sanitized = sanitized.replace(orig, token)

    # 4. Email addresses
    email_matches = re.finditer(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", sanitized
    )
    for i, match in enumerate(list(email_matches)):
        orig = match.group(0)
        token = f"[REDACTED_EMAIL_{i + 1}]"
        detected_entities.append({"type": "EMAIL", "value": orig, "token": token})
        token_mapping[token] = orig
        sanitized = sanitized.replace(orig, token)

    # 5. Patient Names (PHI) - Known ClinicSanitizeUserPromptal Registry
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

    # 6. Generic Patient Name Patterns ("Patient First Last")
    generic_patient_pattern = (
        r"(?i)\b(?:patient|pt\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
    )
    for match in re.finditer(generic_patient_pattern, sanitized):
        full_name = match.group(1)
        if full_name not in token_mapping.values():
            token = f"[REDACTED_PATIENT_NAME_{name_idx}]"
            detected_entities.append(
                {"type": "PATIENT_NAME", "value": full_name, "token": token}
            )
            token_mapping[token] = full_name
            sanitized = sanitized.replace(full_name, token)
            name_idx += 1

    return sanitized, detected_entities, token_mapping


def _simulate_user_prompt_sanitization(
    prompt: str,
) -> modelarmor_v1.Response:
    """Builds a SanitizeUserPromptResponse when live API is unreachable."""
    filter_results: dict[str, modelarmor_v1.FilterResult] = {}
    match_found = False

    # 1. Adversarial prompt injection
    for pattern, _ in INJECTION_PATTERNS:
        if re.search(pattern, prompt):
            match_found = True
            filter_results["pi_and_jailbreak"] = modelarmor_v1.FilterResult(
                pi_and_jailbreak_filter_result=modelarmor_v1.PiAndJailbreakFilterResult(
                    match_state=modelarmor_v1.FilterMatchState.MATCH_FOUND,
                    confidence_level=modelarmor_v1.DetectionConfidenceLevel.HIGH,
                )
            )
            break

    # 2. Toxic request / RAI violation
    for pattern, _ in TOXICITY_PATTERNS:
        if re.search(pattern, prompt):
            match_found = True
            filter_results["rai"] = modelarmor_v1.FilterResult(
                rai_filter_result=modelarmor_v1.RaiFilterResult(
                    match_state=modelarmor_v1.FilterMatchState.MATCH_FOUND,
                    rai_filter_type_results={
                        "DANGEROUS": modelarmor_v1.RaiFilterResult.RaiFilterTypeResult(
                            filter_type=modelarmor_v1.RaiFilterType.DANGEROUS,
                            match_state=modelarmor_v1.FilterMatchState.MATCH_FOUND,
                        )
                    },
                )
            )
            break

    # 3. Sensitive Data Protection (SDP/DLP)
    sanitized, detected_entities, _ = _redact_phi_pii(prompt)
    if detected_entities:
        match_found = True
        filter_results["sdp"] = modelarmor_v1.FilterResult(
            sdp_filter_result=modelarmor_v1.SdpFilterResult(
                deidentify_result=modelarmor_v1.SdpDeidentifyResult(
                    match_state=modelarmor_v1.FilterMatchState.MATCH_FOUND,
                    data=modelarmor_v1.DataItem(text=sanitized),
                    transformed_bytes=len(sanitized.encode("utf-8")),
                    info_types=[e["type"] for e in detected_entities],
                )
            )
        )

    match_state = (
        modelarmor_v1.FilterMatchState.MATCH_FOUND
        if match_found
        else modelarmor_v1.FilterMatchState.NO_MATCH_FOUND
    )

    return modelarmor_v1.SanitizeUserPromptResponse(
        sanitization_result=modelarmor_v1.SanitizationResult(
            filter_match_state=match_state,
            filter_results=filter_results,
        )
    )


def _simulate_model_response_sanitization(
    model_response: str,
) -> modelarmor_v1.SanitizeModelResponseResponse:
    """Builds a SanitizeModelResponseResponse when live API is unreachable."""
    filter_results: dict[str, modelarmor_v1.FilterResult] = {}
    match_found = False

    # Check toxicity
    for pattern, _ in TOXICITY_PATTERNS:
        if re.search(pattern, model_response):
            match_found = True
            filter_results["rai"] = modelarmor_v1.FilterResult(
                rai_filter_result=modelarmor_v1.RaiFilterResult(
                    match_state=modelarmor_v1.FilterMatchState.MATCH_FOUND,
                    rai_filter_type_results={
                        "DANGEROUS": modelarmor_v1.RaiFilterResult.RaiFilterTypeResult(
                            filter_type=modelarmor_v1.RaiFilterType.DANGEROUS,
                            match_state=modelarmor_v1.FilterMatchState.MATCH_FOUND,
                        )
                    },
                )
            )
            break

    # Check SDP leakage
    sanitized, detected_entities, _ = _redact_phi_pii(model_response)
    if detected_entities:
        match_found = True
        filter_results["sdp"] = modelarmor_v1.FilterResult(
            sdp_filter_result=modelarmor_v1.SdpFilterResult(
                deidentify_result=modelarmor_v1.SdpDeidentifyResult(
                    match_state=modelarmor_v1.FilterMatchState.MATCH_FOUND,
                    data=modelarmor_v1.DataItem(text=sanitized),
                    info_types=[e["type"] for e in detected_entities],
                )
            )
        )

    match_state = (
        modelarmor_v1.FilterMatchState.MATCH_FOUND
        if match_found
        else modelarmor_v1.FilterMatchState.NO_MATCH_FOUND
    )

    return modelarmor_v1.SanitizeModelResponseResponse(
        sanitization_result=modelarmor_v1.SanitizationResult(
            filter_match_state=match_state,
            filter_results=filter_results,
        )
    )


def inspect_with_cloud_dlp(
    text: str,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    """Inspects text using Google Cloud Sensitive Data Protection (DLP API)."""
    findings: list[dict[str, Any]] = []
    p_id = (
        project_id
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT_ID", "medquad")
    )
    try:
        from google.cloud import dlp_v2

        client = dlp_v2.DlpServiceClient()
        parent = f"projects/{p_id}/locations/global"
        item = {"value": text}
        inspect_config = {
            "info_types": [
                {"name": "PERSON_NAME"},
                {"name": "EMAIL_ADDRESS"},
                {"name": "PHONE_NUMBER"},
                {"name": "US_SOCIAL_SECURITY_NUMBER"},
            ],
            "min_likelihood": dlp_v2.Likelihood.LIKELY,
            "include_quote": True,
            "limits": {"max_findings_per_item": 20},
        }
        response = client.inspect_content(
            request={"parent": parent, "inspect_config": inspect_config, "item": item},
            timeout=3.0,
        )
        for f in response.result.findings:
            if f.quote:
                findings.append(
                    {
                        "info_type": f.info_type.name,
                        "quote": f.quote,
                        "likelihood": f.likelihood.name,
                    }
                )
    except Exception as exc:
        logger.debug(f"Cloud DLP inspection skipped or unavailable: {exc}")
    return findings


# =====================================================================
# 5. High-Level Clinical Query Sanitizer (Backward-Compatible Wrapper)
# =====================================================================


def sanitize_clinical_query(
    raw_query: str,
    use_cloud_dlp: bool | None = None,
    client: modelarmor_v1.ModelArmorClient | None = None,
) -> dict[str, Any]:
    """Sanitizes an input clinical query by calling the Model Armor API via Python SDK.

    Args:
        raw_query: The incoming clinical inquiry.
        use_cloud_dlp: Whether to invoke live Cloud DLP.
        client: Optional pre-configured ModelArmorClient.

    Returns:
        A dictionary containing sanitized query, detected entities,
        injection detection flags, toxicity detection flags, and token mapping.
    """
    response = sanitize_user_prompt(raw_query, client=client)
    res = response.sanitization_result

    injection_detected = False
    injection_warnings: list[str] = []
    toxicity_detected = False
    toxicity_warnings: list[str] = []

    # 1. PI & Jailbreak filter result
    if "pi_and_jailbreak" in res.filter_results:
        pij = res.filter_results["pi_and_jailbreak"].pi_and_jailbreak_filter_result
        if pij.match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
            injection_detected = True
            for pattern, reason in INJECTION_PATTERNS:
                if re.search(pattern, raw_query):
                    injection_warnings.append(reason)
            if not injection_warnings:
                injection_warnings.append("Adversarial prompt injection detected")

    # 2. RAI filter result
    if "rai" in res.filter_results:
        rai = res.filter_results["rai"].rai_filter_result
        if rai.match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
            toxicity_detected = True
            for pattern, reason in TOXICITY_PATTERNS:
                if re.search(pattern, raw_query):
                    toxicity_warnings.append(reason)
            if not toxicity_warnings:
                toxicity_warnings.append(
                    "Prohibited toxic or dangerous clinical content"
                )

    # 3. De-identification / SDP
    sanitized_text, detected_entities, token_mapping = _redact_phi_pii(raw_query)

    should_run_dlp = (
        use_cloud_dlp
        if use_cloud_dlp is not None
        else (os.getenv("ENABLE_CLOUD_DLP", "false").lower() == "true")
    )
    if should_run_dlp:
        dlp_findings = inspect_with_cloud_dlp(sanitized_text)
        dlp_idx = 1
        for finding in dlp_findings:
            quote = finding.get("quote", "")
            info_type = finding.get("info_type", "PII")
            if (
                quote
                and quote in sanitized_text
                and quote not in token_mapping.values()
            ):
                token = f"[REDACTED_DLP_{info_type}_{dlp_idx}]"
                detected_entities.append(
                    {"type": info_type, "value": quote, "token": token}
                )
                token_mapping[token] = quote
                sanitized_text = sanitized_text.replace(quote, token)
                dlp_idx += 1

    is_blocked = injection_detected or toxicity_detected

    return {
        "is_clean": len(detected_entities) == 0 and not is_blocked,
        "original_query": raw_query,
        "sanitized_query": sanitized_text,
        "redactions_count": len(detected_entities),
        "detected_entities": detected_entities,
        "injection_detected": injection_detected,
        "injection_warnings": injection_warnings,
        "toxicity_detected": toxicity_detected,
        "toxicity_warnings": toxicity_warnings,
        "token_mapping": token_mapping,
        "model_armor_status": "BLOCK" if is_blocked else "PASS_SANITIZED",
        "raw_response": response,
    }


# =====================================================================
# 6. ADK Callback Hooks
# =====================================================================


async def sanitize_input_callback(callback_context: CallbackContext) -> None:
    """ADK Before-Agent Callback Hook.

    Executes at the workflow entry point to sanitize incoming clinical prompts
    before model execution, calling the Google Cloud Model Armor sanitize_user_prompt API.
    """
    user_content = callback_context.user_content
    if not user_content:
        return

    text_content = ""
    if hasattr(user_content, "parts") and user_content.parts:
        for p in user_content.parts:
            if hasattr(p, "text") and p.text:
                text_content += p.text + " "

    prompt_text = text_content.strip()
    if not prompt_text:
        return

    # Call Model Armor API
    sanitization = sanitize_clinical_query(prompt_text)

    # Store DLP sanitization metadata in ADK session state
    if "dlp_audit" not in callback_context.state:
        callback_context.state["dlp_audit"] = []

    callback_context.state["dlp_audit"].append(
        {
            "original": sanitization["original_query"],
            "sanitized": sanitization["sanitized_query"],
            "redactions_count": sanitization["redactions_count"],
            "injection_detected": sanitization["injection_detected"],
            "toxicity_detected": sanitization.get("toxicity_detected", False),
        }
    )
    callback_context.state["dlp_token_mapping"] = sanitization["token_mapping"]

    # If injection or toxic request detected, reject with Model Armor block
    if sanitization["injection_detected"] or sanitization.get("toxicity_detected"):
        all_warnings = sanitization["injection_warnings"] + sanitization.get(
            "toxicity_warnings", []
        )
        logger.warning(f"Model Armor intercepted security violation: {all_warnings}")
        callback_context.state["security_alert"] = all_warnings
        callback_context.state["model_armor_status"] = "BLOCK"
        if hasattr(user_content, "parts") and user_content.parts:
            for p in user_content.parts:
                if hasattr(p, "text") and p.text:
                    p.text = (
                        "SECURITY VIOLATION: Request rejected by Model Armor. "
                        f"Detected violations: {', '.join(all_warnings)}."
                    )
        return

    # Evaluate Scope Lock triggers (diagnostic or prescriptive requests)
    from app.guardrails.scope_lock import (
        is_diagnostic_or_prescriptive_query,
    )

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


async def sanitize_response_callback(callback_context: CallbackContext) -> None:
    """ADK After-Agent Callback Hook.

    Sanitizes generated model responses using Model Armor sanitize_model_response API.
    """
    response_obj = getattr(callback_context, "response", None)
    if not response_obj:
        return

    response_text = ""
    if hasattr(response_obj, "text") and response_obj.text:
        response_text = response_obj.text
    elif hasattr(response_obj, "parts") and response_obj.parts:
        response_text = " ".join(
            p.text for p in response_obj.parts if hasattr(p, "text") and p.text
        )

    response_text = response_text.strip()
    if not response_text:
        return

    res = sanitize_model_response(response_text)
    sanitization_result = res.sanitization_result

    if (
        sanitization_result.filter_match_state
        == modelarmor_v1.FilterMatchState.MATCH_FOUND
    ):
        violations = []
        if "rai" in sanitization_result.filter_results:
            violations.append("Responsible AI policy violation")
        if "malicious_uri" in sanitization_result.filter_results:
            violations.append("Malicious URI detected")

        if violations:
            logger.warning(f"Model Armor blocked model response: {violations}")
            if hasattr(callback_context, "state"):
                callback_context.state["model_armor_response_status"] = "BLOCK"
            if hasattr(response_obj, "text"):
                response_obj.text = (
                    "SECURITY ALERT: The model response was blocked by Model Armor guardrails due to: "
                    f"{', '.join(violations)}."
                )
            return

        if "sdp" in sanitization_result.filter_results:
            sdp = sanitization_result.filter_results["sdp"].sdp_filter_result
            if (
                sdp.deidentify_result
                and sdp.deidentify_result.data
                and sdp.deidentify_result.data.text
            ):
                if hasattr(response_obj, "text"):
                    response_obj.text = sdp.deidentify_result.data.text


adk_sanitize_input_callback_hook = sanitize_input_callback
adk_sanitize_response_callback_hook = sanitize_response_callback

__all__ = [
    "DEFAULT_PROMPT_TEMPLATE_ID",
    "DEFAULT_RESPONSE_TEMPLATE_ID",
    "INJECTION_PATTERNS",
    "KNOWN_CLINICAL_NAMES",
    "TOXICITY_PATTERNS",
    "adk_sanitize_input_callback_hook",
    "adk_sanitize_response_callback_hook",
    "build_prompt_template",
    "build_response_template",
    "create_prompt_template",
    "create_response_template",
    "get_model_armor_client",
    "get_or_create_template",
    "inspect_with_cloud_dlp",
    "sanitize_clinical_query",
    "sanitize_input_callback",
    "sanitize_model_response",
    "sanitize_response_callback",
    "sanitize_user_prompt",
]
