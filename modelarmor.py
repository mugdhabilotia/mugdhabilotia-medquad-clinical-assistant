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

"""Model Armor module root alias for app.guardrails.model_armor."""

from app.guardrails.model_armor import (
    DEFAULT_PROMPT_TEMPLATE_ID,
    DEFAULT_RESPONSE_TEMPLATE_ID,
    INJECTION_PATTERNS,
    KNOWN_CLINICAL_NAMES,
    TOXICITY_PATTERNS,
    adk_sanitize_input_callback_hook,
    adk_sanitize_response_callback_hook,
    build_prompt_template,
    build_response_template,
    create_prompt_template,
    create_response_template,
    get_model_armor_client,
    get_or_create_template,
    inspect_with_cloud_dlp,
    sanitize_clinical_query,
    sanitize_input_callback,
    sanitize_model_response,
    sanitize_response_callback,
    sanitize_user_prompt,
)

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
