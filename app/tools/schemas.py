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

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MedicalSearchInput(BaseModel):
    """Strict input schema for medical retrieval queries according to Sprint 3 specifications."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        description="Clinical keyword or natural language query",
        min_length=3,
        max_length=300,
    )
    max_chunks: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum count of grounded chunks to retrieve (between 1 and 10)",
    )


class PatientQueryInput(BaseModel):
    """Strict input schema for simulated EHR patient record queries."""

    model_config = ConfigDict(extra="forbid")

    patient_id: str = Field(
        ...,
        pattern=r"^(PAT|PT)-[0-9]{5}$",
        description="Simulated EHR patient ID (e.g. PAT-10492 or PT-10492)",
    )


class MedQuADSearchInput(BaseModel):
    """Pydantic schema enforcing strict parameter bounds for MedQuAD semantic retrieval."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Clinical or biomedical search inquiry (e.g., 'EGFR sensitizing mutations Osimertinib NSCLC')",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Strictly bounded count of citations to retrieve (between 1 and 10)",
    )
    domain_filter: str | None = Field(
        default=None,
        max_length=100,
        description="Optional medical specialty domain filter (e.g., 'Oncology', 'Cardiology', 'Endocrinology')",
    )


class ClinicalDBQueryInput(BaseModel):
    """Pydantic schema enforcing strict format bounds for Electronic Health Record database queries."""

    model_config = ConfigDict(extra="forbid")

    patient_id_or_mrn: str | None = Field(
        default=None,
        pattern=r"^(PT-\d{5}|MRN-\d{6})?$",
        description="Strictly formatted patient identifier (e.g., 'PT-10492' or 'MRN-849201')",
    )
    sql_query: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional parameterized read-only SELECT query against Cloud SQL PostgreSQL",
    )
    params: list[Any] = Field(
        default_factory=list,
        max_length=20,
        description="Optional positional parameters for parameterized SQL query ($1, $2, ...)",
    )
    domains: list[str] = Field(
        default=[
            "demographics",
            "conditions",
            "medications",
            "lab_results",
            "vital_signs",
        ],
        description="Specific EHR clinical data domains to query",
    )
    search_term: str | None = Field(
        default=None,
        max_length=200,
        description="Optional clinical keyword or condition to search across patient cohort",
    )
    query_type: str = Field(
        default="summary",
        description="Scope of clinical inquiry ('summary', 'labs', 'medications', 'cohort')",
    )


class ToolExecutionResult(BaseModel):
    """Standardized result envelope for tool executions."""

    status: str = Field(
        ...,
        description="Execution status code (e.g., 'SUCCESS', 'FALLBACK_SUCCESS', 'ERROR_RATE_LIMITED')",
    )
    data: Any = Field(default=None, description="Result payload data")
    error: str | None = Field(
        default=None, description="Error message if execution failed"
    )
    http_code: int | None = Field(
        default=None, description="HTTP status code if an HTTP error occurred"
    )
    retryable: bool = Field(
        default=False,
        description="Whether the client or agent can retry the operation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution tracing and timing metadata"
    )
