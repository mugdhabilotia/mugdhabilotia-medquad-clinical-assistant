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

import logging
import os

logger = logging.getLogger("medquad.mcp.secrets")


class SecretManagerResolver:
    """Lightweight secret resolver.

    In Google Cloud Run and Agent Runtime, secrets from Secret Manager are mounted
    directly as environment variables via the Cloud Console UI or gcloud CLI:
        gcloud run deploy --set-secrets="EHR_DB_PASSWORD=EHR_DB_PASSWORD:latest"
    This eliminates the need for separate SDK client calls and API latency at runtime.
    """

    def __init__(self, ttl_seconds: int = 300):
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, str] = {}

    def get_secret(
        self,
        secret_id: str,
        project_id: str | None = None,
        version_id: str = "latest",
        default: str | None = None,
    ) -> str | None:
        if secret_id in self._cache:
            return self._cache[secret_id]
        val = os.environ.get(secret_id, default)
        if val is not None:
            self._cache[secret_id] = val
        return val


_resolver_instance = SecretManagerResolver()


def get_secret_resolver() -> SecretManagerResolver:
    """Returns the singleton SecretManagerResolver instance."""
    return _resolver_instance


def get_runtime_api_token(token_name: str, default: str = "") -> str:
    """Retrieves secret injected as an environment variable by Cloud Run / Cloud Console."""
    return os.getenv(token_name, default)
