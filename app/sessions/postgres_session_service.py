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

import asyncio
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

from google.adk.events.event import Event
from google.adk.sessions.base_session_service import (
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.session import Session
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

logger = logging.getLogger("medquad.session_service")

DEFAULT_POSTGRES_URL = (
    "postgresql+asyncpg://postgres:postgres@localhost:5432/medquad_clinical"
)
DEFAULT_FALLBACK_URL = "sqlite+aiosqlite:///:memory:"
DEFAULT_MAX_SHORT_TERM_EVENTS = 8


class ActiveStateCache:
    """In-memory active state cache for low-latency retrieval of conversational history

    and short-term clinical context.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _make_key(app_name: str, user_id: str, session_id: str) -> str:
        return f"{app_name}::{user_id}::{session_id}"

    async def get(self, app_name: str, user_id: str, session_id: str) -> Session | None:
        """Retrieve session from active memory cache."""
        async with self._lock:
            key = self._make_key(app_name, user_id, session_id)
            entry = self._cache.get(key)
            if not entry:
                return None
            entry["last_accessed"] = datetime.now(UTC).timestamp()
            return entry["session"]

    async def put(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        session: Session,
        short_term_context: dict[str, Any] | None = None,
    ) -> None:
        """Store or update session and short-term context in active memory cache."""
        async with self._lock:
            key = self._make_key(app_name, user_id, session_id)
            self._cache[key] = {
                "session": session,
                "short_term_context": short_term_context or {},
                "last_accessed": datetime.now(UTC).timestamp(),
            }

    async def evict(self, app_name: str, user_id: str, session_id: str) -> None:
        """Evict session from active memory cache."""
        async with self._lock:
            key = self._make_key(app_name, user_id, session_id)
            self._cache.pop(key, None)

    async def get_short_term_context(
        self, app_name: str, user_id: str, session_id: str
    ) -> dict[str, Any]:
        """Fetch short-term working context variables (e.g. current patient info, temporary observations)."""
        async with self._lock:
            key = self._make_key(app_name, user_id, session_id)
            entry = self._cache.get(key)
            if entry:
                return dict(entry.get("short_term_context", {}))
            return {}

    async def update_short_term_context(
        self,
        app_name: str,
        user_id: str,
        session_id: str,
        context_delta: dict[str, Any],
    ) -> None:
        """Update short-term working context variables."""
        async with self._lock:
            key = self._make_key(app_name, user_id, session_id)
            if key in self._cache:
                self._cache[key]["short_term_context"].update(context_delta)

    async def clear(self) -> None:
        """Clear all entries in active memory cache."""
        async with self._lock:
            self._cache.clear()


class ClinicalSessionSummarizer:
    """Analyzes and condenses older conversation events into structured clinical summaries

    to prevent token bloat during extended multi-turn clinical research sessions.
    """

    @staticmethod
    def extract_event_text(event: Event) -> tuple[str, str]:
        """Extract author and textual representation from an Event."""
        author = event.author or "unknown"
        texts = []
        if event.content and event.content.parts:
            for p in event.content.parts:
                if hasattr(p, "text") and p.text:
                    texts.append(p.text)
                elif hasattr(p, "function_call") and p.function_call:
                    name = getattr(p.function_call, "name", "tool")
                    args = getattr(p.function_call, "args", {})
                    texts.append(f"Tool Call: {name}({args})")
                elif hasattr(p, "function_response") and p.function_response:
                    name = getattr(p.function_response, "name", "tool")
                    response = getattr(p.function_response, "response", {})
                    texts.append(f"Tool Response: {name} -> {response}")
        return author, " ".join(texts)

    @classmethod
    def condense_events(
        cls,
        events: list[Event],
        existing_summary: str = "",
    ) -> str:
        """Condenses older events into a high-density, structured clinical summary."""
        inquiries = []
        literature_citations = set()
        patient_records = []
        verdicts = []
        recommendations = []

        for evt in events:
            author, text = cls.extract_event_text(evt)
            if not text.strip():
                continue

            # Identify citations like [MQ-ONC-001]
            citations = re.findall(r"\[(MQ-[A-Z]+-\d+)\]", text)
            for cit in citations:
                literature_citations.add(cit)

            # Identify patient references
            pt_matches = re.findall(
                r"\b(PT-\d{5}|MRN-\d{6}|Eleanor Vance|Marcus Holloway|Sophia Chen)\b",
                text,
            )
            for pt in pt_matches:
                if pt not in patient_records:
                    patient_records.append(pt)

            # Identify reviewer verdicts
            verdict_matches = re.findall(
                r"\[(APPROVED|APPROVED_WITH_CAUTIONS|REJECTED)\]", text
            )
            for v in verdict_matches:
                verdicts.append(f"{v} (by {author})")

            if author == "user":
                inquiries.append(text[:250])
            elif "root_orchestrator" in author or author == "assistant":
                if "Summary:" in text or "Findings:" in text or "Verdict:" in text:
                    recommendations.append(text[:300])

        summary_lines = []
        if existing_summary.strip():
            summary_lines.append(existing_summary.strip())
            summary_lines.append("\n--- Incremental Session Update ---")

        summary_lines.append("### Cumulative Clinical Summary (Archived Context):")

        if inquiries:
            summary_lines.append("- **Prior Inquiries**: " + " | ".join(inquiries[-3:]))
        if patient_records:
            summary_lines.append(
                "- **Patient Records Referenced**: " + ", ".join(patient_records)
            )
        if literature_citations:
            summary_lines.append(
                "- **MedQuAD Citations Grounded**: "
                + ", ".join(sorted(literature_citations))
            )
        if verdicts:
            summary_lines.append(
                "- **Clinical Audit Verdicts**: " + "; ".join(verdicts)
            )
        if recommendations:
            summary_lines.append(
                "- **Formulated Clinical Guidance**: "
                + " | ".join(recommendations[-2:])
            )

        return "\n".join(summary_lines)


class PostgresClinicalSessionService(DatabaseSessionService):
    """Asynchronous PostgreSQL-backed ADK session service.

    Features:
    1. Asynchronous PostgreSQL backend using SQLAlchemy AsyncEngine and asyncpg.
    2. Active state caching: Conversational history and short-term context are cached
       in memory for low-latency access and fast response times.
    3. Token bloat mitigation: When conversational history exceeds `max_short_term_events`,
       older events are condensed into a structured clinical long-term summary persisted
       in PostgreSQL, while pruning the active event window.
    4. Graceful fallback: If PostgreSQL is unreachable during offline test runs,
       automatically falls back to an async in-memory SQLite engine while retaining
       the exact same caching and summarization pipeline.
    """

    def __init__(
        self,
        db_url: str | None = None,
        max_short_term_events: int = DEFAULT_MAX_SHORT_TERM_EVENTS,
        cache_ttl_seconds: int = 3600,
        **kwargs: Any,
    ):
        resolved_url, is_postgres = self._resolve_db_url(db_url)
        self.max_short_term_events = max_short_term_events
        self.is_postgres = is_postgres
        self.cache = ActiveStateCache(ttl_seconds=cache_ttl_seconds)
        self.summarizer = ClinicalSessionSummarizer()

        super().__init__(db_url=resolved_url, **kwargs)
        logger.info(
            f"Initialized PostgresClinicalSessionService with backend={self.db_engine.dialect.name}, "
            f"max_short_term_events={self.max_short_term_events}"
        )

    @staticmethod
    def _resolve_db_url(db_url: str | None) -> tuple[str, bool]:
        url = (
            db_url
            or os.environ.get("POSTGRES_DB_URL")
            or os.environ.get("DATABASE_URL")
            or os.environ.get("SESSION_SERVICE_URI")
        )
        if url and "postgres" in url:
            return url, True
        if url:
            return url, False

        # In testing or environments without live postgres, verify if localhost postgres is up
        return DEFAULT_POSTGRES_URL, True

    @classmethod
    async def create_resilient(
        cls,
        db_url: str | None = None,
        max_short_term_events: int = DEFAULT_MAX_SHORT_TERM_EVENTS,
        **kwargs: Any,
    ) -> PostgresClinicalSessionService:
        """Factory method that verifies PostgreSQL connection and falls back to async SQLite

        if the target database is unreachable (e.g. during local tests without a PostgreSQL server).
        """
        target_url, is_postgres = cls._resolve_db_url(db_url)
        if is_postgres:
            try:
                # Test connectivity with short timeout
                test_engine = create_async_engine(
                    target_url, connect_args={"timeout": 1.0}
                )
                async with asyncio.timeout(1.5):
                    async with test_engine.connect() as conn:
                        await conn.execute(select(1))
                await test_engine.dispose()
                logger.info("Successfully verified PostgreSQL connection.")
                return cls(
                    db_url=target_url,
                    max_short_term_events=max_short_term_events,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning(
                    f"PostgreSQL probe at {target_url} was not reachable ({exc}). "
                    f"Falling back to async SQLite engine ({DEFAULT_FALLBACK_URL}) "
                    f"with full active state caching and long-term session summarization."
                )
                return cls(
                    db_url=DEFAULT_FALLBACK_URL,
                    max_short_term_events=max_short_term_events,
                    **kwargs,
                )

        return cls(
            db_url=target_url,
            max_short_term_events=max_short_term_events,
            **kwargs,
        )

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Create a new session in the async database and register it in the active state cache."""
        session_state = dict(state or {})
        if "long_term_summary" not in session_state:
            session_state["long_term_summary"] = ""
        if "summary_turn_count" not in session_state:
            session_state["summary_turn_count"] = 0

        session = await super().create_session(
            app_name=app_name,
            user_id=user_id,
            state=session_state,
            session_id=session_id,
        )

        # Store in active memory cache for sub-millisecond retrieval
        await self.cache.put(app_name, user_id, session.id, session)
        return session

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: GetSessionConfig | None = None,
    ) -> Session | None:
        """Fetch session from active state cache if present; otherwise load from PostgreSQL."""
        # 1. Fast path: Check active memory cache
        cached_session = await self.cache.get(app_name, user_id, session_id)
        if cached_session is not None:
            return cached_session

        # 2. Cache miss: Load from PostgreSQL storage
        session = await super().get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            config=config,
        )
        if session is not None:
            # Prune events to short-term window if needed to avoid token bloat
            if len(session.events) > self.max_short_term_events:
                session.events = session.events[-self.max_short_term_events :]
            await self.cache.put(app_name, user_id, session_id, session)

        return session

    async def append_event(self, session: Session, event: Event) -> Event:
        """Persists the event in PostgreSQL, updates the active state cache,

        and triggers long-term clinical summarization when event count exceeds the token threshold.
        """
        # 1. Persist event to the asynchronous database backend
        persisted_event = await super().append_event(session, event)

        # 2. Keep in-memory event list synchronized
        if event not in session.events:
            session.events.append(event)

        # 3. Check for token bloat threshold
        if len(session.events) > self.max_short_term_events:
            await self._summarize_and_prune(session)

        # 4. Update active cache
        await self.cache.put(session.app_name, session.user_id, session.id, session)
        return persisted_event

    async def _summarize_and_prune(self, session: Session) -> None:
        """Condenses older events beyond the short-term window into the long-term session summary,

        persists the summary into PostgreSQL, and trims the active events window.
        """
        older_events = session.events[: -self.max_short_term_events]
        recent_events = session.events[-self.max_short_term_events :]

        if not older_events:
            return

        current_summary = session.state.get("long_term_summary", "")
        updated_summary = self.summarizer.condense_events(
            older_events, existing_summary=current_summary
        )

        turn_count = int(session.state.get("summary_turn_count", 0)) + 1
        pruned_count = int(session.state.get("pruned_events_count", 0)) + len(
            older_events
        )

        # Update in-memory session state
        session.state["long_term_summary"] = updated_summary
        session.state["summary_turn_count"] = turn_count
        session.state["pruned_events_count"] = pruned_count
        session.state["last_summarized_at"] = datetime.now(UTC).isoformat()

        # Prune active events list to prevent token bloat
        session.events = recent_events

        # Persist updated summary to database storage
        schema = self._get_schema_classes()
        is_sqlite = self.db_engine.dialect.name == "sqlite"
        is_postgresql = self.db_engine.dialect.name == "postgresql"
        async with self._with_session_lock(
            app_name=session.app_name,
            user_id=session.user_id,
            session_id=session.id,
        ):
            async with self._rollback_on_exception_session() as sql_session:
                storage_session = await sql_session.get(
                    schema.StorageSession,
                    (session.app_name, session.user_id, session.id),
                )
                if storage_session is not None:
                    db_state = dict(storage_session.state or {})
                    db_state["long_term_summary"] = updated_summary
                    db_state["summary_turn_count"] = turn_count
                    db_state["pruned_events_count"] = pruned_count
                    db_state["last_summarized_at"] = session.state["last_summarized_at"]
                    storage_session.state = db_state

                    now = datetime.now(UTC)
                    if self._uses_naive_datetime():
                        now = now.replace(tzinfo=None)
                    storage_session.update_time = now
                    await sql_session.commit()

                    session._storage_update_marker = storage_session.get_update_marker()
                    session.last_update_time = storage_session.get_update_timestamp(
                        is_sqlite=is_sqlite, is_postgresql=is_postgresql
                    )

        logger.info(
            f"Condensed {len(older_events)} older events into long-term summary for session {session.id}. "
            f"Active short-term window retained: {len(session.events)} events."
        )

    async def delete_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> None:
        """Delete session from PostgreSQL storage and evict from active memory cache."""
        await super().delete_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        await self.cache.evict(app_name, user_id, session_id)

    async def list_sessions(
        self, *, app_name: str, user_id: str | None = None
    ) -> ListSessionsResponse:
        """List sessions from the database backend."""
        return await super().list_sessions(app_name=app_name, user_id=user_id)


_global_session_service: PostgresClinicalSessionService | None = None


def get_postgres_clinical_session_service(
    db_url: str | None = None,
    max_short_term_events: int = DEFAULT_MAX_SHORT_TERM_EVENTS,
) -> PostgresClinicalSessionService:
    """Singleton getter for the PostgresClinicalSessionService."""
    global _global_session_service
    if _global_session_service is None:
        _global_session_service = PostgresClinicalSessionService(
            db_url=db_url,
            max_short_term_events=max_short_term_events,
        )
    return _global_session_service
