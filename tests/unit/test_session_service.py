from datetime import UTC, datetime

import pytest
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types

from app.sessions.postgres_session_service import (
    ActiveStateCache,
    ClinicalSessionSummarizer,
    PostgresClinicalSessionService,
)


@pytest.mark.asyncio
async def test_active_state_cache_operations():
    """Verify in-memory active state cache put, get, evict, and context manipulation."""
    cache = ActiveStateCache(ttl_seconds=60)
    session = Session(
        id="test-session-1",
        app_name="medquad-agent",
        user_id="clinician-1",
        state={"test_key": "test_val"},
        events=[],
        last_update_time=datetime.now(UTC).timestamp(),
    )

    # 1. Put and Get
    await cache.put(
        "medquad-agent",
        "clinician-1",
        "test-session-1",
        session,
        short_term_context={"active_patient": "PT-10492"},
    )
    cached = await cache.get("medquad-agent", "clinician-1", "test-session-1")
    assert cached is not None
    assert cached.id == "test-session-1"

    # 2. Short-term context retrieval and update
    ctx = await cache.get_short_term_context(
        "medquad-agent", "clinician-1", "test-session-1"
    )
    assert ctx["active_patient"] == "PT-10492"

    await cache.update_short_term_context(
        "medquad-agent",
        "clinician-1",
        "test-session-1",
        {"active_patient": "PT-20831", "stage": "Stage IV"},
    )
    updated_ctx = await cache.get_short_term_context(
        "medquad-agent", "clinician-1", "test-session-1"
    )
    assert updated_ctx["active_patient"] == "PT-20831"
    assert updated_ctx["stage"] == "Stage IV"

    # 3. Eviction
    await cache.evict("medquad-agent", "clinician-1", "test-session-1")
    evicted = await cache.get("medquad-agent", "clinician-1", "test-session-1")
    assert evicted is None


def test_clinical_session_summarizer_condensation():
    """Verify that ClinicalSessionSummarizer extracts citations, patients, and verdicts into a summary."""
    events = [
        Event(
            author="user",
            content=types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Investigate frontline EGFR inhibitors for Eleanor Vance (PT-10492)"
                    )
                ],
            ),
        ),
        Event(
            author="researcher_subagent",
            content=types.Content(
                role="assistant",
                parts=[
                    types.Part.from_text(
                        text="Retrieved [MQ-ONC-001] regarding Osimertinib efficacy in exon 19 deletions."
                    )
                ],
            ),
        ),
        Event(
            author="reviewer_subagent",
            content=types.Content(
                role="assistant",
                parts=[
                    types.Part.from_text(
                        text="Clinical Quality Audit: [APPROVED_WITH_CAUTIONS]. Baseline renal clearance verified."
                    )
                ],
            ),
        ),
    ]

    summary = ClinicalSessionSummarizer.condense_events(events)
    assert "Eleanor Vance" in summary or "PT-10492" in summary
    assert "MQ-ONC-001" in summary
    assert "APPROVED_WITH_CAUTIONS" in summary


@pytest.mark.asyncio
async def test_postgres_clinical_session_service_lifecycle_and_caching():
    """Verify session creation, active caching, and retrieval through the async session service."""
    # Uses in-memory async SQLite engine for offline unit test isolation
    service = PostgresClinicalSessionService(
        db_url="sqlite+aiosqlite:///:memory:",
        max_short_term_events=4,
    )
    await service.prepare_tables()

    # 1. Create Session
    session = await service.create_session(
        app_name="medquad-agent",
        user_id="doc_smith",
        session_id="session-clinical-001",
        state={"specialty": "Oncology"},
    )
    assert session.id == "session-clinical-001"
    assert "long_term_summary" in session.state

    # 2. Cache Hit on GetSession
    cached_session = await service.get_session(
        app_name="medquad-agent",
        user_id="doc_smith",
        session_id="session-clinical-001",
    )
    assert cached_session is not None
    assert cached_session.state["specialty"] == "Oncology"

    # 3. Cache Eviction followed by Cache Miss (DB Load)
    await service.cache.evict("medquad-agent", "doc_smith", "session-clinical-001")
    loaded_session = await service.get_session(
        app_name="medquad-agent",
        user_id="doc_smith",
        session_id="session-clinical-001",
    )
    assert loaded_session is not None
    assert loaded_session.id == "session-clinical-001"
    assert loaded_session.state["specialty"] == "Oncology"


@pytest.mark.asyncio
async def test_token_bloat_mitigation_and_long_term_summarization():
    """Verify that when event count exceeds max_short_term_events, older turns are summarized

    into long_term_summary and pruned from active context to prevent token bloat.
    """
    service = PostgresClinicalSessionService(
        db_url="sqlite+aiosqlite:///:memory:",
        max_short_term_events=3,  # Low threshold to trigger summarization quickly
    )
    await service.prepare_tables()

    session = await service.create_session(
        app_name="medquad-agent",
        user_id="oncologist_1",
        session_id="session-summarize-test",
    )

    # Append 5 events sequentially (exceeding threshold of 3)
    turn_texts = [
        ("user", "Query 1: What is Osimertinib indication for PT-10492?"),
        (
            "researcher_subagent",
            "Found [MQ-ONC-001]: Osimertinib is standard 1st-line for EGFR Exon 19 del.",
        ),
        (
            "reviewer_subagent",
            "Audit completed: [APPROVED]. Lab parameters are compatible.",
        ),
        (
            "user",
            "Query 2: Check cardiac safety and QT prolongation parameters.",
        ),
        (
            "researcher_subagent",
            "Found [MQ-CARD-001]: Monitor QTc intervals periodically with baseline ECG.",
        ),
    ]

    for author, text in turn_texts:
        event = Event(
            author=author,
            content=types.Content(
                role="user" if author == "user" else "assistant",
                parts=[types.Part.from_text(text=text)],
            ),
        )
        await service.append_event(session, event)

    # Verify token bloat prevention: active short-term events window is capped at max_short_term_events (3)
    assert len(session.events) <= 3

    # Verify long-term summary contains pruned older context
    assert "long_term_summary" in session.state
    summary = session.state["long_term_summary"]
    assert len(summary) > 0
    assert "MQ-ONC-001" in summary or "PT-10492" in summary
    assert session.state.get("summary_turn_count", 0) >= 1

    # Verify database persistence across reload
    await service.cache.evict("medquad-agent", "oncologist_1", "session-summarize-test")
    reloaded = await service.get_session(
        app_name="medquad-agent",
        user_id="oncologist_1",
        session_id="session-summarize-test",
    )
    assert reloaded is not None
    assert reloaded.state["long_term_summary"] == summary
    assert len(reloaded.events) <= 3
