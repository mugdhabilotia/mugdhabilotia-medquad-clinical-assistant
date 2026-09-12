import pytest
from pydantic import ValidationError
from starlette.applications import Starlette

from app.mcp.client import create_remote_mcp_toolset, get_clinical_tools
from app.mcp.secrets import SecretManagerResolver
from app.mcp.server import (
    create_mcp_sse_app,
    mcp_server,
    query_mock_clinical_db,
    search_medquad_corpus,
)
from app.tools.exception_handlers import (
    UpstreamRateLimitError,
    UpstreamServiceError,
    resilient_function_call,
)
from app.tools.schemas import ClinicalDBQueryInput, MedQuADSearchInput


def test_pydantic_parameter_bounds_enforcement():
    """Verify that Pydantic type schemas strictly enforce parameter bounds."""
    # 1. Valid MedQuADSearchInput
    valid_search = MedQuADSearchInput(
        query="Osimertinib EGFR mutation", top_k=5, domain_filter="Oncology"
    )
    assert valid_search.top_k == 5
    assert valid_search.query == "Osimertinib EGFR mutation"

    # 2. Invalid top_k exceeding upper bound (> 10)
    with pytest.raises(ValidationError):
        MedQuADSearchInput(query="Valid query", top_k=25)

    # 3. Invalid top_k below lower bound (< 1)
    with pytest.raises(ValidationError):
        MedQuADSearchInput(query="Valid query", top_k=0)

    # 4. Invalid query length below min_length (< 2)
    with pytest.raises(ValidationError):
        MedQuADSearchInput(query="A", top_k=3)

    # 5. Valid ClinicalDBQueryInput pattern PT-XXXXX or MRN-XXXXXX
    valid_db = ClinicalDBQueryInput(patient_id_or_mrn="PT-10492")
    assert valid_db.patient_id_or_mrn == "PT-10492"

    valid_mrn = ClinicalDBQueryInput(patient_id_or_mrn="MRN-849201")
    assert valid_mrn.patient_id_or_mrn == "MRN-849201"

    # 6. Invalid patient format violates regex
    with pytest.raises(ValidationError):
        ClinicalDBQueryInput(patient_id_or_mrn="INVALID-ID-123")


def test_standardized_exception_wrapper_http_429():
    """Verify that resilient_function_call handles HTTP 429 rate limit exceptions natively."""

    @resilient_function_call(max_retries=1, initial_backoff_sec=0.01)
    def failing_rate_limit_tool(q: str):
        raise UpstreamRateLimitError("Vertex AI Search quota exhausted")

    result = failing_rate_limit_tool(q="test")
    assert result["status"] == "ERROR_RATE_LIMITED"
    assert result["http_code"] == 429
    assert result["retryable"] is True
    assert "quota exhausted" in result["error"]


def test_standardized_exception_wrapper_http_500():
    """Verify that resilient_function_call handles HTTP 500 upstream service errors natively."""

    @resilient_function_call(max_retries=1, initial_backoff_sec=0.01)
    def failing_500_tool():
        raise UpstreamServiceError("Cloud SQL unreachable", status_code=500)

    result = failing_500_tool()
    assert result["status"] == "ERROR_UPSTREAM_SERVICE"
    assert result["http_code"] == 500
    assert result["retryable"] is True
    assert "Cloud SQL unreachable" in result["error"]


def test_standardized_exception_wrapper_validation_error():
    """Verify that resilient_function_call intercepts Pydantic ValidationError cleanly."""

    @resilient_function_call(max_retries=1)
    def bounded_tool(top_k: int):
        MedQuADSearchInput(query="Valid query", top_k=top_k)
        return {"status": "SUCCESS"}

    result = bounded_tool(top_k=50)  # Violates ge=1, le=10
    assert result["status"] == "ERROR_VALIDATION_FAILED"
    assert result["http_code"] == 400
    assert result["retryable"] is False


def test_secret_manager_resolver_caching_and_fallback(monkeypatch):
    """Verify Secret Manager runtime credential retriever with TTL cache and environment fallback."""
    resolver = SecretManagerResolver(ttl_seconds=60)

    # Test environment fallback when Secret Manager is not active
    monkeypatch.setenv("VERTEX_AI_SEARCH_API_KEY", "mock-secret-key-12345")
    secret = resolver.get_secret("VERTEX_AI_SEARCH_API_KEY")
    assert secret == "mock-secret-key-12345"

    # Test TTL cache hit
    monkeypatch.setenv("VERTEX_AI_SEARCH_API_KEY", "changed-key")
    cached_secret = resolver.get_secret("VERTEX_AI_SEARCH_API_KEY")
    # Should still return original cached value because TTL is active
    assert cached_secret == "mock-secret-key-12345"


def test_mcp_server_registration_and_sse_app():
    """Verify MCP Server tool definitions, Pydantic parameter boundaries, and SSE application creation."""
    # 1. Verify tools registered in FastMCP
    tool_names = [t.name for t in mcp_server._tool_manager.list_tools()]
    assert "search_medquad_corpus" in tool_names
    assert "query_mock_clinical_db" in tool_names

    # 2. Verify Starlette SSE application instantiation for Cloud Run
    sse_app = create_mcp_sse_app()
    assert isinstance(sse_app, Starlette)

    # 3. Test direct execution of bounded MCP tools
    search_res = search_medquad_corpus(query="SGLT2 inhibitors heart failure", top_k=2)
    assert search_res["status"] in ("SUCCESS", "FALLBACK_SUCCESS")

    db_res = query_mock_clinical_db(patient_id_or_mrn="PT-10492")
    assert db_res["status"] == "RECORD_FOUND"


def test_adk_mcp_toolset_initialization(monkeypatch):
    """Verify ADK McpToolset dynamic discovery client initialization with SSE connection params."""
    toolset = create_remote_mcp_toolset(
        sse_endpoint_url="https://medquad-mcp-server-uc.a.run.app/sse",
        auth_token="test-bearer-token",
    )
    assert toolset is not None
    assert (
        toolset.connection_params.url == "https://medquad-mcp-server-uc.a.run.app/sse"
    )
    assert (
        toolset.connection_params.headers["Authorization"] == "Bearer test-bearer-token"
    )

    # Test native ADK tools fallback
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.delenv("CLOUD_RUN_MCP_URL", raising=False)
    native_tools = get_clinical_tools()
    assert len(native_tools) == 2
