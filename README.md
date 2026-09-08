# medquad-agent

AI Clinical and Medical Research Assistant for clinicians and researchers powered by Google ADK (Agent Development Kit).

Implements a decoupled **Supervisor-Worker topology** using a **ReAct execution loop**:

1. **Root Orchestrator Agent (Supervisor)**: Powered by `Gemini 2.5 Flash`. Orchestrates query lifecycle, decomposes clinical inquiries, delegates tasks to workers, and synthesizes the final clinical report.
2. **Researcher Subagent (Worker)**: Powered by `Gemini 2.5 Pro`. Executes deep biomedical research via a structured ReAct loop with:
   - **Vertex AI Search Tool**: Performs semantic retrieval over the MedQuAD corpus with exponential backoff, randomized jitter, and automated fallback to a static medical index.
   - **Mock Clinical DB Tool**: Queries simulated PostgreSQL electronic health records (demographics, ICD-10 conditions, active medications, lab values, vitals, allergies).
3. **Reviewer Subagent (Worker)**: Powered by `Gemini 3.5 Flash`. Acts as an independent clinical quality and safety auditor that evaluates citations, checks for hallucinations and drug contraindications, and computes evidence confidence scores.
4. **Workflow Entry Point Guardrail**: Input sanitization mimicking Google Cloud **Model Armor & DLP** to detect and mask Protected Health Information (PHI) and neutralize adversarial prompt injections before reaching LLM inference.
5. **Asynchronous PostgreSQL Session Service**: Low-latency session management backed by asynchronous PostgreSQL (`asyncpg` / `SQLAlchemy`). Conversational history and short-term context are managed in fast active memory caches, while older turns are condensed into structured clinical long-term summaries persisted in the database to eliminate token bloat.
6. **Model Context Protocol (MCP) on Cloud Run**: Tool logic encapsulated in isolated MCP servers hosted on Cloud Run. Features standard Server-Sent Events (SSE) transport (`/mcp/sse`), dynamic JSON-RPC schema discovery via ADK `McpToolset`, and runtime token resolution from **Google Cloud Secret Manager** using service account credentials.
7. **Pydantic Tool Registration & Resilient Function Calling**: Tool definitions registered natively in ADK with explicit Pydantic type schemas enforcing strict parameter bounds (e.g. query length, top_k bounds, PT-XXXXX / MRN-XXXXXX patterns). Standardized exception wrappers natively intercept and handle HTTP 429 rate limits and HTTP 500 errors with jittered backoff.

## Project Structure

```
medquad-agent/
├── app/
│   ├── agent.py               # Root supervisor & worker subagents definition
│   ├── fast_api_app.py        # FastAPI A2A & Reasoning Engine server (mounts /mcp)
│   ├── guardrails/
│   │   └── model_armor.py     # Input sanitization guardrail (PHI & Injection)
│   ├── mcp/
│   │   ├── server.py          # Isolated Cloud Run MCP server with SSE transport
│   │   ├── client.py          # Dynamic JSON-RPC toolset discovery client
│   │   └── secrets.py         # Google Cloud Secret Manager runtime credential resolver
│   ├── sessions/
│   │   └── postgres_session_service.py # Async PostgreSQL session service & state cache
│   ├── tools/
│   │   ├── schemas.py         # Explicit Pydantic parameter schemas & bound validation
│   │   ├── exception_handlers.py # Resilient function calling (HTTP 429/500 handlers)
│   │   ├── medquad_search.py  # Vertex AI Search + exponential backoff + fallback
│   │   ├── clinical_db.py     # Mock PostgreSQL clinical database tool
│   │   └── static_index.py    # Embedded resilient MedQuAD medical index
│   └── app_utils/             # ADK runner, shared services, and telemetry helpers
├── tests/
│   └── unit/
│       ├── test_clinical_agent.py # Unit tests for supervisor, workers, and guardrails
│       ├── test_session_service.py # Tests for async session cache & summarization
│       └── test_mcp_and_tool_registration.py # Tests for MCP, Secret Manager & Pydantic bounds
├── GEMINI.md                  # AI-assisted development context
├── pyproject.toml             # Python package definition (Python 3.11+, ADK 2.6+)
└── uv.lock                    # Locked dependency graph
```
- **uv**: Python package manager (used for all dependency management in this project) - [Install](https://docs.astral.sh/uv/getting-started/installation/) ([add packages](https://docs.astral.sh/uv/concepts/dependencies/) with `uv add <package>`)
- **agents-cli**: Agents CLI - Install with `uv tool install google-agents-cli`
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)


## Quick Start

Install `agents-cli` and its skills if not already installed:

```bash
uvx google-agents-cli setup
```

Install required packages:

```bash
agents-cli install
```

Test the agent with a local web server:

```bash
agents-cli playground
```

You can also use features from the [ADK](https://adk.dev/) CLI with `uv run adk`.

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `agents-cli install` | Install dependencies using uv                                                         |
| `agents-cli playground` | Launch local development environment                                                  |
| `agents-cli lint`    | Run code quality checks                                                               |
| `agents-cli eval`    | Evaluate agent behavior (generate, grade, analyze, and more — see `agents-cli eval --help`) |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests                                                        |
| `agents-cli deploy`  | Deploy agent to Agent Runtime                                                                |
| `agents-cli publish gemini-enterprise` | Register deployed agent to Gemini Enterprise                    || [A2A Inspector](https://github.com/a2aproject/a2a-inspector) | Launch A2A Protocol Inspector                                                        |

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `agents-cli scaffold enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `agents-cli infra cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `agents-cli scaffold upgrade` | Auto-upgrade to latest version while preserving customizations |

---

## Development

Edit your agent logic in `app/agent.py` and test with `agents-cli playground` - it auto-reloads on save.

## Deployment

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

To add CI/CD and Terraform, run `agents-cli scaffold enhance`.
To set up your production infrastructure, run `agents-cli infra cicd`.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.

## A2A Inspector

This agent supports the [A2A Protocol](https://a2a-protocol.org/). Use the [A2A Inspector](https://github.com/a2aproject/a2a-inspector) to test interoperability.
See the [A2A Inspector docs](https://github.com/a2aproject/a2a-inspector) for details.
