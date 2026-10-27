# Autonomous Multi-Agent Research & Decision System

FastAPI service that researches a question, evaluates its sources, checks claims,
and produces a structured report with citations and quality metrics.

## Workflow

1. **Researcher** searches the configured web provider, adds supplied documents,
   and retrieves evidence with BM25 lexical ranking plus optional embedding
   similarity, then reranks candidates using the hybrid score and title relevance.
2. **Analyst** summarizes evidence, assesses source authority/evidence quality,
   and classifies decision impact.
3. **Critic** checks each finding against the retrieved evidence and identifies
   unsupported claims and open questions.
4. **Approval gate** pauses high-impact work until a human approves or rejects it.
5. **Reporter** generates the cited report, filters citations to retrieved URLs,
   and calculates evaluation metrics.

LangGraph checkpoints workflow state in SQLite by thread ID. Reusing the returned
`thread_id` carries up to five prior report summaries into the next research run.
Each agent step logs latency, status, and model token usage. Optional Phoenix
OpenTelemetry instrumentation can export traces.

## Run locally

Use Python 3.11 or newer:

```powershell
cd autonomous_research_system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Set `OPENAI_API_KEY` and, optionally, `TAVILY_API_KEY` in `.env` or the process
environment. With
`SEARCH_PROVIDER=auto`, Tavily is selected when its key is set; otherwise the
DuckDuckGo provider is used. `.env` is a template and is ignored by Git.

```powershell
python -m uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API documentation.

## API

- `GET /health` - liveness check.
- `POST /research` - submit `query`, optional `documents`, `max_results`, an
  optional existing `thread_id`, or `require_approval`.
- `POST /research/{thread_id}/approval` - submit `approved` and an optional
  `reason` to resume a paused high-impact run.

A research response has `status` of `completed`, `pending_approval`, or
`rejected`. Completed reports contain findings, source reliability assessments,
citations, caveats, and evaluation scores for relevance, factuality, source
reliability, citation coverage/validity, and overall response quality.

## Data handling

Supplied documents and retrieved passages are sent to the configured model and
embedding provider. Only use data approved for those services; do not submit
secrets or restricted enterprise data without the organization's data-processing
and retention approval.

## Tests

```powershell
python -m pytest
```

## Deployment notes

The SQLite checkpointer is intended for local development and a single service
instance with a durable data volume. For multi-instance production deployments,
use a shared PostgreSQL-backed LangGraph checkpointer, authenticated
tenant-scoped thread IDs, managed secrets, TLS, and access controls on the
approval endpoint. Automated evaluation scores are quality signals, not proof
of factual correctness; human review remains necessary for consequential use.
