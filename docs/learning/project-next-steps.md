# DrishtiQ: remaining implementation sequence

Complete and explain each stage before moving to the next.

| Stage | Status | Deliverable |
|---|---|---|
| SQL, embeddings, tools and basic model loop | Implemented; user reports live metrics agent works | Lessons 08–10 |
| Compare periods and retrieve evidence for each | Implemented; local scripted integration and unit checks pass | Lesson 11 |
| Structured investigation report | Implemented; offline checks pass; live checks ongoing | Typed narrative, SQL-derived statistics, citation-period and exact-quote checks; failed output withheld |
| Evidence sufficiency | Baseline measured; quality changes pending | Independently reviewed labels, irrelevant-result handling, held-out abstention tests |
| Persistent human approval | Pending | LangGraph checkpointer, interrupt/resume, approved/rejected audit record |
| Dashboard investigation | Implemented; simplified and tested | Three tabs, question input, session API key, readable report, evidence, SQL results, optional trace/download |
| Database-first dashboard | Implemented | SQL catalog/trends, bounded review pages, optional import, no automatic sample insertion |
| Operational database integration | Pending | Read-only PostgreSQL/MySQL adapter, schema mapping, update/deletion and index synchronization |
| Dashboard approval for AI reports | Pending | Connect validated AI reports to the persistent approval workflow |
| End-to-end evaluation and delivery | Pending | Grounding, failures, latency, setup and operating instructions |

Typed output and valid review IDs do not alone prove claims are supported.
Evidence evaluation remains a separate step. Production access controls and
operations must be addressed before exposing real customer data to multiple users.

## Current file responsibilities

- `app.py`: three-tab dashboard, data selection, SQL trends, paginated review rows,
  optional imports and read-only legacy decision history.
- `src/dashboard_ai.py`: question form, session key, progress, report/evidence display.
- `src/investigation.py`: runs the agent and collects display text, tool results,
  evidence, token totals and completion status. It does not retain credentials.
- `src/reporting.py`: report schemas, separate native JSON generation, exact-quote
  and source-reference validation; semantic meaning is not checked.
- `src/llm.py`: creates the Gemini client; accepts an explicit session key or environment key.
- `src/agent.py`: bounded model/tool loop.
- `src/tools.py`: scoped SQL, comparison and semantic-search tools.
- `src/comparison.py`: two-period descriptive calculations and sample status.
- `src/database.py`: original review storage, fixed SQL, aggregate catalog and bounded pages.
- `src/embeddings.py`: MiniLM encoding and Chroma retrieval.
- `src/schemas.py` and `src/recommendations.py`: existing proposal schema and ID
  membership checks; not yet an AI-report fact checker.
- `src/evaluation.py`: retrieval metrics; generated-answer evaluation remains pending.
- `tests/test_dashboard.py`: scripted submission, credential handling, missing
  index, errors, reruns, scope changes and incomplete-run checks.
- `requirements-dashboard.txt`: full Python 3.12 dashboard dependency entrypoint.
- `src/dashboard_project.py`: architecture, measured checks and remaining work shown in the UI.
- `src/project_status.py`: verification report loading and source freshness checks.
- `scripts/verify_project.py`: runs offline tests and saves a reproducible verification report.

## Finish each stage with evidence

1. **Evaluate AI report meaning and live reliability.** Structure, reference/period
   membership and quote checks are implemented; report statistics are copied from
   SQL. Test complete generated answers for claim support, numerical claims in
   prose, false rejection and appropriate insufficient-evidence responses.
   A valid schema and matching quote do not prove that text supports a claim.
2. **Improve evidence quality.** Independently review the existing 24 labels and
   create new held-out questions, including unanswerable questions. Develop
   relevance filtering/reranking on development data, then evaluate the frozen
   method on the held-out set. Report precision, recall, unsupported returns and
   abstention coverage. Do not choose a threshold from four negative examples
   and present it as general accuracy.
3. **Persist reports and approval.** Save evidence, source snapshot, model,
   prompt/version, tool results, status and timestamp. Add LangGraph persistence
   and human review. Verify restart/resume and duplicate submission behavior;
   invalid reports must not become approved.
4. **Connect a real operational database.** Start with a read-only connection and
   map its schema to `review_id, product, date, rating, text`. Preserve existing
   tool contracts and bound SQL parameters. Import a versioned snapshot first,
   then add incremental ingestion with stable IDs, update/deletion handling and
   synchronized embeddings. Test that new reviews appear and edited/deleted
   reviews cannot leave stale search results. Validate behavior against an actual
   PostgreSQL/MySQL integration test database, not SQLite alone.
5. **Evaluate and deploy.** Test complete answers for factual correctness, grounded
   claims, appropriate uncertainty and failed-tool handling. Record end-to-end
   latency and token usage. Add authenticated access, secret management, backups,
   monitoring and deployment checks before using private multi-user data.

SQLite already persists the current reviews on disk. Moving to a database server
is useful for shared operational access and concurrency; it does not automatically
improve retrieval accuracy or make reviews representative.

The [2026-10-21 audit](../audit-2026-10-21.md) compares real repository source,
documents dashboard limitations, and explains when additional libraries/MCP would help.
