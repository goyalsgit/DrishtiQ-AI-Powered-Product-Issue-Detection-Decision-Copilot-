# DrishtiQ — Customer Feedback Intelligence

A local application for investigating customer-reported product problems using
SQL metrics, review evidence and Gemini-generated draft explanations.

The dashboard connects persistent SQLite storage, local MiniLM/Chroma retrieval,
and a bounded Gemini/LangGraph tool loop. Open **Ask reviews** to enter a
question, then inspect the report, original evidence and SQL results. Connection
settings are in the sidebar. The learning CLI also supports plain chat and fixed RAG.

See [Lesson 10: chat, RAG, and agents](docs/learning/10-agent-lab.md) for commands
and a step-by-step explanation. The offline mode executes real retrieval using
explicitly scripted model messages; it does not generate a Gemini answer.
The user has verified the live metrics agent locally.
[Lesson 11](docs/learning/11-period-comparison.md) adds SQL-backed period
comparison and scoped evidence search for both months; that extension has
offline tests and a scripted integration run, with live behavior still to check.
See the [remaining implementation sequence](docs/learning/project-next-steps.md).

## Start

The current workspace is already installed and has the model cached. Run:

```bash
.venv-agent/bin/python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://127.0.0.1:8501. For a fresh installation:

```bash
python3.12 -m venv .venv-agent
.venv-agent/bin/python -m pip install -r requirements-dashboard.txt
.venv-agent/bin/python -m lessons.04_semantic_search
.venv-agent/bin/python -m scripts.import_public_reviews
```

The AI dashboard uses Python 3.12 in `.venv-agent`. The older `.venv` remains
available for the original Python 3.9 lessons.

### Ask a question in the dashboard

1. Select a saved dataset, product, analysis month, and optional baseline in the sidebar.
2. Expand **Gemini connection** in the sidebar and enter your key in the
   password field. It stays in the Streamlit session; it is not written to a file
   or copied into reports. An existing server environment key also works.
3. For a new dataset, open **Data & trends → Search index maintenance** and click
   **Prepare review search** once.
4. Enter your question and click **Investigate reviews**. This sends the question
   and requested evidence to Gemini. Regular widget reruns do not call Gemini.
5. Read the draft, expand the original reviews and database results, and inspect
   **Technical details: tools and usage** when needed. Download the report if needed.

Changing the selected scope clears the previous report from the view. Each
submission starts a fresh investigation; reports are not persisted across server
restarts. The final narrative now uses a Pydantic schema through Gemini native JSON
output. Review IDs, periods and exact quotes are checked; failed reports are
withheld. SQL statistics are attached by Python. Semantic claim support and
arbitrary numerical claims in narrative remain unverified, and reports are not
part of the persistent approval workflow. Previous rule-based decisions remain readable under
**Data & trends**; they are separate from AI reports. Review dependencies, runtime,
authentication and operations before production deployment.

## Real data

Source: [McAuley Lab Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
[dataset project and citation](https://amazon-reviews-2023.github.io/).

The importer streams the first 50,000 All Beauty records and retains the eight
most frequent parent product IDs in that prefix, regardless of rating. The current
sample contains 932 unique reviews. This is a non-random, incomplete historical
sample, not a representative measure of customer sentiment or brand performance.
The busiest product-month has only 11 reviews; the default 20-review comparison
rule therefore flags insufficient samples. Product IDs are preserved; product names
are not invented. This rule is not a statistical significance test.

`data/amazon_beauty_provenance.json` records the upstream revision, URL, timestamp,
sampling method, CSV hash, and dataset ID. User IDs are not stored. Downloaded
review text is excluded from Git; the reproducible importer and provenance remain.
Consult the upstream source terms before redistributing source data.

## Use the application

The three tabs have distinct purposes:

- **Ask reviews:** enter a question; read the draft and supporting original reviews.
- **Data & trends:** inspect SQL trends and paginated reviews; prepare the search
  index or optionally import a new dataset.
- **Project & quality:** understand the components, measured tests, limitations
  and remaining work.

The default review store is `data/reviews.sqlite`, a persistent database file.
CSV is only an ingestion format; normal dashboard use does not read the original
CSV or require another upload. Set `DRISHTIQ_DB_PATH` to use another compatible
SQLite file. This setting does not implement a PostgreSQL/MySQL connection.
See the [implementation plan](docs/learning/project-next-steps.md) for server-database
integration and the [plain-language project review](docs/project-review.md).

For the public sample, product `B0107QYW14`, month `2016-04` has the most reviews
in a single product-month (11), useful for exploring search.

## Architecture and file map

```text
Import → validation → immutable SQLite dataset snapshot
                         └─ review passages → MiniLM → persistent Chroma index
Question + selected product/month → Gemini requests a tool
  LangGraph executes scoped LangChain tools:
    metrics → fixed, parameterized SQL → exact rating statistics
    search  → MiniLM query vector → filtered Chroma → original SQLite reviews
  Tool results → Gemini → draft explanation + cited reviews → Streamlit
```

- `src/database.py`: dataset identity, transactional imports, persistent reviews,
  composite primary key, scope index, parameterized SQL metrics.
- `src/embeddings.py`: local model, token-based passages, batched indexing,
  dataset-specific collections, readiness marker, filtered search, deduplication.
- `src/analytics.py`: input validation, keyword topics/search, decision persistence.
  Its old in-memory metrics helper remains for Lesson 1-era tests; the dashboard
  now uses `ReviewStore.metrics` against persistent storage.
- `src/schemas.py` and `src/recommendations.py`: structure and citation-membership checks.
- `app.py`: three-tab dashboard, selection, SQL trends, paginated reviews and imports.
- `src/dashboard_ai.py`: connection settings, question form and investigation display.
- `src/investigation.py`: runs the graph and collects draft, evidence and usage.
- `src/reporting.py`: typed final narrative, native structured generation and
  fail-closed source checks. One extra model call after evidence collection;
  no automatic report-repair loop. Original SQL results provide displayed statistics.
- `src/agent.py`, `src/tools.py`, `src/llm.py`: bounded graph, scoped tools and Gemini.
- `src/dashboard_project.py`, `src/project_status.py`: measured quality display and
  source-linked verification status.
- `scripts/import_public_reviews.py`: bounded public-data importer and provenance.
- `lessons/06_persistent_sql.py`: walkthrough of real SQL calculations.

## Data contracts and safeguards

CSV columns: `review_id,product,date,rating,text`. Use a stable product identifier
in `product`, ISO dates (`YYYY-MM-DD`), unique review IDs, integer ratings 1–5.
Missing required fields/invalid ratings or dates are excluded and counted. Exact
duplicates are removed; conflicting review IDs reject the import. IDs read from
CSV remain strings, including leading zeros.

Validated content determines dataset identity, independent of row order or filename.
Reimporting identical content is idempotent. Changes create a new snapshot, preserving
old decisions and preventing stale review embeddings from mixing with new data.
Old snapshots are retained; retention/deletion administration is not implemented.

Long reviews are split into overlapping token passages, with original review IDs
retained. Review counts always come from SQLite, never passage counts. Model/index
configuration is included in collection identity. Interrupted index builds are not
marked ready and may be rerun. Search returns unique reviews and reloads their full
source text from SQLite. Similarity scores are not calibrated confidence values.

The main dashboard shows low-rating share, not a count of specific complaints.
Legacy keyword tags can miss negation/context and are not validated sentiment labels;
they are no longer the main dashboard investigation. Reviews describe reported
problems and may suggest possible causes; they cannot prove manufacturing or
operational root causes. Search does not yet abstain based on an evaluated relevance threshold.

Legacy decisions store dataset identity, product/month/baseline, evidence, reviewer, and time.
The first decision for an exact proposal is preserved. History is filtered by dataset.
Reviewer names are self-entered; this is not authenticated identity. No external action
is executed. Citation existence does not establish that a claim is supported.

## Validation

```bash
.venv-agent/bin/python -m scripts.verify_project
.venv-agent/bin/python -m scripts.check_semantic
.venv-agent/bin/python -m scripts.evaluate_retrieval --k 3 --output reports/retrieval_current.json
```

Unit checks cover calculations, invalid data, duplicate/conflicting IDs, persistence,
SQL scope/injection input, snapshot identity, approval idempotency, and citations.
The semantic integration check uses the real local model with isolated temporary
stores: long-review chunking, repeated indexing, product/month/dataset isolation,
source lookup, and review deduplication. These are functional checks, not a retrieval
benchmark. Dashboard startup and semantic search have also been exercised through
Streamlit AppTest. The latest saved test run is in
`reports/project_verification.json`; the dashboard checks its source fingerprint
before displaying it as current. The 2026-10-21 run passed 56 tests without skips.
Model calls in automated tests are scripted; no live Gemini accuracy is implied.
See the [code/dashboard audit and repository comparison](docs/audit-2026-10-21.md)
for live-check limitations and implementation priorities.

## Next milestones

1. Evaluate structured-report claim support, narrative numerical accuracy and live reliability.
2. Human-reviewed, held-out retrieval evaluation and evidence-sufficiency/abstention.
3. Persistent investigations and LangGraph interrupt/resume approval.
4. Read-only PostgreSQL/MySQL integration and incremental ingestion/index refresh.
5. Deployment hardening: supported runtime, dependency review, authentication,
   data access boundaries, backups, monitoring, retention, and deployment tests.

## Honest resume wording today

**DrishtiQ | Python, SQL, SQLite, Streamlit, Gemini, LangChain, LangGraph, MiniLM, Chroma**

- Built a review analytics application with reproducible public-data ingestion,
  persistent SQL metrics, and dataset-scoped semantic retrieval using local embeddings.
- Added passage indexing, product/date filters, source-linked evidence, and a bounded
  agent that requests scoped tools before drafting review analyses.

Do not claim production readiness, measured model accuracy, or business impact.
The user's live metrics smoke check does not establish general answer quality.
LangGraph integration exists in the CLI and dashboard question flow, with
offline routing and Streamlit interaction tests. Persistent approval for AI
reports is still pending.

## Retrieval evaluation — first measured baseline

```bash
.venv-agent/bin/python -m scripts.evaluate_retrieval --k 3 --output reports/retrieval_current.json
.venv-agent/bin/python -m lessons.07_retrieval_evaluation
.venv-agent/bin/python -m lessons.07_retrieval_evaluation --case Q21
```

`tests/retrieval_cases.json` contains 24 development questions: 20 with evidence
and 4 with no evidence, across four completely inspected product-month scopes
(38 reviews). Labels were prepared by the assistant before retrieval was run;
independent human review is pending. These are not held-out evaluation results.

`src/evaluation.py` calculates precision@k, recall@k, reciprocal rank, and integrity
checks. Precision divides by the requested k, even if fewer records are returned.
No-evidence cases are excluded from those averages and separately measured for
unsupported returns. `scripts/evaluate_retrieval.py` compares existing keyword
and semantic search, validates that labels belong to the exact source snapshot,
and writes JSON plus a readable Markdown report under `reports/`.

The initial k=3 run measured semantic macro recall 86.7% vs keyword 81.7%, and
precision 43.3% vs 41.7%. Keyword MRR was higher (0.817 vs 0.792). Both returned
results on all four no-evidence cases. This is not overall RAG accuracy and does
not establish that semantic search is universally better. Many questions have
only one relevant review, so requesting three results necessarily admits extras
unless the retriever abstains or returns fewer. There were no scope violations
or duplicate results in this run.

Source inspection found jewelry reviews in All Beauty. The category name is
not a verified skincare-only label. Cases deliberately include those records
rather than silently relabelling them.

Next: independently review labels, add separate held-out cases, and evaluate
an evidence-sufficiency/abstention mechanism before treating generated answers as reliable.
Do not tune a cutoff on the four negative questions and claim validated quality.
The report records case hash, dataset identity, model name, index configuration,
library versions, and per-query warm latency. It is not a load benchmark.
