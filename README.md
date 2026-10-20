# DristiQ — Customer Feedback Intelligence

A local application for importing customer reviews, calculating SQL metrics,
retrieving review evidence by meaning, and recording human decisions.

The dashboard connects persistent SQLite storage and local MiniLM/Chroma
retrieval. A separate learning CLI now implements Gemini chat, fixed RAG,
and a bounded LangGraph tool loop. The dashboard does not yet use these modules.

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
.venv/bin/streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://127.0.0.1:8501. For a fresh installation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-app.txt
.venv/bin/python -m lessons.04_semantic_search
.venv/bin/python -m scripts.import_public_reviews
```

Dependencies are pinned around this workspace's Python 3.9 learning environment.
Migration to a supported newer Python and a reviewed dependency set is required
before treating it as a production deployment. This environment emits a LibreSSL
compatibility warning from urllib3; no TLS verification has been disabled.

## Real data

Source: [McAuley Lab Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
[dataset project and citation](https://amazon-reviews-2023.github.io/).

The importer streams the first 50,000 All Beauty records and retains the eight
most frequent parent product IDs in that prefix, regardless of rating. The current
sample contains 932 unique reviews. This is a non-random, incomplete historical
sample, not a representative measure of customer sentiment or brand performance.
The busiest product-month has only 11 reviews; default 20-review alert guards
therefore withhold alerts. Product IDs are preserved; product names are not invented.

`data/amazon_beauty_provenance.json` records the upstream revision, URL, timestamp,
sampling method, CSV hash, and dataset ID. User IDs are not stored. Downloaded
review text is excluded from Git; the reproducible importer and provenance remain.
Consult the upstream source terms before redistributing source data.

## Use the application

1. Select the saved public dataset, or upload a CSV and click **Validate and save dataset**.
2. Select a product, analysis month, and an earlier baseline month.
3. Inspect metrics. They query the persistent SQLite review table.
4. In **Review evidence**, choose **Semantic** and build the index if requested.
5. Enter a question. Results include original review IDs/text, matched passages,
   and similarity scores, filtered by dataset, product, and month.
6. In **Decisions**, inspect the rule-generated proposal and evidence; record approval/rejection.

For the public sample, product `B0107QYW14`, month `2016-04` has the most reviews
in a single product-month (11), useful for exploring search. The results are not
an evaluated claim about retrieval accuracy.

## Architecture and file map

```text
CSV → validation → immutable SQLite dataset snapshot
                       ├─ fixed SQL → rating metrics
                       └─ token passages → MiniLM → persistent Chroma collection
Question + product/month → query embedding → filtered Chroma search
                         → review IDs → source rows in SQLite → dashboard evidence
Keyword topic rules → proposal → Pydantic + citation check → human → SQLite audit
```

- `src/database.py`: dataset identity, transactional imports, persistent reviews,
  composite primary key, scope index, parameterized SQL metrics.
- `src/embeddings.py`: local model, token-based passages, batched indexing,
  dataset-specific collections, readiness marker, filtered search, deduplication.
- `src/analytics.py`: input validation, keyword topics/search, decision persistence.
  Its old in-memory metrics helper remains for Lesson 1-era tests; the dashboard
  now uses `ReviewStore.metrics` against persistent storage.
- `src/schemas.py` and `src/recommendations.py`: structure and citation-membership checks.
- `app.py`: import forms, saved dataset selection, metrics, search, human decisions.
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

Complaint charts still use keywords within low-rating reviews. These tags can miss
negation/context and are not validated sentiment labels. Spike alerts use sample-size
and percentage-point thresholds, not a significance test. Neither retrieves nor
proves root causes. Search does not yet abstain based on an evaluated relevance threshold.

Decisions store dataset identity, product/month/baseline, evidence, reviewer, and time.
The first decision for an exact proposal is preserved. History is filtered by dataset.
Reviewer names are self-entered; this is not authenticated identity. No external action
is executed. Citation existence does not establish that a claim is supported.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.check_semantic
.venv/bin/python -m lessons.06_persistent_sql
```

Unit checks cover calculations, invalid data, duplicate/conflicting IDs, persistence,
SQL scope/injection input, snapshot identity, approval idempotency, and citations.
The semantic integration check uses the real local model with isolated temporary
stores: long-review chunking, repeated indexing, product/month/dataset isolation,
source lookup, and review deduplication. These are functional checks, not a retrieval
benchmark. Dashboard startup and semantic search have also been exercised through
Streamlit AppTest.

## Next milestones

1. Human-labelled retrieval evaluation: relevant/non-relevant examples, out-of-scope
   questions, retrieval recall/precision, failure review, response latency.
2. Grounded Gemini answers with metrics, review citations, abstention, and output checks.
3. LangGraph routing and persisted interrupt/resume approval.
4. Deployment hardening: supported runtime, dependency review, authentication,
   data access boundaries, backups, monitoring, retention, and deployment tests.

## Honest resume wording today

**DristiQ | Python, SQL, SQLite, Streamlit, Sentence Transformers, Chroma, Pydantic**

- Built a review analytics application with reproducible public-data ingestion,
  persistent SQL metrics, and dataset-scoped semantic retrieval using local embeddings.
- Added passage indexing, product/date filters, source-linked evidence, and validated
  recommendations with persistent human decision history.

Do not claim production readiness, measured model accuracy, or business impact.
The user's live metrics smoke check does not establish general answer quality.
LangGraph integration currently exists in the learning CLI, with offline
routing tests, not in the dashboard approval flow.

## Retrieval evaluation — first measured baseline

```bash
.venv/bin/python -m scripts.evaluate_retrieval --k 3
.venv/bin/python -m lessons.07_retrieval_evaluation
.venv/bin/python -m lessons.07_retrieval_evaluation --case Q21
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
an evidence-sufficiency/abstention mechanism before adding generated answers.
Do not tune a cutoff on the four negative questions and claim validated quality.
The report records case hash, dataset identity, model name, index configuration,
library versions, and per-query warm latency. It is not a load benchmark.
