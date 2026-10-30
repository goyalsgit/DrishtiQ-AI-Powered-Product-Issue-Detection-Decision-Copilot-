# DrishtiQ: measured capacity and deployment

Measured on 22 October 2026. This document distinguishes observed backend performance, AI answer quality, and proposed deployment settings.

## What the numbers mean

- **Real data:** the application currently contains 932 distinct public reviews. This is the saved sample size, not a software limit.
- **Scale fixture:** the benchmark resamples those texts into larger datasets with new IDs. It runs real embedding computation, Chroma indexing, SQL aggregates, and semantic searches. It does not create additional independent customer evidence.
- **Concurrency:** a worker repeatedly performs one SQL aggregate plus one semantic search. Twenty workers means twenty concurrent backend tasks, not twenty verified browser users receiving Gemini answers.
- **p95 latency:** 95% of measured requests finished within this time. Throughput is completed requests per second.

## Benchmark method

Use `scripts/benchmark_capacity.py` with the Python 3.12 agent environment. The public sample and downloaded model must already exist.

```bash
.venv-agent/bin/python -B -m scripts.benchmark_capacity --rows 10000 --requests 240
.venv-agent/bin/python -B -m scripts.benchmark_capacity --rows 50000 --requests 240
```

Run these sequentially. Each uses temporary SQLite and Chroma databases and removes them afterward; the application's saved reviews are unchanged. Reports remain in `reports/capacity_<rows>.json`.

Machine: Apple M4, 10 logical CPU cores, 16 GiB physical RAM, macOS, Python 3.12.13. Embeddings run on CPU, with four PyTorch threads. Packages: sentence-transformers 3.4.1, PyTorch 2.6.0, Chroma 0.6.3. Model: `sentence-transformers/all-MiniLM-L6-v2`.

Each dataset puts 10% of records in one busy product/month. A quarter of queries target that scope; the rest target smaller scopes. Each of four concurrency levels runs 240 requests. Results must contain six distinct IDs within the requested scope, and SQL counts must match the fixture. These checks do not judge semantic relevance.

Limitations: one trial per level, warm queries, repeated texts and questions, no browser/HTTP overhead, no Gemini calls, no writes during search, no arrival queue, no cold-start timing in query latency. Peak process memory includes benchmark setup, model, index, and query work; it is not total server memory. Disk measurements exclude model files, dependencies, and backups. These tests establish observed operation at a tested scale, not maximum capacity or a production SLA.

## Results

| Metric | 10,000 synthetic reviews | 50,000 synthetic reviews |
| --- | ---: | ---: |
| Indexed passages | 10,197 | 50,988 |
| SQL import time | 0.435 s | 2.143 s |
| Model load time | 2.638 s | 1.920 s |
| Embedding + index build time | 46.350 s | 253.099 s |
| Indexing throughput | 215.75 reviews/s | 197.55 reviews/s |
| Peak benchmark process memory | 737.2 MiB | 778.9 MiB |
| SQLite disk | 4.95 MiB | 24.92 MiB |
| Chroma disk | 36.31 MiB | 171.88 MiB |
| Completed requests / attempted | 960 / 960 | 960 / 960 |
| Observed errors | 0 | 0 |

The largest run has 217 populated product/month scopes: one contains 5,000 reviews and the others contain 208–209 each. Searches are filtered by product and month; this is not an unfiltered search workload or a test where all 50,000 reviews belong to one product/month.

| Workers | 10k: p95 | 10k: requests/s | 50k: median | 50k: p95 | 50k: requests/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 22.31 ms | 51.17 | 48.90 ms | 58.56 ms | 19.56 |
| 5 | 351.66 ms | 17.01 | 710.85 ms | 831.61 ms | 7.13 |
| 10 | 1,792.20 ms | 9.14 | 2,154.13 ms | 6,197.26 ms | 3.91 |
| 20 | 3,283.00 ms | 10.04 | 4,245.61 ms | 12,688.73 ms | 4.06 |

All values describe one SQL aggregate plus semantic retrieval, excluding Gemini and browser work. At 50k/20 workers, the SQL aggregate alone had p95 14.61 ms; the remaining request path needs profiling. Zero observed failures across these finite tests does not imply zero production failures.

Raw evidence: [10k report](../reports/capacity_10000.json), [50k report](../reports/capacity_50000.json).

An honest portfolio statement:

> Built a persistent SQL and vector-search review analytics application; benchmarked 50,000 synthetic review records and 50,988 indexed passages on an Apple M4, completing 960 backend requests across 1–20 concurrent workers without observed errors. Measured p95 search-plus-SQL latency of 59 ms at one worker and 12.69 s at twenty, identifying concurrency as the next optimization target.

Do not substitute “50,000 unique real customer reviews,” “supports 20 AI users,” or “production scale” for these measured claims. No maximum review capacity has been established.

## What to optimize next

The 10,000-record run shows lower throughput at higher concurrency. More threads alone do not make this implementation faster. Profile before assigning a precise cause.

1. Add a bounded queue around CPU-heavy search. Measure queue wait separately from execution time.
2. Profile `ReviewSearch.search`: it currently loads every review in the selected product/month into a pandas dataframe on each query. Fetching only retrieved IDs with dataset/product/month checks is a candidate improvement.
3. Compare identical workloads before and after each change. Record p95, throughput, memory, errors, and retrieval quality; do not trade correctness for a better latency number.
4. Run a longer load test on the actual deployment, including uneven arrivals and a large busy product, then test ingestion and updates separately.

These are proposed improvements, not completed optimizations. Existing code already reuses the embedding model, persists completed vector indexes, batches index writes, and pages dashboard tables. No percentage speedup is claimed without a comparable baseline.

## How many users can receive AI answers?

The backend benchmark cannot answer this alone. A full request includes browser/server handling, one or more Gemini calls, retrieval, and report validation. The agent permits up to four evidence-loop model calls plus an optional final structured-report call; retries can consume further provider requests.

Approximate steady-state ceiling:

```text
answers/minute <= minimum of:
  project RPM / measured provider calls per answer
  project input TPM / measured total input tokens per answer
  deployed application's measured answers/minute
```

Allow headroom for bursts, retries, and other usage; daily/spend limits may also bind. Gemini limits depend on model and project tier and apply per project, not per API key. Check the actual limits in AI Studio. [Google rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits)

**Illustration only:** 60 RPM divided by four calls per answer gives a ceiling of 15 answers/minute before token limits and headroom. If each active user submits one question every two minutes, 30 such users would consume that entire theoretical rate. This is not a measured capacity claim; bursts could still overload it.

For a defensible user count, test actual browser sessions with realistic question intervals, measure complete-answer p95, failures/timeouts/429s, queue wait, provider calls, tokens, and cost. Streamlit sessions use WebSockets: a load tool that only GETs the homepage does not test the AI flow. Use browser automation or expose and test an equivalent application API.

## AI quality is a separate measurement

`reports/retrieval_agent_k6.json` records semantic Recall@6 of 95.83% on 20 answerable development questions. That means finding labeled relevant reviews within six returned results. It is not 95.83% answer accuracy: candidate pools were only 8–11 reviews, labels still need human review, and all four no-evidence questions also returned results.

Next evaluation: a separate human-reviewed test set with answerable, unsupported, ambiguous, and comparison questions. Measure retrieval recall/precision, claim support, citation correctness, SQL result correctness, and correct abstention independently. Record human agreement and denominators. The current project uses a custom retrieval evaluator; do not claim RAGAS or a completed end-to-end accuracy evaluation.

## Deploy a controlled pilot

This is a proposed runbook, not a completed cloud deployment. Start with one service because the current application uses local SQLite and Chroma. A 4 GiB RAM instance is a starting sizing estimate for a small pilot, not a promise of capacity; benchmark on the chosen CPU and monitor memory.

1. Push the application code to your GitHub repository. Exclude keys, virtual environments, and local database/model files. Create a Render **Python web service**, connect that repository, and select Python 3.12. [Render service setup](https://render.com/docs/web-services)
2. Set the build command to install the complete dashboard dependencies and download the embedding model:

   ```bash
   pip install -r requirements-dashboard.txt && python -m lessons.04_semantic_search
   ```

   The lesson downloads the exact model to `.model_cache` and runs a tiny demonstration. The application later loads that model locally. The base `requirements.txt` alone does not install the complete dashboard/agent stack. Dependency ranges should be locked after validating the Linux build.
3. Attach a persistent disk at `/opt/render/project/src/data` for a native Python service. This preserves SQLite, Chroma, and imported data. Default Render storage is ephemeral; disks are available on paid services and only at runtime. The attached disk also restricts this service to one instance. [Persistent disk behavior and limitations](https://render.com/docs/disks)
4. In server environment settings, set `GOOGLE_API_KEY` and `GEMINI_MODEL` to a model available to your account. Do not put the key in Git or a screenshot. Set the start command:

   ```bash
   python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true --browser.gatherUsageStats=false
   ```

   Render requires listening on `0.0.0.0` and the service port. Set health-check path `/_stcore/health`. [Render port binding](https://render.com/docs/web-services), [Streamlit health-check example](https://docs.streamlit.io/deploy/tutorials/docker)
5. After deployment, use the **running service's Shell** to import the public sample:

   ```bash
   python -m scripts.import_public_reviews
   ```

   Then build the index using the dashboard's index preparation control. Do not import into the persistent disk during the build: that disk is unavailable there. A fresh mount also hides repository files under `data`, so regenerate the sample/provenance through the importer.
6. Verify an SQL count, a semantic query, and a cited AI report. Restart the service and verify that the dataset and index remain available. Establish database-consistent backups and test a restore.
7. Before inviting public traffic, add access control, per-user request limits, a global concurrency limit, and monitoring. These controls are not established by the backend benchmark. Run the complete browser-to-Gemini load test, then publish the measured capacity and hosting configuration.

For multiple application instances, move the authoritative records to a shared database such as PostgreSQL and use a separately hosted vector service; update the code accordingly. Current `DRISHTIQ_DB_PATH` is a SQLite file path, not PostgreSQL support. Keep ingestion/indexing off interactive request paths. Source updates/deletions and index synchronization need an explicit implementation; the existing immutable snapshots do not provide live synchronization.
