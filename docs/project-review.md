# DrishtiQ: what works, what it means, what is next

Review date: 2026-10-21. See the [extended audit](audit-2026-10-21.md).

## Simple explanation

Yeh project product/support team ko customer reviews samajhne mein help karta hai.
Analyst product aur month select karke question poochta hai. System database se
rating statistics nikalta hai, relevant reviews dhoondhta hai, aur Gemini unke
basis par explanation ka draft banata hai. Analyst original reviews check kar
sakta hai before deciding what to investigate.

For example, “Customers product picture se disappointed kyun hain?” may retrieve
reviews about a stone's colour differing from the photo. That supports a
**reported appearance mismatch**. It does not prove that a photo editor changed
saturation, a supplier changed the stone, or a factory caused the problem.
Those are hypotheses requiring listing images, inventory or quality-control data.

## Dashboard

1. **Ask reviews:** question, draft answer, evidence and database numbers.
2. **Data & trends:** saved review count, monthly trend, original reviews in pages,
   optional import and search-index preparation.
3. **Project & quality:** how components connect, measured tests and remaining work.

Only three headline metrics remain: review count, average rating and low-rating
share. Low-rating share means the percentage of 1–2 star reviews. It is not the
percentage mentioning a particular defect.

The decorative Issue Radar, keyword complaint summaries, duplicate controls and
new legacy-decision entry were removed from the main experience. Existing saved
legacy decisions remain readable. Technical traces are collapsed, not discarded.

Enter a Gemini key in the sidebar's **Gemini connection** password field, or set
`GOOGLE_API_KEY` in the environment that starts Streamlit. The sidebar key is held
in the Streamlit session and omitted from saved/downloaded reports. There is no
automatic `.env` loading. Do not put a key in source code.

## Database versus CSV

The current application reads `data/reviews.sqlite`. This is a persistent file,
not an in-memory database. Closing Streamlit does not erase reviews.

The inspected workspace contains **932 reviews across 8 products in one dataset**.
SQLite's integrity check returned `ok`, and the foreign-key check returned no
violations. These reviews are a historical public sample, not a connected
company's live production reviews.

CSV is an optional transport/import format:

```text
CSV import once → validation → SQLite rows saved
Later dashboard questions → SQLite + Chroma → Gemini draft
```

The normal dashboard no longer loads the entire review table or a source CSV to
populate selectors. SQL returns grouped product/month statistics and pages of
up to 25 original reviews.

For a company's PostgreSQL/MySQL system, we need a connection, schema mapping
and synchronization process. We cannot connect simply by changing the SQLite
filename. A good first integration is a read-only, versioned snapshot of the
company reviews; incremental updates follow once IDs, updates and deletions are
handled. We do not yet have that connection.

## What actually happens when you ask

1. **Streamlit** sends your question and selected product/month into the investigation.
2. **Gemini** sees tool descriptions and can request metrics, evidence search or
   comparison. It produces a tool name and arguments; Python executes the request.
3. **LangChain** supplies the model/tool interfaces and structured messages.
4. **LangGraph** routes between the model and tools, enforcing call limits.
5. **SQL tools** use fixed, parameterized queries for exact counts/averages.
   Gemini does not generate arbitrary SQL. Fixed SQL is a deliberate scope boundary.
6. **Search tool** embeds the search text with local MiniLM, searches Chroma within
   the selected scope, and fetches original reviews by ID from SQLite.
7. **Tool results** become messages returned to Gemini. After evidence collection,
   a separate native JSON call creates the typed narrative. Pydantic checks its
   fields; Python checks review IDs, periods and quoted text. SQL supplies displayed
   statistics directly. Gemini does not calculate vector similarity.
8. **Dashboard** shows the checked draft, original evidence, SQL results and optional
   trace. Failed reports are withheld. Matching quotations do not prove interpretations.

RAG is implemented: retrieve evidence, provide it to the model, generate an answer.
The agent adds model-selected tool calls around that process. Semantic grounding
evaluation and persistent human approval are not finished.

## What we measured

| Check | Observed result | Meaning |
|---|---|---|
| Offline automated suite | 56 passed, 0 failed/errors/skips | Checked program behavior works on tested cases |
| Database checks | Integrity `ok`; no foreign-key violations | Local SQLite file passed these consistency checks |
| Semantic precision@3 | 43.3% | Average relevant result slots among the three requested |
| Semantic recall@3 | 86.7% | Average fraction of labelled relevant reviews retrieved |
| Keyword precision/recall@3 | 41.7% / 81.7% | Comparison baseline on the same questions |
| Semantic no-evidence cases | Returned results for 4 out of 4 | Retriever does not yet reliably say “no relevant evidence” |
| Retrieval scope violations | 0 | No wrong dataset/product/month results in this evaluation |
| Semantic precision/recall@6 | 24.2% / 95.8% | Matches the agent's six-result retrieval limit; direct queries only |

The suite covers persistence, validation, SQL metrics and scope, comparison,
tool argument handling, bounded routing, session credentials, submission, error
handling and stale-report clearing. Automated model responses are scripted.
The dashboard was also opened and visually inspected in Chrome.

The retrieval benchmark uses 24 development questions: 20 answerable and 4 with
no evidence. Labels were prepared by the assistant; independent human review is
pending. This is not a held-out evaluation, live Gemini accuracy score or a
production load test. Returning an irrelevant review is a retrieval defect;
it does not by itself establish that the generated answer hallucinated.

Reports: [automated checks](../reports/project_verification.json),
[retrieval diagnostic](../reports/retrieval_current.md).

Do not say “our AI is 86.7% accurate.” Say:
“On a small development benchmark, semantic search retrieved 86.7% of labelled
relevant reviews on average. Precision and unanswerable-query handling still
need improvement.”

## Priority order

1. Evaluate structured report meaning and live reliability; measure false rejection
   and unsupported claims. Structural and source-reference checks are implemented.
2. Improve retrieval relevance and no-evidence handling; independently review
   labels and test unseen questions. Evaluate generated claims separately.
3. Persist investigations and implement human approval/resume.
4. Add read-only PostgreSQL/MySQL integration and reliable index synchronization.
5. Finish end-to-end evaluation, access control and operational deployment checks.

See [completion criteria and file map](learning/project-next-steps.md).

Interview explanation:
“I built a customer-review investigation assistant. It uses fixed SQL for rating
statistics and MiniLM/Chroma for semantic evidence retrieval. Gemini requests
scoped tools through LangChain, while LangGraph runs a bounded tool loop.
The UI shows the evidence behind each draft. I test program behavior and retrieval
quality separately, and distinguish reported problems from possible root causes.”
