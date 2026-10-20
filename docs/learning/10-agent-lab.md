# Lesson 10: chat → RAG → an agent

Use the project root as your terminal working directory.
The separate `.venv-agent` uses Python 3.12. The dashboard still uses `.venv`.

Install this lesson on a new machine:

```bash
python3.12 -m venv .venv-agent
.venv-agent/bin/python -m pip install -r requirements-agent-search.txt
```

Semantic modes require saved reviews, the cached MiniLM model, and a ready
Chroma index built in the existing dashboard. This lesson uses the existing
index; it does not import data or automatically build a new one.

## 1. Observe a loop without an API key

```bash
.venv-agent/bin/python -m lessons.10_gemini_agent --offline --semantic --question "damaged packaging"
```

This uses **scripted model messages**, real SQL, real MiniLM embeddings and
real Chroma search. It is an execution demonstration, not evidence that
Gemini can choose the correct tools or interpret reviews.

Read the trace in order:

1. `AVAILABLE TOOL`: tool schema describes the input.
2. `NODE: model`: requests a function by name with JSON arguments.
3. `NODE: tools`: Python executes the function.
4. `TOOL RESULT`: a ToolMessage carries the output and matching call ID.
5. The model receives the growing message history.
6. A response without tool calls ends the graph.

## 2. Plain GenAI: a question, no review access

```bash
.venv-agent/bin/python -m lessons.10_gemini_agent --mode chat --ask-key --question "Explain low-rating share to a beginner."
```

`--ask-key` prompts privately and keeps the key only in this process.
Alternatively, set `GOOGLE_API_KEY` in your terminal and omit `--ask-key`.
Do not paste keys into chat or source files. A variable exported in one terminal
does not automatically become visible to an already running editor/agent.

`src/llm.py` creates `ChatGoogleGenerativeAI`, the LangChain adapter for Gemini.
The default model is `gemini-3.5-flash-lite`; use `--model` or `GEMINI_MODEL`
to select another tool-capable Gemini model available to your account.
Model access, quotas, and live API behavior have to be checked with your key.

`model.invoke(messages)` sends messages to Gemini. It returns an `AIMessage`.
Its content may contain text/content blocks; `usage_metadata` reports token
usage when provided. The lesson prints these without printing credentials.

## 3. RAG: our code chooses retrieval

```bash
.venv-agent/bin/python -m lessons.10_gemini_agent --mode rag --semantic --ask-key --question "What evidence is there of damaged packaging?"
```

Python calls metrics and semantic search first, then prints and sends a JSON
context containing the question, scope, metrics, and actual reviews to Gemini.
The embedding coordinates stay local. The model receives retrieved text.
This is one fixed retrieval sequence followed by one generation call.

## 4. Agent: the model chooses tools

```bash
.venv-agent/bin/python -m lessons.10_gemini_agent --mode agent --semantic --ask-key --question "Check the low-rating statistics and search for damaged packaging. Explain what the evidence supports."
```

The live mode sends the question, tool schemas, and subsequent tool results
to Google. It can make several model calls, subject to your API quota/billing.

The same `src/tools.py` functions are used in RAG and agent modes. What changes
is **who chooses when to call them**.

```text
START → model → tool requests? ── no → END
          ↑          │
          │         yes
          │          ↓
          └────── tools
```

`src/agent.py` defines:

- **State**: messages plus model/tool call counters.
- **Reducer**: `MessagesState` merges new messages into conversation state.
- **Model node**: calls Gemini with tool definitions bound through `bind_tools`.
- **Tools node**: validates inputs through LangChain and invokes Python functions.
- **Conditional edge**: reads the model's `tool_calls` to choose tools or finish.
- **Limits**: at most 4 model turns and 6 requested tool executions per run.
  A model turn may retry once on a provider error. A turn is not necessarily
  a single underlying HTTP request.

The application chooses dataset/product/month. Those are closed over by the
tool factory; the model cannot change them through the exposed schema.
By default the CLI picks the newest saved dataset and its busiest product-month.
Use `--dataset ID --product NAME --month YYYY-MM` to select an explicit scope.
Add `--compare` to compare against the previous calendar month, or
`--baseline YYYY-MM` to choose an earlier month explicitly. See
[Lesson 11: comparing periods](11-period-comparison.md).

## 5. Understand what is and is not verified

The graph, tool routing, scope, argument rejection, error handling and limits
have offline tests. The model's final text is an **unverified draft**.
Prompt instructions to cite evidence are not a claim-checking system.
Low ratings are not complaint labels. Top search hits cannot measure complaint
prevalence or establish causes. An irrelevant nearest match should lead to
insufficient evidence, not a confident explanation.

There is no dashboard integration, persistent conversation checkpoint, human
approval node, or decision write in this lesson. Those are later steps.
Each invocation starts a new conversation.

```bash
.venv-agent/bin/python -m unittest discover -s tests -p 'test_agent.py' -v
```

Exercises:

1. Run a metrics question without `--semantic`: only the metrics tool is available.
2. Enable `--semantic`: inspect the extra tool schema.
3. Compare fixed RAG retrieval with the live model's chosen tool sequence.
4. Ask about an unsupported complaint. Check whether the answer admits uncertainty.
5. Match each tool request ID to its ToolMessage response.

References:

- https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
- https://docs.langchain.com/oss/python/langgraph/quickstart
- https://ai.google.dev/gemini-api/docs/models
