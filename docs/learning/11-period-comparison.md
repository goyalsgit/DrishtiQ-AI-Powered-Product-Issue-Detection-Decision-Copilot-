# Lesson 11: compare two months using SQL-backed tools

## What changed

Previously, the agent could inspect one month. With comparison enabled, it can
call `compare_review_periods()` and search evidence in either configured month.
The app still fixes the dataset and product. The model cannot choose an
arbitrary database scope or lower the minimum sample rule.

## Run against saved reviews, without an API call

```bash
.venv-agent/bin/python -m lessons.10_gemini_agent \
  --offline --compare --semantic \
  --question "poor product quality"
```

The scripted model requests comparison, then current and baseline searches.
SQLite, MiniLM and Chroma are real. The model requests and final text in this
mode are scripted, not Gemini output.

The CLI prints the chosen scope. Without explicit selection it uses the newest
dataset and its busiest product-month. `--compare` chooses the previous
**calendar** month, even if that month has no reviews; missing data is reported,
not replaced with zero percent.

## Run with Gemini

```bash
.venv-agent/bin/python -m lessons.10_gemini_agent \
  --compare --semantic --ask-key \
  --question "Use compare_review_periods. Report the change in low-rating share and sample limitations. Search for poor product quality in both periods, cite relevant reviews, and distinguish observations from possible causes."
```

Omit `--ask-key` if your terminal already has `GOOGLE_API_KEY`.
To choose exact periods, use, for example:

```bash
.venv-agent/bin/python -m lessons.10_gemini_agent \
  --product B0107QYW14 --month 2016-04 --baseline 2016-03 \
  --semantic --ask-key \
  --question "Compare these months and investigate product quality using evidence from both."
```

These product/month values exist in the current saved public sample. Add
`--dataset` with its saved ID if another dataset becomes the default.

## Follow the code

1. `lessons/10_gemini_agent.py`: selects current and baseline months.
2. `src/tools.py`: exposes `compare_review_periods()` to the model.
3. `src/comparison.py`: calls `ReviewStore.metrics()` once for each month.
4. `src/database.py`: runs fixed, parameterized SQL for both calls.
5. `src/comparison.py`: computes current share minus baseline share in Python.
6. `src/agent.py`: puts the returned dictionary into a ToolMessage.
7. Gemini receives that message and may request evidence searches.

No extra graph node is needed for comparison. The existing tools node can
execute the newly registered function by name.

A model-selected evidence request looks like:

```json
{
  "name": "search_review_evidence",
  "args": {"question": "poor product quality", "period": "baseline"}
}
```

`period` is limited to `current` or `baseline`. Python maps it to the selected
month before searching. Baseline requests fail when no baseline is configured.

## Read the numbers correctly

The local scripted run on product B0107QYW14 produced:

| Metric | March 2016 | April 2016 |
|---|---:|---:|
| Saved reviews | 6 | 11 |
| Low ratings | 1 | 3 |
| Low-rating share | 16.67% | 27.27% |

The change is **+10.61 percentage points**, calculated from unrounded shares.
The configured minimum is 20 reviews per period, so status is
`insufficient_sample`. This is a descriptive change in the saved subset.
The minimum rule does not establish statistical significance.

Possible tool statuses:

- `missing_period_data`: one or both periods have no reviews; share change is null.
- `insufficient_sample`: both periods have data but at least one is below the minimum.
- `descriptive_comparison`: both meet the minimum; causation is still unproven.

The tool can establish a change in low-rating share. Top search hits do not
establish that packaging complaints became more prevalent. That requires
consistent complaint labelling across the full selected review populations.

## Verification

```bash
.venv-agent/bin/python -m unittest discover -s tests -v
```

Tests cover period arithmetic, missing months, zero low ratings, sample rules,
dataset/product isolation, tool-result delivery, and rejected scope overrides.
Live answer quality for this new tool still requires a Gemini run.

Next: structured investigation reports and deterministic checks for cited IDs
and reported metrics, followed by evidence-quality evaluation.
