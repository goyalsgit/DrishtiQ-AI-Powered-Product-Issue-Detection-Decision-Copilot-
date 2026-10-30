# Retrieval evaluation — development diagnostic

**Assistant-reviewed labels; independent human review pending. Not full RAG accuracy.**

Cases: 24. Requested results per query: 6.

| Method | Precision@k | Recall@k | MRR@k | Unsupported returns | Scope violations |
|---|---:|---:|---:|---:|---:|
| semantic | 24.2% | 95.8% | 0.804 | 100.0% | 0 |
| keyword | 21.7% | 84.2% | 0.817 | 100.0% | 0 |

## Cases to inspect

Missed relevant reviews and unsupported returns:

- semantic, Q01: Which reviews say the color differs from the product picture? — missing=1; unsupported return=None.
- semantic, Q14: Which reviews say the fragrance fades or is weak after drying? — missing=1; unsupported return=None.
- semantic, Q21: Which reviews report a shattered glass bottle on delivery? — missing=0; unsupported return=True.
- semantic, Q22: Which reviews report a battery catching fire? — missing=0; unsupported return=True.
- semantic, Q23: Which reviews prove that the necklace cures eczema? — missing=0; unsupported return=True.
- semantic, Q24: Which reviews report an unauthorized credit card charge? — missing=0; unsupported return=True.
- keyword, Q01: Which reviews say the color differs from the product picture? — missing=1; unsupported return=None.
- keyword, Q05: Which reviews praise the low price or value for money? — missing=1; unsupported return=None.
- keyword, Q09: Which review raises concerns about manufacturing labor conditions? — missing=1; unsupported return=None.
- keyword, Q14: Which reviews say the fragrance fades or is weak after drying? — missing=1; unsupported return=None.
- keyword, Q16: Which review mentions that the product cannot be returned? — missing=1; unsupported return=None.
- keyword, Q21: Which reviews report a shattered glass bottle on delivery? — missing=0; unsupported return=True.
- keyword, Q22: Which reviews report a battery catching fire? — missing=0; unsupported return=True.
- keyword, Q23: Which reviews prove that the necklace cures eczema? — missing=0; unsupported return=True.
- keyword, Q24: Which reviews report an unauthorized credit card charge? — missing=0; unsupported return=True.

## Interpretation limits

- Assistant-reviewed labels, pending independent human review; 24 development cases in four small scopes.
- No held-out evaluation or full RAG answer generation. These are retrieval diagnostics, not an accuracy claim.
- Precision@k divides by k, even if fewer results are returned; answerable cases only.
- No-evidence cases are excluded from precision/recall and reported separately.
- Unsupported return means results for a scope with no labelled evidence; it is not a generated hallucination.
- No abstention threshold is implemented. Single-user warm-query timings exclude model load and indexing.
