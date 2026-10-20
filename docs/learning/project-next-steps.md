# DrishtiQ: remaining implementation sequence

Complete and explain each stage before moving to the next.

| Stage | Status | Deliverable |
|---|---|---|
| SQL, embeddings, tools and basic model loop | Implemented; user reports live metrics agent works | Lessons 08–10 |
| Compare periods and retrieve evidence for each | Implemented; local scripted integration and unit checks pass | Lesson 11 |
| Structured investigation report | Next | Typed report, numerical provenance and citation-membership checks |
| Evidence sufficiency | Pending | Labelled evaluation, irrelevant-result handling, abstention tests |
| Persistent human approval | Pending | LangGraph checkpointer, interrupt/resume, approved/rejected audit record |
| Dashboard integration | Pending | Investigation controls, trace, evidence and approval display |
| End-to-end evaluation and delivery | Pending | Grounding, failures, latency, setup and operating instructions |

Typed output and valid review IDs do not alone prove claims are supported.
Evidence evaluation remains a separate step. Production access controls and
operations must be addressed before exposing real customer data to multiple users.
