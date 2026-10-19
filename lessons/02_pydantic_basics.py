"""Run from the project root: .venv/bin/python -m lessons.02_pydantic_basics

No AI call: these are hand-written examples of possible future model output.
"""
from pydantic import ValidationError
from src.schemas import IssueSummary

# Step 1: A normal Python dictionary does not enforce our business rules.
example_response = {
    'product': 'Cloudveil Moisturizer',
    'issue': 'Customers report leaking jars.',
    'evidence_ids': ['DEMO-0-3-000', 'DEMO-0-3-004'],
    'recommended_action': 'Request a packaging quality review.',
}

# Step 2: Check it against the schema (our required fields and types).
summary = IssueSummary.model_validate(example_response)
print('1. ACCEPTED: all required fields have the expected structure.')
print(summary.model_dump_json(indent=2))

# Step 3: An empty evidence list violates our minimum-length rule.
bad_response = {**example_response, 'evidence_ids': []}
try:
    IssueSummary.model_validate(bad_response)
except ValidationError as error:
    print('\n2. REJECTED: the evidence list is empty.')
    for problem in error.errors():
        print(f"   {problem['loc'][0]}: {problem['msg']}")

# Step 4: A made-up ID is still a string, so structure validation accepts it.
# We need a separate database/retrieval check to establish that citations exist.
unverified = IssueSummary.model_validate(
    {**example_response, 'evidence_ids': ['MADE-UP-ID']}
)
print('\n3. LIMITATION: a made-up ID passes structure validation.')
print('   Accepted ID:', unverified.evidence_ids[0])
print('   Later we must check IDs against retrieved reviews and check claim support.')
