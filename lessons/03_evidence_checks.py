"""Run: .venv/bin/python -m lessons.03_evidence_checks"""
from src.recommendations import validate_recommendation

# Pretend these two review IDs came from our selected evidence.
available_ids = ['R101', 'R102']
proposal = {
    'product': 'Cloudveil Moisturizer',
    'issue': 'Customers report leaking jars.',
    'evidence_ids': ['R101'],
    'recommended_action': 'Request a packaging quality review.',
}

summary = validate_recommendation(proposal, available_ids)
print('ACCEPTED: cited review belongs to the selected evidence:', summary.evidence_ids)

proposal['evidence_ids'] = ['MADE-UP-ID']
try:
    validate_recommendation(proposal, available_ids)
except ValueError as error:
    print('REJECTED:', error)

print('An existing citation still does not prove that its text supports a claim.')
