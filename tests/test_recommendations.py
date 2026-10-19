import unittest
from src.recommendations import validate_recommendation


class RecommendationTests(unittest.TestCase):
    def test_structure_and_selected_evidence(self):
        payload = dict(product='P', issue='Packaging complaint',
                       evidence_ids=['R1'], recommended_action='Investigate')
        self.assertEqual(validate_recommendation(payload, ['R1']).product, 'P')
        for ids in [[], ['invented'], ['R1', 'outside-filter']]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_recommendation(dict(payload, evidence_ids=ids), ['R1'])
