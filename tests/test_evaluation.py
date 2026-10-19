import unittest
from src.evaluation import score_retrieval, aggregate


class EvaluationTests(unittest.TestCase):
    def test_known_precision_recall_and_rank(self):
        score = score_retrieval(['x', 'a', 'b'], ['a', 'b', 'c', 'd'], ['x', 'a', 'b', 'c', 'd'], 3)
        self.assertAlmostEqual(score['precision_at_k'], 2/3)
        self.assertEqual(score['recall_at_k'], 0.5)
        self.assertEqual(score['reciprocal_rank_at_k'], 0.5)

    def test_duplicates_do_not_inflate_metrics(self):
        score = score_retrieval(['a', 'a', 'outside'], ['a', 'b'], ['a', 'b'], 3)
        self.assertEqual(score['recall_at_k'], 0.5)
        self.assertEqual(score['duplicate_count'], 1)
        self.assertEqual(score['out_of_scope_ids'], ['outside'])

    def test_no_evidence_separate_from_answerable(self):
        supported = score_retrieval(['a'], ['a'], ['a'], 3)
        unsupported = score_retrieval(['a'], [], ['a'], 3)
        self.assertIsNone(unsupported['recall_at_k'])
        self.assertTrue(unsupported['unsupported_return'])
        summary = aggregate([{'metrics': supported}, {'metrics': unsupported}])
        self.assertEqual(summary['macro_recall_at_k'], 1)
        self.assertEqual(summary['unsupported_return_rate'], 1)
        self.assertFalse(score_retrieval([], [], ['a'], 3)['unsupported_return'])

    def test_invalid_labels_and_k(self):
        with self.assertRaises(ValueError):
            score_retrieval([], ['outside'], ['a'], 3)
        with self.assertRaises(ValueError):
            score_retrieval([], [], [], 0)
