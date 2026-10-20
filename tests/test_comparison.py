import tempfile
import unittest
from pathlib import Path

import pandas as pd
from src.comparison import compare_periods, previous_month
from src.database import ReviewStore


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ReviewStore(Path(self.tmp.name) / "reviews.sqlite")
        rows = []
        # Ten percent -> twenty percent: +10 percentage points, not +10 percent.
        for month, low_count in [("2026-01", 2), ("2026-02", 4)]:
            for i in range(20):
                rows.append({
                    "review_id": f"{month}-{i}", "product": "A",
                    "date": month + "-01", "rating": 1 if i < low_count else 5,
                    "text": "Fictional comparison test review.",
                })
        # These records must not enter product A's aggregates.
        rows.append({"review_id": "other-product", "product": "B",
                     "date": "2026-02-01", "rating": 1, "text": "Other product."})
        self.dataset, _ = self.store.ingest(pd.DataFrame(rows), "comparison-test")
        self.other_dataset, _ = self.store.ingest(pd.DataFrame([{
            "review_id": "other-dataset", "product": "A", "date": "2026-02-01",
            "rating": 1, "text": "Other dataset.",
        }]), "other-dataset")

    def compare(self, **kwargs):
        return compare_periods(self.store, self.dataset, "A",
                               kwargs.pop("current", "2026-02"),
                               kwargs.pop("baseline", "2026-01"), **kwargs)

    def test_exact_change_and_scope(self):
        result = self.compare()
        self.assertEqual(result["current"]["total_reviews"], 20)
        self.assertEqual(result["baseline"]["total_reviews"], 20)
        self.assertEqual(result["current"]["low_rating_share_percent"], 20)
        self.assertEqual(result["baseline"]["low_rating_share_percent"], 10)
        self.assertEqual(result["low_rating_share_change_percentage_points"], 10)
        self.assertEqual(result["status"], "descriptive_comparison")
        self.assertTrue(result["meets_minimum_reviews"])

    def test_small_sample_keeps_descriptive_change_but_warns(self):
        result = self.compare(minimum_reviews=21)
        self.assertEqual(result["low_rating_share_change_percentage_points"], 10)
        self.assertEqual(result["status"], "insufficient_sample")
        self.assertFalse(result["meets_minimum_reviews"])

    def test_missing_baseline_is_not_zero_percent(self):
        result = self.compare(baseline="2025-12")
        self.assertIsNone(result["baseline"]["low_rating_share_percent"])
        self.assertIsNone(result["low_rating_share_change_percentage_points"])
        self.assertEqual(result["status"], "missing_period_data")

    def test_missing_current_is_not_zero_percent(self):
        result = self.compare(current="2026-03")
        self.assertIsNone(result["current"]["low_rating_share_percent"])
        self.assertIsNone(result["low_rating_share_change_percentage_points"])

    def test_zero_low_ratings_is_valid_data(self):
        frame = pd.DataFrame([
            {"review_id": "a", "product": "Z", "date": "2026-01-01", "rating": 5, "text": "Good"},
            {"review_id": "b", "product": "Z", "date": "2026-02-01", "rating": 1, "text": "Bad"},
        ])
        dataset, _ = self.store.ingest(frame, "zero-baseline")
        result = compare_periods(self.store, dataset, "Z", "2026-02", "2026-01")
        self.assertEqual(result["baseline"]["low_rating_share_percent"], 0)
        self.assertEqual(result["low_rating_share_change_percentage_points"], 100)

    def test_rejects_invalid_periods_and_configuration(self):
        for baseline in ["2026-02", "2026-03", "2026-13", "2026-1", ""]:
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                self.compare(baseline=baseline)
        for minimum in [0, -1, True, 1.5]:
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                self.compare(minimum_reviews=minimum)

    def test_previous_calendar_month_including_year_boundary(self):
        self.assertEqual(previous_month("2026-01"), "2025-12")
        self.assertEqual(previous_month("2024-03"), "2024-02")


if __name__ == "__main__":
    unittest.main()
