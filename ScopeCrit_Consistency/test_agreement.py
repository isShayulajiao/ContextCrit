"""Offline checks against hand-calculated coincidence/chance expectations."""
import json
import unittest

from agreement import DIMENSIONS, LABELS, SCORES, fleiss_kappa, krippendorff_alpha_ordinal, summarize


def review(score, label="PASS"):
    return {
        "scores": {dim: score for dim in DIMENSIONS},
        "pass_fail_review": label,
        "evidence_error_ids": [],
        "short_rationale": "Synthetic fixture, not real dataset evaluation.",
    }


class OrdinalAlphaTests(unittest.TestCase):
    def test_binary_manual_value(self):
        # n=9, n1=4, n2=5. Do=(2/9)d, De=(5/9)d, alpha=0.6.
        value = krippendorff_alpha_ordinal([[1, 1, 1], [1, 2, 2], [2, 2, 2]])
        self.assertAlmostEqual(value["value"], 0.6)
        self.assertEqual(value["included_ratings"], 9)

    def test_empirical_ordinal_distances_not_interval_distances(self):
        # n1=2,n2=3,n5=4. d12=6.25,d25=12.25,d15=36.
        # Do=37/9, De=945/72, alpha=649/945.
        value = krippendorff_alpha_ordinal([[1, 1, 2], [2, 2, 5], [5, 5, 5]])
        self.assertAlmostEqual(value["value"], 649 / 945)
        self.assertAlmostEqual(value["observed_disagreement"], 37 / 9)
        self.assertAlmostEqual(value["expected_disagreement"], 945 / 72)

    def test_missing_values_are_not_imputed(self):
        value = krippendorff_alpha_ordinal([[1, 1, None], [1, 2], [2, 2, None]])
        # n=6, balanced marginal, Do=d/3, De=3d/5.
        self.assertAlmostEqual(value["value"], 4 / 9)
        self.assertEqual(value["included_ratings"], 6)

    def test_singletons_excluded_from_marginals(self):
        base = [[1, 1, 1], [1, 2, 2], [2, 2, 2]]
        value = krippendorff_alpha_ordinal(base + [[5, None, None], [None, None, None]])
        self.assertAlmostEqual(value["value"], 0.6)
        self.assertEqual(value["category_counts"]["5"], 0)
        self.assertEqual(value["excluded_units_fewer_than_two_ratings"], 2)

    def test_perfect_agreement_with_variation(self):
        self.assertEqual(krippendorff_alpha_ordinal([[1, 1, 1], [5, 5, 5]])["value"], 1)

    def test_constant_undefined(self):
        value = krippendorff_alpha_ordinal([[5, 5, 5], [5, 5, 5]])
        self.assertIsNone(value["value"])
        self.assertIn("zero_expected_disagreement", value["reason"])

    def test_no_pairs_undefined(self):
        value = krippendorff_alpha_ordinal([[None, None], [1, None]])
        self.assertIsNone(value["value"])
        self.assertEqual(value["included_units"], 0)

    def test_strict_score_validation(self):
        for invalid in (True, False, 1.0, 0, 6, "3"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                krippendorff_alpha_ordinal([[invalid, 3]])


class FleissKappaTests(unittest.TestCase):
    def test_manual_value(self):
        # Pbar=7/9, Pe=(4/9)^2+(5/9)^2=41/81 -> kappa=.55.
        value = fleiss_kappa([["PASS"] * 3, ["PASS", "FAIL", "FAIL"], ["FAIL"] * 3])
        self.assertAlmostEqual(value["value"], 0.55)
        self.assertAlmostEqual(value["observed_agreement"], 7 / 9)
        self.assertAlmostEqual(value["expected_agreement"], 41 / 81)

    def test_missing_rater_rows_excluded(self):
        full = [["PASS"] * 3, ["FAIL"] * 3]
        value = fleiss_kappa(full + [["PASS", None, "FAIL"], [None] * 3])
        self.assertEqual(value["value"], 1)
        self.assertEqual(value["included_units"], 2)
        self.assertEqual(value["excluded_units_nonuniform_or_missing_ratings"], 2)

    def test_unexpected_rater_count_excluded(self):
        value = fleiss_kappa([["PASS"] * 4, ["PASS", "PASS", None, "PASS"]])
        self.assertIsNone(value["value"])
        self.assertEqual(value["included_units"], 0)

    def test_constant_and_empty_undefined(self):
        for rows in ([], [[None] * 3], [["PASS"] * 3]):
            self.assertIsNone(fleiss_kappa(rows)["value"])
            self.assertIsNotNone(fleiss_kappa(rows)["reason"])

    def test_negative_kappa_not_clamped(self):
        self.assertAlmostEqual(fleiss_kappa([list(LABELS)])["value"], -0.5)

    def test_labels_nominal(self):
        with self.assertRaises(ValueError):
            fleiss_kappa([[1, 2, 3]])
        for invalid in (True, 1, 0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fleiss_kappa([], expected_raters=invalid)


class ScoreFleissKappaTests(unittest.TestCase):
    def test_all_five_categories_manual_value(self):
        # Each row has two equal scores and one different score: Pbar=1/3.
        # All five category marginals are 3/15: Pe=1/5, kappa=1/6.
        rows = [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5], [5, 5, 1]]
        result = fleiss_kappa(rows, categories=SCORES)
        self.assertAlmostEqual(result["value"], 1 / 6)
        self.assertAlmostEqual(result["observed_agreement"], 1 / 3)
        self.assertAlmostEqual(result["expected_agreement"], 1 / 5)
        self.assertEqual(result["category_proportions"], {str(score): 0.2 for score in SCORES})

    def test_score_kappa_is_nominal_not_ordinal_alpha(self):
        rows = [[1, 1, 1], [1, 2, 2], [2, 2, 2]]
        self.assertAlmostEqual(fleiss_kappa(rows, categories=SCORES)["value"], 0.55)
        self.assertAlmostEqual(krippendorff_alpha_ordinal(rows)["value"], 0.6)
        # Relabeling categories preserves nominal kappa regardless of distance.
        mapped = [[1 if score == 1 else 5 for score in row] for row in rows]
        self.assertAlmostEqual(fleiss_kappa(mapped, categories=SCORES)["value"], 0.55)

    def test_negative_score_kappa_not_clamped(self):
        result = fleiss_kappa([[1, 2, 3], [3, 4, 5]], categories=SCORES)
        self.assertAlmostEqual(result["expected_agreement"], 2 / 9)
        self.assertAlmostEqual(result["value"], -2 / 7)

    def test_missing_scores_require_all_three_judges(self):
        rows = [[1, 1, 1], [5, 5, 5], [2, None, 2], [None, None, None]]
        result = fleiss_kappa(rows, categories=SCORES)
        self.assertEqual(result["value"], 1)
        self.assertEqual(result["included_units"], 2)
        self.assertEqual(result["excluded_units_nonuniform_or_missing_ratings"], 2)
        self.assertEqual(result["category_proportions"]["2"], 0)

    def test_constant_empty_and_all_missing_scores_undefined(self):
        for rows in ([], [[None] * 3], [[4, 4, 4], [4, 4, 4]]):
            with self.subTest(rows=rows):
                result = fleiss_kappa(rows, categories=SCORES)
                self.assertIsNone(result["value"])
                self.assertIsNotNone(result["reason"])
                self.assertEqual(set(result["category_proportions"]), {str(score) for score in SCORES})
                json.dumps(result, allow_nan=False)

    def test_strict_score_and_domain_validation(self):
        for invalid in (True, False, 1.0, 0, 6, "3"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fleiss_kappa([[invalid, 2, 3]], categories=SCORES)
        for invalid_domain in ((True, 2), (1.0, 2), (0, 1), (1, 6), (1, "2"), (1, 1), ()):
            with self.subTest(domain=invalid_domain), self.assertRaises(ValueError):
                fleiss_kappa([], categories=invalid_domain)

    def test_default_label_api_unchanged(self):
        rows = [["PASS"] * 3, ["PASS", "FAIL", "FAIL"], ["FAIL"] * 3]
        self.assertEqual(fleiss_kappa(rows), fleiss_kappa(rows, categories=LABELS))


class ReportTests(unittest.TestCase):
    names = ["qwen", "glm", "grok"]

    def setUp(self):
        self.records = [
            {"sample_id": "s1", "task": "T1", "ratings": {name: review(1) for name in self.names}},
            {"sample_id": "s2", "task": "T2", "ratings": {name: review(5, "FAIL") for name in self.names}},
            {"sample_id": "s3", "task": "T2", "ratings": {"qwen": review(3, "REVIEW"), "glm": None}},
        ]

    def test_json_serializable_with_missing(self):
        result = summarize(self.records, self.names)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["total_samples"], 3)
        self.assertEqual(result["overall"]["complete_all_judge_review_samples"], 2)
        self.assertEqual(result["overall"]["per_judge"]["qwen"]["complete_reviews"], 3)
        self.assertEqual(result["overall"]["per_judge"]["grok"]["missing_or_incomplete_reviews"], 1)
        self.assertEqual(result["by_task"]["T2"]["sample_count"], 2)
        self.assertFalse(result["human_comparison"]["provided"])
        for dim in DIMENSIONS:
            self.assertEqual(result["overall"]["dimensions"][dim]["krippendorff_alpha_ordinal"]["value"], 1)
            kappa = result["overall"]["dimensions"][dim]["fleiss_kappa_nominal"]
            self.assertEqual(kappa["value"], 1)
            self.assertEqual(kappa["included_units"], 2)
            self.assertEqual(kappa["excluded_units_nonuniform_or_missing_ratings"], 1)
            self.assertEqual(kappa["category_proportions"]["3"], 0)
            for task in ("T1", "T2"):
                self.assertIn("fleiss_kappa_nominal", result["by_task"][task]["dimensions"][dim])
            # singleton qwen score is pooled distribution data, but not alpha marginals
            self.assertEqual(result["overall"]["dimensions"][dim]["pooled_scores"]["distribution"]["3"], 1)

    def test_empty_report(self):
        result = summarize([], self.names)
        json.dumps(result, allow_nan=False)
        self.assertIsNone(result["overall"]["labels"]["fleiss_kappa_nominal"]["value"])
        self.assertEqual(result["by_task"], {})

    def test_pair_disagreement_and_bias(self):
        self.records[0]["ratings"]["glm"] = review(4, "REVIEW")
        result = summarize(self.records, self.names)
        pair = result["overall"]["dimensions"]["Accuracy"]["pairwise"][0]
        self.assertEqual(pair["left"], "qwen")
        self.assertEqual(pair["right"], "glm")
        self.assertEqual(pair["paired_samples"], 2)
        self.assertEqual(pair["mean_absolute_difference"], 1.5)
        self.assertEqual(pair["mean_left_minus_right"], -1.5)
        self.assertEqual(pair["difference_at_least_two_rate"], 0.5)

    def test_human_pairing_without_polluting_three_judge_statistics(self):
        before = summarize(self.records, self.names)
        human = {"s1": review(2, "REVIEW"), "s3": review(3), "unknown": review(4)}
        after = summarize(self.records, self.names, human)
        self.assertEqual(before["overall"], after["overall"])
        result = after["human_comparison"]
        self.assertTrue(result["provided"])
        self.assertEqual(result["matched_samples"], 2)
        self.assertEqual(result["unknown_sample_ids_excluded"], ["unknown"])
        self.assertEqual(result["per_judge"]["qwen"]["labels"]["paired_samples"], 2)
        self.assertEqual(result["per_judge"]["grok"]["labels"]["paired_samples"], 1)
        self.assertEqual(result["per_judge"]["qwen"]["dimensions"]["Accuracy"]["mean_left_minus_right"], 0.5)
        json.dumps(after, allow_nan=False)

    def test_cached_reviews_can_be_reanalyzed_without_mutation(self):
        # Reports are derived entirely from already stored reviews; adding kappa
        # changes no score, label, rubric, prompt, or cached record.
        snapshot = json.dumps(self.records, sort_keys=True, allow_nan=False)
        first = summarize(self.records, self.names)
        second = summarize(json.loads(snapshot), self.names)
        self.assertEqual(first, second)
        self.assertEqual(snapshot, json.dumps(self.records, sort_keys=True, allow_nan=False))
        self.assertIn("ignores the distance", first["methodology"]["score_fleiss_kappa"])

    def test_requires_three_distinct_judges(self):
        for names in (["a", "b"], ["a", "a", "b"], ["a", "", "b"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                summarize([], names)


if __name__ == "__main__":
    unittest.main()
