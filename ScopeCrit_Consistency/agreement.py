"""Ordinal and nominal agreement without third-party runtime dependencies.

All input rows are units (samples), columns are judges. Missing values are None.
Ordinal alpha follows Krippendorff's coincidence matrix, including the empirical
ordinal distance; squared numerical score distance would be interval alpha.
"""
from collections import Counter
from itertools import combinations
from math import isfinite
from statistics import mean, median

DIMENSIONS = (
    "Accuracy", "Naturalness", "Informativeness",
    "Contextual Fidelity", "Evidence Grounding",
)
LABELS = ("PASS", "REVIEW", "FAIL")
SCORES = (1, 2, 3, 4, 5)


def _score(value):
    if value is None:
        return None
    if type(value) is not int or value not in SCORES:
        raise ValueError("Scores must be strict integers from 1 to 5, or None.")
    return value


def _label(value):
    if value is None:
        return None
    if not isinstance(value, str) or value not in LABELS:
        raise ValueError("Labels must be PASS, REVIEW, FAIL, or None.")
    return value


def _finite(value):
    return float(value) if value is not None and isfinite(value) else None


def krippendorff_alpha_ordinal(matrix):
    """Return alpha plus denominator and exclusion information.

    Units with fewer than two available ratings contribute neither observed
    coincidences nor pooled marginal frequencies. Missing ratings on otherwise
    pairable units are supported, and need not be filled or imputed.
    """
    rows = [[_score(x) for x in row if x is not None] for row in matrix]
    paired = [row for row in rows if len(row) >= 2]
    counts = Counter(x for row in paired for x in row)
    n = sum(counts.values())
    result = {
        "value": None,
        "reason": None,
        "total_units": len(rows),
        "included_units": len(paired),
        "excluded_units_fewer_than_two_ratings": len(rows) - len(paired),
        "included_ratings": n,
        "category_counts": {str(x): counts[x] for x in SCORES},
        "observed_disagreement": None,
        "expected_disagreement": None,
    }
    if not paired:
        result["reason"] = "insufficient_pairable_units"
        return result

    # The difference of ordinal midranks equals the sum of the category
    # frequencies between the two endpoints, minus half each endpoint count.
    distance = {}
    for left in SCORES:
        for right in SCORES:
            low, high = sorted((left, right))
            span = sum(counts[k] for k in SCORES if low <= k <= high)
            distance[left, right] = (span - (counts[left] + counts[right]) / 2) ** 2

    observed_sum = 0.0
    for row in paired:
        unit = Counter(row)
        observed_sum += sum(
            unit[left] * unit[right] * distance[left, right]
            for left in SCORES for right in SCORES if left != right
        ) / (len(row) - 1)
    observed = observed_sum / n
    expected = sum(
        counts[left] * counts[right] * distance[left, right]
        for left in SCORES for right in SCORES if left != right
    ) / (n * (n - 1))
    result["observed_disagreement"] = _finite(observed)
    result["expected_disagreement"] = _finite(expected)
    if expected == 0:
        result["reason"] = "zero_expected_disagreement_all_ratings_same_category"
        return result
    result["value"] = _finite(1 - observed / expected)
    return result


def fleiss_kappa(matrix, expected_raters=3, categories=None):
    """Nominal Fleiss kappa with fixed raters and an explicit category domain.

    categories=None retains PASS/REVIEW/FAIL. Use categories=SCORES for the
    discrete scores 1..5: a disagreement between 1 and 2 has the same weight as
    one between 1 and 5. This is deliberately different from ordinal alpha.
    Rows with missing or an unexpected number of ratings are excluded. The
    evaluator uses expected_raters=3, with the same three named judges.
    """
    if type(expected_raters) is not int or expected_raters < 2:
        raise ValueError("expected_raters must be an integer of at least two.")
    domain = tuple(LABELS if categories is None else categories)
    if not domain:
        raise ValueError("categories must be non-empty.")
    numeric = all(type(category) is int for category in domain)
    textual = all(type(category) is str and category for category in domain)
    if not numeric and not textual:
        raise ValueError("categories must be all strict score integers or all non-empty strings.")
    if numeric:
        for category in domain:
            _score(category)
    if len(set(domain)) != len(domain):
        raise ValueError("categories must be distinct.")

    def validate(value):
        if value is None:
            return None
        if numeric:
            _score(value)
        elif type(value) is not str:
            raise ValueError("Nominal textual ratings must be strings or None.")
        if value not in domain:
            raise ValueError("Rating is outside the specified categories.")
        return value

    rows = [[validate(x) for x in row] for row in matrix]
    complete = [
        row for row in rows
        if len(row) == expected_raters and all(value is not None for value in row)
    ]
    counts = Counter(x for row in complete for x in row)
    result = {
        "value": None,
        "reason": None,
        "total_units": len(rows),
        "included_units": len(complete),
        "excluded_units_nonuniform_or_missing_ratings": len(rows) - len(complete),
        "raters_per_unit": expected_raters,
        "observed_agreement": None,
        "expected_agreement": None,
        "category_proportions": {str(category): None for category in domain},
    }
    if not complete:
        result["reason"] = "no_complete_units"
        return result
    n = len(complete)
    m = expected_raters
    observed = mean(
        sum(c * (c - 1) for c in Counter(row).values()) / (m * (m - 1))
        for row in complete
    )
    proportions = {str(category): counts[category] / (n * m) for category in domain}
    expected = sum(p * p for p in proportions.values())
    result.update({
        "observed_agreement": _finite(observed),
        "expected_agreement": _finite(expected),
        "category_proportions": proportions,
    })
    if abs(1 - expected) < 1e-15:
        result["reason"] = "zero_denominator_all_ratings_same_category"
        return result
    result["value"] = _finite((observed - expected) / (1 - expected))
    return result


def _score_summary(values):
    values = [_score(v) for v in values if v is not None]
    counts = Counter(values)
    return {
        "count": len(values),
        "mean": _finite(mean(values)) if values else None,
        "median": _finite(median(values)) if values else None,
        "distribution": {str(x): counts[x] for x in SCORES},
    }


def _score_pair(matrix, left_name, right_name):
    pairs = [(left, right) for left, right in matrix if left is not None and right is not None]
    gaps = [abs(left - right) for left, right in pairs]
    n = len(pairs)
    return {
        "left": left_name,
        "right": right_name,
        "paired_samples": n,
        "exact_agreement_rate": sum(g == 0 for g in gaps) / n if n else None,
        "mean_absolute_difference": _finite(mean(gaps)) if n else None,
        "mean_left_minus_right": _finite(mean(left - right for left, right in pairs)) if n else None,
        "difference_at_least_two_count": sum(g >= 2 for g in gaps),
        "difference_at_least_two_rate": sum(g >= 2 for g in gaps) / n if n else None,
        "absolute_difference_distribution": {str(x): gaps.count(x) for x in range(5)},
        "ordinal_alpha": krippendorff_alpha_ordinal(matrix),
    }


def _cohen_kappa(pairs):
    n = len(pairs)
    if not n:
        return {"value": None, "reason": "no_paired_labels"}
    observed = sum(left == right for left, right in pairs) / n
    lefts = Counter(left for left, _ in pairs)
    rights = Counter(right for _, right in pairs)
    expected = sum(lefts[label] * rights[label] for label in LABELS) / (n * n)
    if abs(1 - expected) < 1e-15:
        return {"value": None, "reason": "zero_denominator_all_ratings_same_category"}
    return {"value": _finite((observed - expected) / (1 - expected)), "reason": None}


def _label_pair(matrix, left_name, right_name):
    pairs = [(left, right) for left, right in matrix if left is not None and right is not None]
    n = len(pairs)
    return {
        "left": left_name,
        "right": right_name,
        "paired_samples": n,
        "exact_agreement_rate": sum(left == right for left, right in pairs) / n if n else None,
        "disagreement_count": sum(left != right for left, right in pairs),
        "cohen_kappa_nominal": _cohen_kappa(pairs),
        "confusion_counts": {
            left: {right: sum(a == left and b == right for a, b in pairs) for right in LABELS}
            for left in LABELS
        },
    }


def _get_review(record, name):
    ratings = record.get("ratings") or {}
    review = ratings.get(name)
    if review is not None and not isinstance(review, dict):
        raise ValueError("Every review must be a dict or None.")
    return review or {}


def _get_score(record, name, dimension):
    review = _get_review(record, name)
    return _score((review.get("scores") or {}).get(dimension))


def _get_label(record, name):
    return _label(_get_review(record, name).get("pass_fail_review"))


def _complete_review(record, name):
    return all(_get_score(record, name, dim) is not None for dim in DIMENSIONS) and _get_label(record, name) is not None


def _summarize_group(records, names):
    dimensions = {}
    for dim in DIMENSIONS:
        matrix = [[_get_score(record, name, dim) for name in names] for record in records]
        dimensions[dim] = {
            "krippendorff_alpha_ordinal": krippendorff_alpha_ordinal(matrix),
            "fleiss_kappa_nominal": fleiss_kappa(matrix, expected_raters=len(names), categories=SCORES),
            "pooled_scores": _score_summary(x for row in matrix for x in row),
            "pairwise": [
                _score_pair([(row[i], row[j]) for row in matrix], names[i], names[j])
                for i, j in combinations(range(len(names)), 2)
            ],
        }
    labels = [[_get_label(record, name) for name in names] for record in records]
    complete_labels = [row for row in labels if all(x is not None for x in row)]
    all_labels = Counter(x for row in labels for x in row if x is not None)
    complete_reviews = sum(all(_complete_review(record, name) for name in names) for record in records)
    return {
        "sample_count": len(records),
        "complete_all_judge_review_samples": complete_reviews,
        "per_judge": {
            name: {
                "complete_reviews": sum(_complete_review(record, name) for record in records),
                "missing_or_incomplete_reviews": sum(not _complete_review(record, name) for record in records),
                "scores": {
                    dim: _score_summary(_get_score(record, name, dim) for record in records)
                    for dim in DIMENSIONS
                },
                "label_distribution": {
                    label: sum(_get_label(record, name) == label for record in records)
                    for label in LABELS
                },
            }
            for name in names
        },
        "dimensions": dimensions,
        "labels": {
            "fleiss_kappa_nominal": fleiss_kappa(labels, expected_raters=len(names)),
            "pooled_distribution": {label: all_labels[label] for label in LABELS},
            "complete_label_samples": len(complete_labels),
            "unanimous_counts": {
                label: sum(all(x == label for x in row) for row in complete_labels)
                for label in LABELS
            },
            "disagreement_samples": sum(len(set(row)) > 1 for row in complete_labels),
            "pairwise": [
                _label_pair([(row[i], row[j]) for row in labels], names[i], names[j])
                for i, j in combinations(range(len(names)), 2)
            ],
        },
    }


def _human_comparison(records, names, human_reviews):
    if human_reviews is None:
        return {"provided": False, "reason": "no_human_reviews_supplied", "matched_samples": 0, "per_judge": {}}
    if not isinstance(human_reviews, dict):
        raise ValueError("human_reviews must map sample_id to a review dict.")
    known_ids = {record.get("sample_id") for record in records}
    selected = [record for record in records if record.get("sample_id") in human_reviews]
    comparisons = {}
    for name in names:
        augmented = [
            {
                "sample_id": record.get("sample_id"),
                "task": record.get("task"),
                "ratings": {"human": human_reviews[record.get("sample_id")], "judge": _get_review(record, name)},
            }
            for record in selected
        ]
        comparisons[name] = {
            "dimensions": {
                dim: _score_pair(
                    [(_get_score(record, "human", dim), _get_score(record, "judge", dim)) for record in augmented],
                    "human", name,
                )
                for dim in DIMENSIONS
            },
            "labels": _label_pair(
                [(_get_label(record, "human"), _get_label(record, "judge")) for record in augmented],
                "human", name,
            ),
        }
    return {
        "provided": True,
        "supplied_samples": len(human_reviews),
        "matched_samples": len(selected),
        "unknown_sample_ids_excluded": sorted(str(sample_id) for sample_id in human_reviews if sample_id not in known_ids),
        "per_judge": comparisons,
    }


def summarize(records, judge_names, human_reviews=None):
    """Summarize per-dimension ordinal alpha, nominal kappa and disagreement.

    records: iterable of {sample_id, task, ratings: {judge_name: review_or_None}}
    review: {scores: {dimension: strict_int_1_to_5}, pass_fail_review: LABEL, ...}
    human_reviews: optional {sample_id: review}; human ratings never enter the
    three-model Fleiss statistic. No decision thresholds are inferred here.
    """
    records = list(records)
    names = list(judge_names)
    if len(names) != 3 or len(set(names)) != 3 or any(not isinstance(x, str) or not x for x in names):
        raise ValueError("Exactly three distinct non-empty judge names are required.")
    tasks = sorted({str(record.get("task") or "UNKNOWN") for record in records})
    return {
        "methodology": {
            "score_range": [1, 5],
            "dimensions": list(DIMENSIONS),
            "score_agreement": "Each of the five dimensions reports both Krippendorff ordinal alpha and nominal Fleiss kappa over discrete scores 1..5.",
            "score_alpha": "Empirical ordinal distances; units with at least two ratings; no imputation.",
            "score_fleiss_kappa": "Nominal categories 1,2,3,4,5; all three named judges required per unit; ignores the distance between score categories, unlike ordinal alpha.",
            "label_agreement": "Nominal Fleiss kappa on units with all three named judges; PASS/REVIEW/FAIL categories",
            "undefined_statistics": "null with reason; unanimous constant ratings have zero chance-correction denominator",
            "interpretation": "Agreement is not proof of truth or evidence correctness; no acceptance threshold is selected after observing results.",
            "human_comparison": "Paired human-versus-each-judge statistics; human ratings are not added to three-model Fleiss kappa.",
        },
        "judge_names": names,
        "total_samples": len(records),
        "overall": _summarize_group(records, names),
        "by_task": {
            task: _summarize_group([record for record in records if str(record.get("task") or "UNKNOWN") == task], names)
            for task in tasks
        },
        "human_comparison": _human_comparison(records, names, human_reviews),
    }
