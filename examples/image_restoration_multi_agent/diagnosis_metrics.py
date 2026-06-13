"""Stage E baseline metrics for VLM degradation diagnosis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from schemas import DegradationType, DiagnosisParseStatus, VLMDiagnosisAttempt


def build_diagnosis_metrics(
    records: Sequence[tuple[DegradationType, VLMDiagnosisAttempt]],
) -> dict[str, Any]:
    """Aggregate API, parsing, classification, and latency diagnostics."""

    labels = [category.value for category in DegradationType]
    confusion_matrix = {truth: {prediction: 0 for prediction in labels} for truth in labels}
    invalid_by_true = {truth: 0 for truth in labels}
    true_counts = {truth: 0 for truth in labels}
    predicted_counts = {prediction: 0 for prediction in labels}
    valid_count = 0
    correct_count = 0

    for truth, attempt in records:
        truth_value = truth.value
        true_counts[truth_value] += 1
        if attempt.parse_status != DiagnosisParseStatus.VALID or attempt.diagnosis is None:
            invalid_by_true[truth_value] += 1
            continue
        prediction = attempt.diagnosis.primary_type.value
        valid_count += 1
        predicted_counts[prediction] += 1
        confusion_matrix[truth_value][prediction] += 1
        if prediction == truth_value:
            correct_count += 1

    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = confusion_matrix[label][label]
        precision = true_positive / predicted_counts[label] if predicted_counts[label] else 0.0
        recall = true_positive / true_counts[label] if true_counts[label] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    total = len(records)
    latencies = [attempt.latency_seconds for _, attempt in records]
    return {
        "sample_count": total,
        "api_success_rate": sum(attempt.api_succeeded for _, attempt in records) / total if total else 0.0,
        "parse_success_rate": valid_count / total if total else 0.0,
        "accuracy_on_valid": correct_count / valid_count if valid_count else None,
        "valid_response_count": valid_count,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
        "invalid_by_true": invalid_by_true,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
    }
