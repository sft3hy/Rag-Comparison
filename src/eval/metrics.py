"""Evaluation metrics for RAG pipelines."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import re
from scipy import stats
from loguru import logger


def recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    """Calculate Recall@k.

    Args:
        relevant_ids: List of relevant document IDs
        retrieved_ids: List of retrieved document IDs
        k: Number of top results to consider

    Returns:
        Recall@k score
    """
    if not relevant_ids:
        return 0.0

    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    hits = len(retrieved_k.intersection(relevant_set))
    return hits / len(relevant_set)


def mean_reciprocal_rank(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Args:
        relevant_ids: List of relevant document IDs
        retrieved_ids: List of retrieved document IDs

    Returns:
        MRR score
    """
    relevant_set = set(relevant_ids)

    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            return 1.0 / i

    return 0.0


def exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate Exact Match score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def token_f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def rouge_l(prediction: str, ground_truth: str) -> float:
    """Calculate ROUGE-L score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        ROUGE-L F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    # Calculate LCS
    lcs_length = _lcs_length(pred_tokens, truth_tokens)

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(truth_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Calculate longest common subsequence length.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        LCS length
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def character_error_rate(prediction: str, ground_truth: str) -> float:
    """Calculate Character Error Rate (CER).

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text

    Returns:
        CER score
    """
    if not ground_truth:
        return 0.0 if not prediction else 1.0

    distance = _levenshtein_distance(prediction, ground_truth)
    return distance / len(ground_truth)


def word_error_rate(prediction: str, ground_truth: str) -> float:
    """Calculate Word Error Rate (WER).

    Args:
        prediction: Predicted text
        ground_truth: Ground truth text

    Returns:
        WER score
    """
    pred_words = prediction.split()
    truth_words = ground_truth.split()

    if not truth_words:
        return 0.0 if not pred_words else 1.0

    distance = _levenshtein_distance(pred_words, truth_words)
    return distance / len(truth_words)


def _levenshtein_distance(seq1, seq2) -> int:
    """Calculate Levenshtein distance between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Edit distance
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # deletion
                    dp[i][j - 1] + 1,  # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    return dp[m][n]


def numeric_accuracy(
    prediction: str,
    ground_truth: str,
    abs_tolerance: float = 1e-6,
    rel_tolerance: float = 0.01,
) -> float:
    """Calculate accuracy for numeric answers.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        abs_tolerance: Absolute tolerance
        rel_tolerance: Relative tolerance

    Returns:
        1.0 if within tolerance, 0.0 otherwise
    """
    pred_num = _extract_number(prediction)
    truth_num = _extract_number(ground_truth)

    if pred_num is None or truth_num is None:
        return exact_match(prediction, ground_truth)

    # Check absolute and relative tolerance
    abs_diff = abs(pred_num - truth_num)
    rel_diff = abs_diff / abs(truth_num) if truth_num != 0 else abs_diff

    if abs_diff <= abs_tolerance or rel_diff <= rel_tolerance:
        return 1.0

    return 0.0


def _extract_number(text: str) -> Optional[float]:
    """Extract first number from text.

    Args:
        text: Input text

    Returns:
        Extracted number or None
    """
    # Match numbers including scientific notation
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    match = re.search(pattern, text)

    if match:
        try:
            return float(match.group())
        except ValueError:
            return None

    return None


def table_cell_f1(
    predicted_table: List[List[str]], ground_truth_table: List[List[str]]
) -> float:
    """Calculate cell-level F1 for table extraction.

    Args:
        predicted_table: Predicted table cells
        ground_truth_table: Ground truth table cells

    Returns:
        Cell-level F1 score
    """
    if not predicted_table or not ground_truth_table:
        return 0.0

    # Flatten tables
    pred_cells = [normalize_answer(cell) for row in predicted_table for cell in row]

    truth_cells = [normalize_answer(cell) for row in ground_truth_table for cell in row]

    # Calculate matches
    pred_set = Counter(pred_cells)
    truth_set = Counter(truth_cells)

    common = pred_set & truth_set
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_cells)
    recall = num_common / len(truth_cells)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def table_structure_accuracy(
    predicted_shape: Tuple[int, int], ground_truth_shape: Tuple[int, int]
) -> float:
    """Calculate structure accuracy for tables.

    Args:
        predicted_shape: (n_rows, n_cols) of predicted table
        ground_truth_shape: (n_rows, n_cols) of ground truth table

    Returns:
        1.0 if structure matches, 0.0 otherwise
    """
    return float(predicted_shape == ground_truth_shape)


def bootstrap_confidence_interval(
    scores: np.ndarray, confidence_level: float = 0.95, n_bootstrap: int = 10000
) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval.

    Args:
        scores: Array of scores
        confidence_level: Confidence level (e.g., 0.95)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    bootstrap_means = []

    n_samples = len(scores)
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(sample))

    mean_score = np.mean(scores)
    alpha = 1 - confidence_level

    lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return mean_score, lower_bound, upper_bound


def paired_t_test(
    scores1: np.ndarray, scores2: np.ndarray, alternative: str = "two-sided"
) -> Tuple[float, float]:
    """Perform paired t-test.

    Args:
        scores1: Scores from first method
        scores2: Scores from second method
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Tuple of (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_rel(scores1, scores2, alternative=alternative)
    return t_stat, p_value


def wilcoxon_test(
    scores1: np.ndarray, scores2: np.ndarray, alternative: str = "two-sided"
) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test.

    Args:
        scores1: Scores from first method
        scores2: Scores from second method
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Tuple of (statistic, p_value)
    """
    statistic, p_value = stats.wilcoxon(
        scores1, scores2, alternative=alternative, zero_method="wilcox"
    )
    return statistic, p_value


class MetricsCalculator:
    """Calculator for all evaluation metrics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize metrics calculator.

        Args:
            config: Configuration dict
        """
        self.config = config or {}
        logger.info("Metrics calculator initialized")

    def calculate_retrieval_metrics(
        self,
        relevant_ids: List[str],
        retrieved_ids: List[str],
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Calculate all retrieval metrics.

        Args:
            relevant_ids: Relevant document IDs
            retrieved_ids: Retrieved document IDs
            k_values: K values for Recall@k

        Returns:
            Dict of metric scores
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        metrics = {}

        # Recall@k
        for k in k_values:
            metrics[f"recall@{k}"] = recall_at_k(relevant_ids, retrieved_ids, k)

        # MRR
        metrics["mrr"] = mean_reciprocal_rank(relevant_ids, retrieved_ids)

        return metrics

    def calculate_qa_metrics(
        self, prediction: str, ground_truth: str
    ) -> Dict[str, float]:
        """Calculate all QA metrics.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            Dict of metric scores
        """
        return {
            "exact_match": exact_match(prediction, ground_truth),
            "f1": token_f1_score(prediction, ground_truth),
            "rouge_l": rouge_l(prediction, ground_truth),
            "numeric_accuracy": numeric_accuracy(prediction, ground_truth),
        }

    def calculate_ocr_metrics(
        self, prediction: str, ground_truth: str
    ) -> Dict[str, float]:
        """Calculate OCR quality metrics.

        Args:
            prediction: OCR output
            ground_truth: Ground truth text

        Returns:
            Dict of metric scores
        """
        return {
            "cer": character_error_rate(prediction, ground_truth),
            "wer": word_error_rate(prediction, ground_truth),
        }

    def calculate_statistical_significance(
        self, scores1: np.ndarray, scores2: np.ndarray, test: str = "paired_t"
    ) -> Dict[str, float]:
        """Calculate statistical significance.

        Args:
            scores1: First set of scores
            scores2: Second set of scores
            test: 'paired_t' or 'wilcoxon'

        Returns:
            Dict with test results
        """
        if test == "paired_t":
            statistic, p_value = paired_t_test(scores1, scores2)
        elif test == "wilcoxon":
            statistic, p_value = wilcoxon_test(scores1, scores2)
        else:
            raise ValueError(f"Unknown test: {test}")

        mean_diff = np.mean(scores1 - scores2)

        # Calculate confidence intervals
        mean1, ci1_lower, ci1_upper = bootstrap_confidence_interval(scores1)
        mean2, ci2_lower, ci2_upper = bootstrap_confidence_interval(scores2)

        return {
            "test": test,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "mean_diff": float(mean_diff),
            "method1_mean": float(mean1),
            "method1_ci": (float(ci1_lower), float(ci1_upper)),
            "method2_mean": float(mean2),
            "method2_ci": (float(ci2_lower), float(ci2_upper)),
            "significant": p_value < 0.05,
        }
