import numpy as np
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score,
)


def clustering_scores(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """
    Return common clustering metrics for predictions against ground truth.

    Parameters
    ----------
    truth : np.ndarray
        Ground truth cluster labels.
    prediction : np.ndarray
        Predicted cluster labels.

    Returns
    -------
    dict[str, float]
        A dictionary containing the following metrics:
        - "adjusted_rand": Adjusted Rand Index.
        - "homogeneity": Homogeneity score.
        - "completeness": Completeness score.
        - "v_measure": V-measure score.
        - "fowlkes_mallows": Fowlkes-Mallows score.
    """
    return {
        "adjusted_rand": adjusted_rand_score(truth, prediction),
        "homogeneity": homogeneity_score(truth, prediction),
        "completeness": completeness_score(truth, prediction),
        "v_measure": v_measure_score(truth, prediction),
        "fowlkes_mallows": fowlkes_mallows_score(truth, prediction),
    }


def prediction_scores(
    truth: np.ndarray,
    prediction: np.ndarray,
    prob: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Return common binary-classification metrics for predictions against ground truth.

    Parameters
    ----------
    truth : np.ndarray
        Ground truth binary labels.
    prediction : np.ndarray
        Predicted binary labels.
    prob : Optional[np.ndarray],
        Predicted probabilities for the positive class. Required for ROC AUC and average precision calculation.

    Returns
    -------
    dict[str, float]
        - "accuracy": Accuracy of the classifier.
        - "precision": Positive-class precision.
        - "recall": Positive-class recall.
        - "f1": Positive-class F1 score.
        - "balanced_accuracy": Balanced accuracy of the classifier.
        - "matthews_correlation": Matthews correlation coefficient of the classifier.
        - "roc_auc": ROC AUC score for the positive class (if `prob` is provided and there are two classes).
        - "average_precision": Average precision score for the positive class (if `prob` is provided and there are two classes).
    """

    metrics = {
        "accuracy": accuracy_score(truth, prediction),
        "precision": precision_score(truth, prediction, zero_division=0),
        "recall": recall_score(truth, prediction, zero_division=0),
        "f1": f1_score(truth, prediction, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "matthews_correlation": matthews_corrcoef(truth, prediction),
        "roc_auc": None,
        "average_precision": None,
    }

    if prob is not None and np.unique(truth).size == 2:
        metrics["roc_auc"] = roc_auc_score(truth, prob)
        metrics["average_precision"] = average_precision_score(truth, prob)

    return metrics


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    """
    Return common binary-classification metrics for a 2x2 matrix.

    Parameters
    ----------
    confusion : np.ndarray
        A 2x2 confusion matrix. The rows correspond to the true classes and the columns correspond to the predicted classes.

    Returns
    -------
    dict[str, float]
        A dictionary containing the following metrics:
        - "accuracy": Accuracy of the classifier.
        - "precision": Positive-class precision.
        - "recall": Positive-class recall.
        - "f1": Positive-class F1 score.
    """
    confusion = np.asarray(confusion)
    if confusion.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix, got {confusion.shape}.")
    return {
        "accuracy": accuracy_from_confusion(confusion),
        "precision": precision_from_confusion(confusion),
        "recall": recall_from_confusion(confusion),
        "f1": f1_from_confusion(confusion),
    }


def accuracy_from_confusion(confusion: np.ndarray) -> float:
    """
    Return accuracy for a 2x2 confusion matrix.

    Parameters
    ----------
    confusion : np.ndarray
        A 2x2 confusion matrix. The rows correspond to the true classes and the columns correspond to the predicted classes.

    Returns
    -------
    float
        Accuracy of the classifier.
    """
    tot = np.asarray(confusion).sum()
    return np.divide(
        np.trace(confusion),
        tot,
        where=tot != 0,
        out=np.zeros_like(np.trace(confusion), dtype=float),
    )


def precision_from_confusion(confusion: np.ndarray) -> float:
    """
    Return positive-class precision for a 2x2 confusion matrix.

    Parameters
    ----------
    confusion : np.ndarray
        A 2x2 confusion matrix. The rows correspond to the true classes and the columns correspond to the predicted classes.

    Returns
    -------
    float
        Positive-class precision of the classifier.
    """
    _, fp = confusion[0]
    _, tp = confusion[1]
    return np.divide(
        tp, tp + fp, where=tp + fp != 0, out=np.zeros_like(tp, dtype=float)
    )


def recall_from_confusion(confusion: np.ndarray) -> float:
    """
    Return positive-class recall for a 2x2 confusion matrix.

    Parameters
    ----------
    confusion : np.ndarray
        A 2x2 confusion matrix. The rows correspond to the true classes and the columns correspond to the predicted classes.

    Returns
    -------
    float
        Positive-class recall of the classifier.
    """
    _, _ = confusion[0]
    fn, tp = confusion[1]
    return np.divide(
        tp, tp + fn, where=tp + fn != 0, out=np.zeros_like(tp, dtype=float)
    )


def f1_from_confusion(confusion: np.ndarray) -> float:
    """
    Return positive-class F1 for a 2x2 confusion matrix.

    Parameters
    ----------
    confusion : np.ndarray
        A 2x2 confusion matrix. The rows correspond to the true classes and the columns correspond to the predicted classes.

    Returns
    -------
    float
        Positive-class F1 score of the classifier.
    """
    _, fp = confusion[0]
    fn, tp = confusion[1]
    return np.divide(
        2 * tp,
        2 * tp + fp + fn,
        where=2 * tp + fp + fn != 0,
        out=np.zeros_like(tp, dtype=float),
    )
