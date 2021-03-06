# Adpated from https://www.kaggle.com/theoviel/evaluation-metric-folds-baseline

import numpy as np

from sklearn.metrics import f1_score
from typing import List


def micro_f1(preds: List[List[int]], truths: List[List[int]]) -> float:
    """
    Micro F1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions. E.g., [[0, 0, 1], [0, 0, 0]].
        truths (list of lists of ints): Ground truths. E.g., [[0, 0, 1], [1, 0, 0]].

    Returns:
        float: F1 score. E.g., 0.6666666666666666
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def spans_to_binary(spans: List[List[int]], length: int = None) -> List[List[int]]:
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans. E.g., [[0, 5], [10, 15]].
        length (int): Length of the binarized spans. E.g., 10.

    Returns:
        np array [length]: Binarized spans.
            E.g., [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.].
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def span_micro_f1(preds: List[List[List[int]]], truths: List[List[List[int]]]) -> float:
    """
    Micro F1 on spans.

    Args:
        preds (list of lists of lists two ints): Prediction spans of batch of sequences.
            E.g., [[[696, 724]], [[668, 693]], [[203, 217]], [[70, 91], [176, 183]],
                  [[222, 258]], [[321, 329], [404, 413], [652, 661]], [[26, 38], [96, 118]],
                  [[56, 69]], [[5, 9]], [[10, 11]]].
        truths (list of lists of lists of two ints): Ground truth spans of batch of sequences.
            E.g., [[[696, 724]], [[668, 693]], [[203, 217]], [[70, 91], [176, 183]],
                  [[222, 258]], [[321, 329], [404, 413]], [[26, 38], [96, 118]], [[56, 69]],
                  [[5, 9]], [[10, 11]]].

    Returns:
        float: F1 score. E.g., 0.9779951100244498.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0,
                     np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)
