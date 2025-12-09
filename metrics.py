import logging
import numpy as np
import torch
from typing import Dict, List, Union

def ranking_metrics(scores, target, k_list=[1, 3, 10]) -> Dict[str, Union[int, float, List[int]]]:
    """
    Compute link prediction metrics (MR, MRR, Hits@K).
    
    Args:
        scores: Ranking scores for entities
        target: Target entity index
        k_list: List of K values for Hits@K metric
    
    Returns:
        Dictionary with keys: 'mr', 'mrr', 'hits', 'target_score'
    """
    _, sorted_idx = torch.sort(scores)
    find_target = sorted_idx == target

    target_rank = int(torch.nonzero(find_target)[0, 0] + 1)
    target_score = float(scores[target].item())  # Get the score of the target entity
    hits_at_k = [int(target_rank <= k) for k in k_list]

    return {
        'mr': target_rank,
        'mrr': 1 / target_rank,
        'hits': hits_at_k,
        'target_score': target_score
    }

def classification_metrics(predictions: list, true_labels: list, scores: list = None) -> Dict[str, float]:
    """
    Compute classification metrics including accuracy, precision, recall, F1, PR AUC, and ROC AUC.
    
    Args:
        predictions: Binary predictions (0 or 1)
        true_labels: Ground truth binary labels (0 or 1)
        scores: Optional confidence scores for AUC calculations. If None, uses predictions as scores.
    
    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc'
    """
    try:
        y_pred = np.asarray(predictions)
        y_true = np.asarray(true_labels)
    except Exception as e:
        raise ValueError(f"Error converting inputs to numpy arrays: {e}")
    if y_pred.shape != y_true.shape:
        raise ValueError("Predictions and true labels must have the same shape.")

    # TP: y_pred == 1 AND y_true == 1
    TP = np.sum((y_pred == 1) & (y_true == 1))
    # TN: y_pred == 0 AND y_true == 0
    TN = np.sum((y_pred == 0) & (y_true == 0))
    # FP: y_pred == 1 AND y_true == 0
    FP = np.sum((y_pred == 1) & (y_true == 0))
    # FN: y_pred == 0 AND y_true == 1
    FN = np.sum((y_pred == 0) & (y_true == 1))

    # (TP + TN) / (TP + TN + FP + FN)
    total_samples = TP + TN + FP + FN
    accuracy = (TP + TN) / total_samples if total_samples > 0 else 0.0

    # TP / (TP + FP)
    precision_denominator = TP + FP
    precision = TP / precision_denominator if precision_denominator > 0 else 0.0

    # TP / (TP + FN)
    recall_denominator = TP + FN
    recall = TP / recall_denominator if recall_denominator > 0 else 0.0

    # 2 * (Precision * Recall) / (Precision + Recall)
    f1_denominator = precision + recall
    f1_score = 2 * (precision * recall) / f1_denominator if f1_denominator > 0 else 0.0
    
    # Compute PR AUC and ROC AUC if scores are provided
    pr_auc = 0.0
    roc_auc = 0.0
    if scores is not None:
        try:
            from sklearn.metrics import auc, precision_recall_curve, roc_curve, roc_auc_score
            y_scores = np.asarray(scores)
            
            # ROC AUC
            roc_auc = roc_auc_score(y_true, y_scores)
            
            # PR AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall_curve, precision_curve)
        except ImportError:
            logging.warning("scikit-learn not available. PR AUC and ROC AUC set to 0.0")
        except Exception as e:
            logging.warning(f"Error computing AUC metrics: {e}. PR AUC and ROC AUC set to 0.0")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc
    }
