import numpy as np
from scipy.stats import iqr, rankdata
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def eval_scores(
    scores: list[float],
    true_scores: list[float],
    th_steps: int,
) -> tuple[list[float], list[float]]:
    """
    Calculate F1 scores for different thresholds.

    Args:
        scores: Anomaly scores.
        true_scores: Ground truth labels.
        th_steps: Number of threshold steps.
        return_thresold: Whether to return thresholds. Defaults to False.

    Returns:
        F1 scores for each threshold.
        Thresholds (if return_thresold is True).
    """
    padding_zeros = [0] * (len(true_scores) - len(scores))

    if len(padding_zeros) > 0:
        scores = padding_zeros + scores

    ranked_scores = rankdata(scores, method="ordinal")
    threshold_steps = th_steps
    threshold_values = np.array(range(threshold_steps)) * 1.0 / threshold_steps
    f1_scores = []
    threshold_list = []
    for step in range(threshold_steps):
        current_prediction = ranked_scores > threshold_values[step] * len(scores)

        f1_scores.append(f1_score(true_scores, current_prediction))

        score_threshold_index = ranked_scores.tolist().index(
            int(threshold_values[step] * len(scores) + 1)
        )
        threshold_list.append(scores[score_threshold_index])

    return f1_scores, threshold_list


def get_err_median_and_iqr(
    predicted: list[float], groundtruth: list[float]
) -> tuple[float, float]:
    """
    Calculate the median and interquartile range of the absolute error.

    Args:
        predicted: Predicted values.
        groundtruth: Ground truth values.

    Returns:
        Median and IQR of the absolute error.
    """
    absolute_errors = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    error_median = np.median(absolute_errors)
    error_interquartile_range = iqr(absolute_errors)

    return float(error_median), float(error_interquartile_range)


def get_full_err_scores(
    test_result: tuple[list[float], list[float], list[float]],
    val_result: tuple[list[float], list[float], list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate error scores for all features.

    Args:
        test_result: Test results.
        val_result: Validation results.

    Returns:
        Tuple containing:
        - all_feature_scores: Error scores for all features.
        - all_normal_scores: Normal distribution scores for all features.
    """
    test_array = np.array(test_result)
    val_array = np.array(val_result)

    num_features = test_array.shape[-1]
    all_feature_scores = []
    all_normal_scores = []

    for feature_idx in range(num_features):
        test_feature_data = test_array[:2, :, feature_idx]
        val_feature_data = val_array[:2, :, feature_idx]

        feature_scores = get_err_scores(
            (test_feature_data[0], test_feature_data[1]),
        )
        normal_scores = get_err_scores(
            (val_feature_data[0], val_feature_data[1]),
        )

        all_feature_scores.append(feature_scores)
        all_normal_scores.append(normal_scores)

    return np.array(all_feature_scores), np.array(all_normal_scores)


def get_final_err_scores(
    test_result: tuple[list[float], list[float], list[float]],
    val_result: tuple[list[float], list[float], list[float]],
) -> np.ndarray:
    """
    Get final error scores.

    Args:
        test_result: Test results.
        val_result: Validation results.

    Returns:
        Final error scores.
    """
    full_scores, _ = get_full_err_scores(test_result, val_result)

    all_scores = np.max(full_scores, axis=0)

    return all_scores


def get_err_scores(test_res: tuple[list[float], list[float]]) -> np.ndarray:
    """
    Calculate error scores.

    Args:
        test_res: Test results (predictions, ground truth).

    Returns:
        Smoothed error scores.
    """
    test_predict, test_gt = test_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(
        np.subtract(
            np.array(test_predict).astype(np.float64),
            np.array(test_gt).astype(np.float64),
        )
    )
    epsilon = 1e-2

    err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i - before_num : i + 1])

    return smoothed_err_scores


def get_loss(predict: list[float], gt: list[float]) -> float:
    """
    Calculate mean squared error loss.

    Args:
        predict: Predicted values.
        gt: Ground truth values.

    Returns:
        Mean squared error.
    """
    ground_truth_list = np.array(gt)
    predicted_list = np.array(predict)

    loss = mean_squared_error(predicted_list, ground_truth_list)

    return float(loss)


def get_f1_scores(
    total_err_scores: np.ndarray, gt_labels: list[float], topk: int = 1
) -> list[float]:
    """
    Calculate F1 scores for top-k error scores.

    Args:
        total_err_scores: Total error scores.
        gt_labels: Ground truth labels.
        topk: Number of top scores to consider. Defaults to 1.

    Returns:
        F1 scores.
    """
    print("total_err_scores", total_err_scores.shape)
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []

    for i, indexs in enumerate(topk_indices):
        sum_score = sum(
            score
            for k, score in enumerate(
                sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])
            )
        )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas, _ = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas


def get_val_performance_data(
    total_err_scores: np.ndarray,
    normal_scores: np.ndarray,
    gt_labels: list[float],
    topk: int = 1,
) -> tuple[float, float, float, float, float]:
    """
    Get performance data for validation set.

    Args:
        total_err_scores: Total error scores.
        normal_scores: Normal scores.
        gt_labels: Ground truth labels.
        topk: Number of top scores to consider. Defaults to 1.

    Returns:
        F1 score, precision, recall, AUC score, and threshold.
    """
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    total_topk_err_scores = np.sum(
        np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0
    )

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return float(f1), float(pre), float(rec), float(auc_score), float(thresold)


def get_best_performance_data(
    total_err_scores: np.ndarray,
    gt_labels: list[float],
    topk: int = 1,
) -> tuple[float, float, float, float, float]:
    """
    Get best performance data.

    Args:
        total_err_scores: Total error scores.
        gt_labels: Ground truth labels.
        topk: Number of top scores to consider. Defaults to 1.

    Returns:
        Best F1 score, precision, recall, AUC score, and threshold.
    """
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(
        total_err_scores, range(total_features - topk - 1, total_features), axis=0
    )[-topk:]

    total_topk_err_scores = np.sum(
        np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0
    )

    final_topk_fmeas, thresolds = eval_scores(total_topk_err_scores, gt_labels, 400)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return (
        float(max(final_topk_fmeas)),
        float(pre),
        float(rec),
        float(auc_score),
        float(thresold),
    )


def print_score(
    test_result: tuple[list[float], list[float], list[float]],
    val_result: tuple[list[float], list[float], list[float]],
    report: str,
) -> None:
    """
    Calculate and print the model's performance scores.

    Args:
        test_result: Results from testing the model.
        val_result: Results from validating the model.
        report: Type of report to generate ('best' or 'val').
    """
    np_test_result = np.array(test_result)

    test_labels = np_test_result[2, :, 0].tolist()
    test_scores, normal_scores = get_full_err_scores(test_result, val_result)

    top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
    top1_val_info = get_val_performance_data(
        test_scores, normal_scores, test_labels, topk=1
    )

    print("\n=========================** Result **============================\n")

    if report == "best":
        info = top1_best_info
    elif report == "val":
        info = top1_val_info
    else:
        raise ValueError("Invalid report type. Use 'best' or 'val'.")

    print(f"F1 score: {info[0]}")
    print(f"precision: {info[1]}")
    print(f"recall: {info[2]}\n")
