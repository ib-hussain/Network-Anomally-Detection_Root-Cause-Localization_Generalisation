import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import auc
from prts import ts_precision, ts_recall
from timeeval.metrics.vus_metrics import RangePrVUS, RangeRocVUS
import pandas as pd ###################
from gragod.metrics.models import MetricsResult, SystemMetricsResult
from gragod.metrics.visualization import print_all_metrics
from gragod.types import Datasets

N_TH_SAMPLES_DEFAULT = 100
MAX_BUFFER_SIZE_DEFAULT = {Datasets.TELCO: 2, Datasets.SWAT: 3, Datasets.CISCO: 4}

# TODO: Check neither labels or predictions are None


class MetricsCalculator:
    """Calculator for precision, recall, and F1 metrics."""

    # TODO: Save scores, labels, predictions, system_scores, system_labels,
    #       system_predictions to calculate metrics later
    def __init__(
        self,
        dataset: Datasets,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        scores: torch.Tensor,
    ):
        """
        Initialize calculator with labels and predictions.

        Args:
            labels: Ground truth labels tensor (n_samples, n_nodes)
            predictions: Predicted labels tensor (n_samples, n_nodes)
        """
        self.dataset = dataset
        self.scores = scores
        self.labels = labels
        self.predictions = predictions
        self.system_scores = torch.sum(scores, dim=1)
        self.system_labels = (torch.sum(labels, dim=1) > 0).int()
        self.system_predictions = (torch.sum(predictions, dim=1) > 0).int()

        self.calculate_only_system_metrics = labels.ndim == 0 or labels.shape[1] in [
            0,
            1,
        ]

    def calculate_precision(self) -> MetricsResult | SystemMetricsResult:
        """
        Calculate precision metrics.

        Precision = True Positives / Predicted Positives

        Returns:
            MetricsResult | SystemMetricsResult: Precision metrics.
        """
        system_true_positives = torch.sum(self.system_labels & self.system_predictions)
        system_predicted_positives = torch.sum(self.system_predictions)

        system_precision = (
            system_true_positives / system_predicted_positives
            if system_predicted_positives > 0
            else 0
        )
        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_precision))

        true_positives = torch.sum((self.labels == 1) & (self.predictions == 1), dim=0)
        predicted_positives = torch.sum(self.predictions == 1, dim=0)

        per_class_precision = torch.where(
            predicted_positives > 0,
            true_positives / predicted_positives,
            torch.zeros_like(predicted_positives, dtype=torch.float),
        )
        global_precision = (
            true_positives.sum() / predicted_positives.sum()
            if predicted_positives.sum() > 0
            else 0
        )
        mean_precision = torch.mean(per_class_precision)

        return MetricsResult(
            metric_global=float(global_precision),
            metric_mean=float(mean_precision),
            metric_per_class=per_class_precision,
            metric_system=float(system_precision),
        )

    def calculate_recall(self) -> MetricsResult | SystemMetricsResult:
        """
        Calculate recall metrics.

        Recall = True Positives / Actual Positives

        Returns:
            MetricsResult | SystemMetricsResult: Recall metrics.
        """
        system_true_positives = torch.sum(self.system_labels & self.system_predictions)
        system_actual_positives = torch.sum(self.system_labels)
        system_recall = (
            system_true_positives / system_actual_positives
            if system_actual_positives > 0
            else 0
        )

        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_recall))

        true_positives = torch.sum((self.labels == 1) & (self.predictions == 1), dim=0)
        actual_positives = torch.sum(self.labels == 1, dim=0)

        per_class_recall = torch.where(
            actual_positives > 0,
            true_positives / actual_positives,
            torch.zeros_like(actual_positives, dtype=torch.float),
        )

        mean_recall = torch.mean(per_class_recall)
        global_recall = (
            true_positives.sum() / actual_positives.sum()
            if actual_positives.sum() > 0
            else 0
        )

        return MetricsResult(
            metric_global=float(global_recall),
            metric_mean=float(mean_recall),
            metric_per_class=per_class_recall,
            metric_system=float(system_recall),
        )

    def calculate_f1(
        self,
        precision: MetricsResult | SystemMetricsResult,
        recall: MetricsResult | SystemMetricsResult,
    ) -> MetricsResult | SystemMetricsResult:
        """
        Calculate F1 score from precision and recall results.

        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        When Precision + Recall = 0, F1 = 0

        Returns:
            MetricsResult | SystemMetricsResult: F1 score metrics.
        """

        # Handle division by zero for system metrics
        system_denominator = precision.metric_system + recall.metric_system
        system_f1 = (
            0.0
            if system_denominator == 0
            else (2 * precision.metric_system * recall.metric_system)
            / system_denominator
        )

        if isinstance(precision, SystemMetricsResult) or isinstance(
            recall, SystemMetricsResult
        ):
            return SystemMetricsResult(metric_system=float(system_f1))

        # Handle division by zero for per-class metrics
        denominator = precision.metric_per_class + recall.metric_per_class
        per_class_f1 = torch.zeros_like(denominator)
        non_zero_mask = denominator > 0
        per_class_f1[non_zero_mask] = (
            2
            * (
                precision.metric_per_class[non_zero_mask]
                * recall.metric_per_class[non_zero_mask]
            )
            / denominator[non_zero_mask]
        )

        mean_f1 = torch.mean(per_class_f1)

        # Handle division by zero for global metrics
        if precision.metric_global is not None and recall.metric_global is not None:
            global_denominator = precision.metric_global + recall.metric_global
            global_f1 = float(
                0.0
                if global_denominator == 0
                else (2 * precision.metric_global * recall.metric_global)
                / global_denominator
            )
        else:
            global_f1 = None

        return MetricsResult(
            metric_global=global_f1,
            metric_mean=float(mean_f1),
            metric_per_class=per_class_f1,
            metric_system=float(system_f1),
        )
    
    def FPR_TPR(self, pred):

        #system_pred=(torch.sum(pred, dim=1) > 0).int()
        #system_true_positives = torch.sum(self.system_labels & system_pred)
        #system_false_positives = torch.sum((system_pred == 1) & (self.system_labels == 0))
        #system_false_negatives = torch.sum((system_pred == 0) & (self.system_labels == 1))
        #system_true_negatives = torch.sum((system_pred == 0) & (self.system_labels == 0))

        #system_FPR = (
        #    system_false_positives / (system_false_positives + system_true_negatives)
        #    if system_false_positives > 0
        #    else 0
        #)

        #system_TPR =(
        #    system_true_positives / (system_true_positives + system_false_negatives)
        #    if system_false_positives > 0
        #    else 0
        #)
        #if self.calculate_only_system_metrics:
        #    return SystemMetricsResult(metric_system=(system_FPR, system_TPR))
        #true_positives = torch.sum((((self.labels == 1) & (pred == 1))).int())
        #false_positives= torch.sum(((pred==1) & (self.labels ==0)).int())
        #true_negatives= torch.sum(((pred==0) & (self.labels==1)).int())
        #false_negatives=torch.sum(((pred==0) & (self.labels==1)).int())
        true_positives=0
        false_positives=0
        true_negatives=0
        false_negatives=0
        for k in range(len(pred)):
            if pred[k]== self.labels[k]==1:
                true_positives+=1
            if pred[k]== self.labels[k]==0:
                true_negatives+=1
            if pred[k]==1 and self.labels[k]!=pred[k]:
                false_positives+=1
            if pred[k]==0 and self.labels[k]!=pred[k]:
                false_negatives+=1


        true_negatives=torch.tensor(true_negatives)
        true_positives=torch.tensor(true_positives)
        false_negatives=torch.tensor(false_negatives)
        false_positives=torch.tensor(false_positives)


        #print(true_positives)
        #print(true_negatives)
        #print(false_negatives)
        #print(false_positives)


        per_fpr= torch.where(
            false_positives+true_negatives>0,
            false_positives/(false_positives+true_negatives),
            torch.zeros_like(false_positives, dtype=torch.float),
        )

        per_tpr=torch.where(
            true_positives+false_negatives>0,
            true_positives/(true_positives+false_negatives),
            torch.zeros_like(true_positives, dtype=torch.float),
        )

        mean_fpr = torch.mean(per_fpr)
        mean_tpr= torch.mean(per_tpr)

        global_fpr=(
            false_positives.sum()/(false_positives.sum()+ true_negatives.sum())
        )

        global_tpr=(
            true_positives.sum()/(true_positives.sum()+false_negatives.sum())
        )



        return(per_fpr,per_tpr)
        



    def calculate_roc_score(self, Nstep) -> MetricsResult | SystemMetricsResult :
        

        roc_score=0
        mem_tprk=0
        mem_fprk=0
        fpr_list=[]
        tpr_list=[]

        scored_renorm=(self.system_scores-torch.min(self.system_scores))/(torch.max(self.system_scores)-torch.min(self.system_scores))
        #mean = torch.mean(self.system_scores)
        #std = torch.std(self.system_scores)
        #scored_renorm = (self.system_scores - mean) / std
        #q1 = torch.quantile(self.system_scores, 0.25)
        #q3 = torch.quantile(self.system_scores, 0.75)
        #iqr = q3 - q1
        #scored_renorm = (self.system_scores - torch.median(self.system_scores)) / iqr

        #max_val=torch.max(scored_renorm)
        #min_val=torch.min(scored_renorm)
        ths=np.linspace(0,1,Nstep)
        #ths=np.linspace(min_val,max_val,Nstep)

        for k in range(1,Nstep+1):
            scored=(scored_renorm > ths[k-1]).int() 

            fprk,tprk=self.FPR_TPR(scored)

            deltfpr=fprk-mem_fprk
            delttpr=tprk-mem_tprk
            roc_score+=(deltfpr*delttpr)/2


            mem_fprk=fprk
            mem_tprk=tprk
            fpr_list.append(fprk)
            tpr_list.append(tprk)

        roc_score=roc_score
        rc2=auc(fpr_list,tpr_list)
        fc=torch.tensor(rc2)
        per_rc2 = torch.where(
            fc > 0,
            fc,
            torch.zeros_like(fc, dtype=torch.float),
        )
        #print(roc_score)
        print(rc2)
        fpr_np = np.array(fpr_list)
        tpr_np = np.array(tpr_list)

      
        df = pd.DataFrame({'FPR': fpr_np, 'TPR': tpr_np, 'ths':ths})

        df.to_csv("fpr_tpr.csv", index=False)
        rc3=torch.tensor([rc2, roc_score])

        return MetricsResult(
            metric_global=rc2,
            metric_mean=float(rc2),
            metric_per_class=rc3,
            metric_system=float(rc2))




    def calculate_range_based_recall(
        self, alpha: float = 1.0
    ) -> MetricsResult | SystemMetricsResult:
        """
        Calculate range-based recall metrics.
        Based on https://arxiv.org/pdf/1803.03639.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.

        Returns:
            MetricsResult | SystemMetricsResult: Recall metrics.
        """
        system_labels_np = np.array(self.system_labels)
        system_predictions_np = np.array(self.system_predictions)

        system_recall = (
            ts_recall(system_labels_np, system_predictions_np, alpha=alpha)
            if not (
                np.allclose(np.unique(system_predictions_np), np.array([0]))
                or np.allclose(np.unique(system_labels_np), np.array([0]))
            )
            else 0
        )

        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_recall))

        labels_np = np.array(self.labels)
        predictions_np = np.array(self.predictions)

        per_class_recall = [
            (
                ts_recall(labels_np[:, i], predictions_np[:, i], alpha=alpha)
                # if there are no anomalies detected, recall is 0
                if not (
                    np.allclose(np.unique(predictions_np[:, i]), np.array([0]))
                    or np.allclose(np.unique(labels_np[:, i]), np.array([0]))
                )
                else 0
            )
            for i in range(self.labels.shape[1])
        ]
        per_class_recall = torch.tensor(per_class_recall, dtype=torch.float)
        mean_recall = torch.mean(per_class_recall)

        # doesn't make sense the global recall in range based metrics
        global_recall = None

        return MetricsResult(
            metric_global=global_recall,
            metric_mean=float(mean_recall),
            metric_per_class=per_class_recall,
            metric_system=float(system_recall),
        )

    def calculate_range_based_precision(self) -> MetricsResult | SystemMetricsResult:
        """
        Calculate range-based precision metrics.
        Based on https://arxiv.org/pdf/1803.03639.

        Returns:
            MetricsResult | SystemMetricsResult: Precision metrics.
        """
        system_labels_np = np.array(self.system_labels)
        system_predictions_np = np.array(self.system_predictions)

        system_precision = (
            ts_precision(system_labels_np, system_predictions_np, alpha=0)
            if not (
                np.allclose(np.unique(system_predictions_np), np.array([0]))
                or np.allclose(np.unique(system_labels_np), np.array([0]))
            )
            else 0
        )

        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_precision))

        labels_np = np.array(self.labels)
        predictions_np = np.array(self.predictions)

        per_class_precision = [
            (
                ts_precision(labels_np[:, i], predictions_np[:, i], alpha=0)
                # if there are no anomalies detected, precision is 0
                if not (
                    np.allclose(np.unique(predictions_np[:, i]), np.array([0]))
                    or np.allclose(np.unique(labels_np[:, i]), np.array([0]))
                )
                else 0
            )
            for i in range(self.labels.shape[1])
        ]
        per_class_precision = torch.tensor(per_class_precision, dtype=torch.float)

        mean_precision = torch.mean(per_class_precision)

        # doesn't make sense the global precision in range based metrics
        global_precision = None

        return MetricsResult(
            metric_global=global_precision,
            metric_mean=float(mean_precision),
            metric_per_class=per_class_precision,
            metric_system=float(system_precision),
        )

    def calculate_range_based_f1(
        self,
        range_based_precision: MetricsResult | SystemMetricsResult,
        range_based_recall: MetricsResult | SystemMetricsResult,
    ) -> MetricsResult | SystemMetricsResult:
        """
        Calculate range-based F1 score metrics.
        """
        return self.calculate_f1(range_based_precision, range_based_recall)

    def calculate_vus_roc(
        self,
        max_buffer_size: int | None = None,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ) -> MetricsResult | SystemMetricsResult:
        """
        Calculate VUS-ROC metrics.
        Based on https://www.paparrizos.org/papers/PaparrizosVLDB22b.pdf.

        Args:
            max_buffer_size: Maximum size of the buffer region around an anomaly.
                We iterate over all buffer sizes from 0 to ``max_buffer_size`` to
                create the surface.
            max_th_samples: Calculating precision and recall for many thresholds is
                quite slow. We, therefore, uniformly sample thresholds from the
                available score space. This parameter controls the maximum number of
                thresholds; too low numbers degrade the metrics' quality.

        Returns:
            MetricsResult | SystemMetricsResult: VUS-ROC metrics.
        """
        if max_buffer_size is None:
            max_buffer_size = MAX_BUFFER_SIZE_DEFAULT[self.dataset]

        system_labels_float64 = np.array(self.system_labels, dtype=np.float64)
        system_scores_float64 = np.array(self.system_scores, dtype=np.float64)

        vus_roc = RangeRocVUS(
            max_buffer_size=max_buffer_size,
            compatibility_mode=True,
            max_samples=max_th_samples,
        )

        system_vus_roc = (
            vus_roc(
                y_true=system_labels_float64,
                y_score=system_scores_float64,
            )
            if torch.sum(self.system_labels) > 0
            else 0
        )

        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_vus_roc))

        scores_float64 = np.array(self.scores, dtype=np.float64)
        labels_float64 = np.array(self.labels, dtype=np.float64)

        per_class_vus_roc = [
            (
                vus_roc(
                    y_true=labels_float64[:, i],
                    y_score=scores_float64[:, i],
                )
                if not (
                    np.allclose(np.unique(labels_float64[:, i]), np.array([0]))
                    or np.allclose(np.unique(scores_float64[:, i]), np.array([0]))
                )
                else 0
            )
            for i in range(labels_float64.shape[1])
        ]
        mean_vus_roc = torch.mean(torch.tensor(per_class_vus_roc))

        global_vus_roc = None

        return MetricsResult(
            metric_global=global_vus_roc,
            metric_mean=float(mean_vus_roc),
            metric_per_class=torch.tensor(per_class_vus_roc),
            metric_system=float(system_vus_roc),
        )

    def calculate_vus_pr(
        self,
        max_buffer_size: int | None = None,
        max_th_samples: int = N_TH_SAMPLES_DEFAULT,
    ) -> MetricsResult | SystemMetricsResult:
        """
        Calculate VUS-PR metrics.
        Based on https://www.paparrizos.org/papers/PaparrizosVLDB22b.pdf.

        Args:
            max_buffer_size: Maximum size of the buffer region around an anomaly.
                We iterate over all buffer sizes from 0 to ``max_buffer_size`` to
                create the surface.
            max_th_samples: Calculating precision and recall for many thresholds is
                quite slow. We, therefore, uniformly sample thresholds from the
                available score space. This parameter controls the maximum number of
                thresholds; too low numbers degrade the metrics' quality.

        Returns:
            MetricsResult | SystemMetricsResult: VUS-PR metrics.
        """
        if max_buffer_size is None:
            max_buffer_size = MAX_BUFFER_SIZE_DEFAULT[self.dataset]

        system_labels_float64 = np.array(self.system_labels, dtype=np.float64)
        system_scores_float64 = np.array(self.system_scores, dtype=np.float64)

        vus_pr = RangePrVUS(
            max_buffer_size=max_buffer_size,
            compatibility_mode=True,
            max_samples=max_th_samples,
        )

        system_vus_pr = (
            vus_pr(
                y_true=system_labels_float64,
                y_score=system_scores_float64,
            )
            if torch.sum(self.system_labels) > 0
            else 0
        )

        if self.calculate_only_system_metrics:
            return SystemMetricsResult(metric_system=float(system_vus_pr))

        scores_float64 = np.array(self.scores, dtype=np.float64)
        labels_float64 = np.array(self.labels, dtype=np.float64)

        per_class_vus_pr = [
            (
                vus_pr(
                    y_true=labels_float64[:, i],
                    y_score=scores_float64[:, i],
                )
                if not (
                    np.allclose(np.unique(labels_float64[:, i]), np.array([0]))
                    or np.allclose(np.unique(scores_float64[:, i]), np.array([0]))
                )
                else 0
            )
            for i in range(labels_float64.shape[1])
        ]
        mean_vus_pr = torch.mean(torch.tensor(per_class_vus_pr))

        global_vus_pr = None

        return MetricsResult(
            metric_global=global_vus_pr,
            metric_mean=float(mean_vus_pr),
            metric_per_class=torch.tensor(per_class_vus_pr),
            metric_system=float(system_vus_pr),
        )

    def get_all_metrics(self, alpha: float = 1.0) -> dict[str, torch.Tensor]:
        """
        Calculate all metrics and return as dictionary.

        Args:
            alpha: Relative importance of existence reward. 0 ≤ alpha ≤ 1.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of metrics.
        """
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        print(recall)
        f1 = self.calculate_f1(precision, recall)
        auc_score=self.calculate_roc_score(Nstep=40)
        range_based_precision = self.calculate_range_based_precision()
        range_based_recall = self.calculate_range_based_recall(alpha=alpha)
        range_based_f1 = self.calculate_range_based_f1(
            range_based_precision, range_based_recall
        )
        vus_roc = self.calculate_vus_roc()
        vus_pr = self.calculate_vus_pr()

        return {
            **precision.model_dump("precision"),
            **recall.model_dump("recall"),
            **f1.model_dump("f1"),
            **auc_score.model_dump("auc"),
            **range_based_precision.model_dump("range_based_precision"),
            **range_based_recall.model_dump("range_based_recall"),
            **range_based_f1.model_dump("range_based_f1"),
            **vus_roc.model_dump("vus_roc"),
            **vus_pr.model_dump("vus_pr"),
        }


def get_metrics(
    dataset: Datasets,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    range_metrics_alpha: float = 1.0,
) -> dict:
    """
    Calculate and visualize all metrics for given predictions and labels.

    Args:
        predictions: Predicted labels tensor
        labels: Ground truth labels tensor

    Returns:
        Dictionary containing all calculated metrics
    """
    calculator = MetricsCalculator(
        dataset=dataset, labels=labels, predictions=predictions, scores=scores
    )
    metrics = calculator.get_all_metrics(alpha=range_metrics_alpha)

    return metrics


def get_metrics_and_save(
    dataset: Datasets,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    save_dir: Path,
    dataset_split: str,
):
    metrics = get_metrics(dataset, predictions, labels, scores)
    print_all_metrics(metrics, f"------- {dataset_split.capitalize()} -------")
    json.dump(
        metrics,
        open(
            os.path.join(
                save_dir,
                f"{dataset_split}_metrics.json",
            ),
            "w",
        ),
    )
    return metrics
