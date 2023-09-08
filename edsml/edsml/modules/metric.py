from typing import Dict

import torch
import torchmetrics
from torchmetrics import Metric

from .registry import registry


@registry.metric("TorchMetric")
def TorchMetric(name, **kwargs):
    """
    Returns a TorchMetrics existing metric

    Parameters
    ----------
    name : str
        The name of the metric
    """
    return getattr(torchmetrics, name)(**kwargs)


@registry.metric("BinaryMetric")
class BinaryMetric(Metric):
    """
    Custom metric class that overrides torchmetrics `Metric` class
    and log F1-score and accuracy for binary classification.
    """

    full_state_update: bool = True

    def __init__(
        self,
        dist_sync_on_step: bool = True,
        epsilon: float = 1e-7,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("TP", default=torch.tensor([0]), dist_reduce_fx="sum")
        self.add_state("FP", default=torch.tensor([0]), dist_reduce_fx="sum")
        self.add_state("TN", default=torch.tensor([0]), dist_reduce_fx="sum")
        self.add_state("FN", default=torch.tensor([0]), dist_reduce_fx="sum")

        self.epsilon = epsilon

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Function to update the current state of the class variables.
        (It supposes one label example by line). For binary classification.

        Parameters
        ----------
        preds: torch.Tensor
            Tensor of current predictions.
        target: torch.Tensor
            Tensor of true classes.

        Returns
        -------
        None
        """

        if preds.ndim == 2:
            # Two labels --> We take the first item
            preds, target = preds[:, 0], target[:, 0]

        self.TP += torch.sum((preds == 1) & ((target == 1)))
        self.FP += torch.sum((preds == 1) & ((target == 0)))
        self.TN += torch.sum((preds == 0) & ((target == 0)))
        self.FN += torch.sum((preds == 0) & ((target == 1)))

    def compute(self) -> Dict:
        """
        Function to compute metric values from the current state of class variables.

        Parameters
        ----------
        None

        Returns
        -------
        metrics: Dict
            Dictionnary with computed F1-score and accuracy.
        """
        precision = self.TP.float() / (self.TP.float() + self.FP.float() + self.epsilon)
        recall = self.TP.float() / (self.TP.float() + self.FN.float() + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        accuracy = (self.TP.float() + self.TN.float()) / (
            self.TP.float()
            + self.FP.float()
            + self.TN.float()
            + self.FN.float()
            + self.epsilon
        )
        metrics = {"f1": f1, "accuracy": accuracy, "precision": precision}

        return metrics
