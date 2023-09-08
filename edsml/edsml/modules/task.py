import torch
from torch import nn

from .registry import registry


@registry.task("LogitsToPrediction")
class LogitsToPrediction(nn.Module):
    def __init__(self, num_classes, multilabel):
        super().__init__()
        self.num_classes = num_classes
        self.multilabel = multilabel

        if num_classes > 1 and not multilabel:
            self.to_prediction = self.with_argmax
        # elif num_classes > 1 and multilabel:
        #     pass
        else:  # num_classes = 1 or multilabel
            self.to_prediction = self.with_sigmoid

    @staticmethod
    def with_sigmoid(logits):
        return torch.sigmoid(logits).round().int()

    @staticmethod
    def with_argmax(logits):
        return torch.argmax(logits, dim=1)
