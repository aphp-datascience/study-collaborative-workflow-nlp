from typing import Optional

import torch
from torch import nn

from .registry import registry


class BaseClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()


@registry.classification_head("roberta")
class RobertaClassificationHead(BaseClassificationHead):
    def __init__(
        self,
        dropout: float,
        hidden_size: int,
        num_classes: int,
        bias_init: Optional[float] = None,
    ):
        """
        Classification head defined in the RobertaForSentenceClassification model from
        https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/roberta/modeling_roberta.py#L1438

        Parameters
        ----------
        dropout : float
            Dropout rate
        hidden_size : int
            embedding size (768 for BERT)
        num_classes : int
            Output size
        bias_init : float, optional
            Initial bias value of the last layer, by default None
        """
        super().__init__()
        self.num_classes = num_classes
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_classes)

        if bias_init is not None:
            with torch.no_grad():
                self.out_proj.bias.copy_(torch.tensor(bias_init))

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@registry.classification_head("simple")
class SimpleClassificationHead(BaseClassificationHead):
    def __init__(
        self,
        dropout: float,
        hidden_size: int,
        num_classes: int,
        bias_init: Optional[float] = None,
    ):
        """
        Simple dense layer

        Parameters
        ----------
        dropout : float
            Dropout rate
        hidden_size : int
            embedding size (768 for BERT)
        num_classes : int
            Output size
        bias_init : float, optional
            Initial bias value of the last layer, by default None
        """
        super().__init__()
        self.num_classes = num_classes
        dense = nn.Linear(hidden_size, num_classes)
        if bias_init is not None:
            with torch.no_grad():
                dense.bias.copy_(torch.tensor(bias_init))
        self.module = nn.Sequential(
            nn.Dropout(dropout),
            dense,
        )

    def forward(self, x, **kwargs):
        return self.module(x, **kwargs)
