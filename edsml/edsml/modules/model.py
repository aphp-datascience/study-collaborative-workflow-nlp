from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn.modules.loss import _Loss
from torchmetrics import Metric, MetricCollection
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .classification_head import BaseClassificationHead
from .helpers.misc import shift
from .optimizer import LinearSchedule, ScheduledOptimizer
from .registry import registry
from .task import LogitsToPrediction


@registry.model("from-checkpoint")
def from_checkpoint(
    model: Union[Type[LightningModule], str],
    checkpoint_path: Union[str, Path],
) -> LightningModule:
    """
    Get a PyTorch Lightning model and load its state from checkpoint

    Parameters
    ----------
    model : Union[Type[LightningModule], str]
        Either the name under which the model is registered, or the model class
    checkpoint_path : Union[str, Path]
        Path to the checkpoint file

    Returns
    -------
    LightningModule
        The module with updated weights
    """
    if isinstance(model, str):
        model = registry.model.get(model)

    return model.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"))


@registry.model("transformer")
def transformer_from_pretrained(path: Union[str, Path]):
    return AutoModel.from_pretrained(path)


@registry.model("tokenizer")
def tokenizer_from_pretrained(
    path: Union[str, Path],
    additional_special_tokens: Optional[List[str]] = None,
):
    return AutoTokenizer.from_pretrained(
        path,
        additional_special_tokens=additional_special_tokens,
    )


class Base(LightningModule):
    def __init__(
        self,
        transformer: PreTrainedModel,
        loss: _Loss,
        classification_head: BaseClassificationHead,
        task: LogitsToPrediction,
        optimizer_params: Dict[str, Any],
        metrics: Dict[str, Metric],
        dropout: Optional[float] = None,
        batch_size: int = 32,
        label="label",
        label_dtype=None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        **kwargs,
    ):
        """
        Base model for sentence classification

        Parameters
        ----------
        transformer : PreTrainedModel
            Embedding-generating layer
        loss : _Loss
            A PyTorch loss
        classification_head : BaseClassificationHead
            Layer added on top of the embeddings
        task : LogitsToPrediction
            Which task to perform
        optimizer_params : Dict[str, Any]
            Parameters used in the `configure_parameters` hook
        metrics : Dict[str, Metric]
            Dictionary of metrics (e.g. for train and valid)
        dropout : Optional[float], optional
            Dropout rate, by default None
        batch_size : int, optional
            Batch size, by default 32
        label : str, optional
            Key of the label in the dataset, by default "label"
        label_dtype : _type_, optional
            torch dtype of the label, for eventual casting, by default None
        tokenizer : PreTrainedTokenizer, by default None
            Optional tokenizer, to save it for inference
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Embedding layers
        self.transformer = transformer

        # Pooling to focus on entities
        self.pooler = Pooler(
            mode="mean",
            dropout_p=dropout or self.transformer.embeddings.dropout.p,
            input_size=self.transformer.pooler.dense.out_features,
        )

        # Classification head
        self.classification_head = classification_head
        self.num_classes = classification_head.num_classes

        # Activation function for inference
        self.to_prediction = task.to_prediction

        # Label
        self.label = label
        self.label_dtype = label_dtype

        check = getattr(torch, label_dtype, None)
        if self.label_dtype is not None and not isinstance(check, torch.dtype):
            raise TypeError(
                f"Provided label_dtype ({label_dtype}) isn't a valid torch dtype"
            )

        # Loss
        self.criterion = loss

        # Metrics
        base_metrics = MetricCollection(metrics)

        self.train_metrics = base_metrics.clone(prefix="train/")
        self.valid_metrics = base_metrics.clone(prefix="valid/")

        # Optimizer
        self.optimizer_params = optimizer_params

        # Training
        self.batch_size = batch_size

        # (Optional) for inference
        self.tokenizer = tokenizer

    def cast(self, t: torch.tensor):
        if self.label_dtype is None:
            return t
        return getattr(t, self.label_dtype)()

    def _step(self, batch, compute_loss: bool = True):
        pass

    def forward(self, batch):
        pred, _, _ = self._step(batch, compute_loss=False)
        return pred

    def training_step(self, batch, batch_nb):
        pred, logits, loss = self._step(batch)

        return {"loss": loss, "pred": pred, "label": batch[self.label]}

    def training_step_end(self, outputs):
        outputs["loss"] = outputs["loss"].sum()

        self.train_metrics(
            preds=outputs["pred"],
            target=outputs["label"],
        )

        self.log_dict(self.train_metrics, on_step=True, on_epoch=False)

        self.log(
            "train/loss_step",
            outputs["loss"],
            on_step=True,
            on_epoch=False,
        )

        return outputs

    def training_epoch_end(self, outputs):
        train_loss = sum([output["loss"] for output in outputs])

        self.log_dict(self.train_metrics, on_epoch=True, prog_bar=True)

        self.log(
            "train/loss",
            train_loss,
            prog_bar=True,
            on_epoch=True,
        )

        # self.train_metrics.reset()

    def validation_step(self, batch, batch_nb):
        pred, logits, loss = self._step(batch)

        return {"loss": loss, "pred": pred, "label": batch[self.label]}

    def validation_step_end(self, outputs):
        outputs["loss"] = outputs["loss"].sum()

        self.valid_metrics(
            preds=outputs["pred"],
            target=outputs["label"],
        )

        self.log_dict(self.valid_metrics, on_step=True, on_epoch=True)

        self.log(
            "valid/loss_step",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
        )

        return outputs

    def validation_epoch_end(self, outputs):
        # metrics = self.valid_metrics.compute()
        valid_loss = sum([output["loss"] for output in outputs])

        self.log_dict(self.valid_metrics, on_epoch=True, prog_bar=True)

        self.log(
            "valid/loss",
            valid_loss,
            prog_bar=True,
            on_epoch=True,
        )

        # self.valid_metrics.reset()

    def configure_optimizers(self):
        optimizer = ScheduledOptimizer(
            torch.optim.Adam(
                [
                    {
                        "params": self.transformer.parameters(),
                        "lr": self.optimizer_params["transformer"]["lr"],
                        "schedules": LinearSchedule(
                            path="lr",
                            warmup_rate=self.optimizer_params["transformer"][
                                "warmup_rate"
                            ],
                            total_steps=self.optimizer_params["total_steps"],
                        ),
                    },
                    {
                        "params": self.classification_head.parameters(),
                        "lr": self.optimizer_params["head"]["lr"],
                        "schedules": LinearSchedule(
                            path="lr",
                            warmup_rate=self.optimizer_params["head"]["warmup_rate"],
                            total_steps=self.optimizer_params["total_steps"],
                        ),
                    },
                ]
            )
        )
        return optimizer


class Pooler(torch.nn.Module):
    def __init__(
        self,
        mode="mean",
        dropout_p=0.0,
        input_size=None,
        n_heads=None,
        do_value_proj=False,
    ):
        super().__init__()
        self.mode = mode
        assert mode in ("max", "sum", "mean", "attention", "first", "last")
        self.dropout = torch.nn.Dropout(dropout_p)
        if mode == "attention":
            self.key_proj = torch.nn.Linear(input_size, n_heads)
            self.value_proj = (
                torch.nn.Linear(input_size, input_size) if do_value_proj else None
            )
        self.output_size = input_size

    def forward(self, features, mask):
        device = features.device
        if self.mode == "attention" and isinstance(mask, tuple):
            position = torch.arange(features.shape[-2], device=device).reshape(
                [1] * (features.ndim - 2) + [features.shape[-2]]
            )
            mask = (mask[0].unsqueeze(-1) <= position) & (
                position < mask[1].unsqueeze(-1)
            )
            features = features.unsqueeze(-3)
        if isinstance(mask, tuple):
            original_dtype = features.dtype
            if features.dtype == torch.int or features.dtype == torch.long:
                features = features.float()
            begins, ends = mask
            if self.mode == "first":
                ends = torch.minimum(begins + 1, ends)
            if self.mode == "last":
                begins = torch.maximum(ends - 1, begins)
            begins = begins.expand(
                *features.shape[: begins.ndim - 1], begins.shape[-1]
            ).clamp_min(0)
            ends = ends.expand(
                *features.shape[: begins.ndim - 1], ends.shape[-1]
            ).clamp_min(0)
            final_shape = (*begins.shape, *features.shape[begins.ndim :])
            features = features.view(-1, features.shape[-2], features.shape[-1])
            begins = begins.reshape(
                features.shape[0],
                begins.numel() // features.shape[0] if len(features) else 0,
            )
            ends = ends.reshape(
                features.shape[0],
                ends.numel() // features.shape[0] if len(features) else 0,
            )

            max_window_size = (
                max(0, int((ends - begins).max())) if 0 not in ends.shape else 0
            )
            flat_indices = (
                torch.arange(max_window_size, device=device)[None, None, :]
                + begins[..., None]
            )
            flat_indices_mask = flat_indices < ends[..., None]
            flat_indices += (
                torch.arange(len(flat_indices), device=device)[:, None, None]
                * features.shape[1]
            )

            flat_indices = flat_indices[flat_indices_mask]
            res = F.embedding_bag(
                input=flat_indices,
                weight=self.dropout(features.reshape(-1, features.shape[-1])),
                offsets=torch.cat(
                    [
                        torch.tensor([0], device=device),
                        flat_indices_mask.sum(-1).reshape(-1),
                    ]
                )
                .cumsum(0)[:-1]
                .clamp_max(flat_indices.shape[0]),
                mode=self.mode if self.mode not in ("first", "last") else "max",
            ).reshape(final_shape)
            if res.dtype != original_dtype:
                res = res.type(original_dtype)
            return res
        elif torch.is_tensor(mask):
            features = features
            features = self.dropout(features)
            if self.mode == "first":
                mask = ~shift(mask.long(), n=1, dim=-1).cumsum(-1).bool() & mask
            elif self.mode == "last":
                mask = mask.flip(-1)
                mask = (~shift(mask.long(), n=1, dim=-1).cumsum(-1).bool() & mask).flip(
                    -1
                )

            if mask.ndim <= features.ndim - 1:
                mask = mask.unsqueeze(-1)
            if 0 in mask.shape:
                return features.sum(-2)
            if self.mode == "attention":
                weights = (
                    self.key_proj(features).masked_fill(~mask, -100000).softmax(-2)
                )  # ... tokens heads
                values = (
                    self.value_proj(features)
                    if self.value_proj is not None
                    else features
                )
                values = values.view(
                    *values.shape[:-1], weights.shape[-1], -1
                )  # ... tokens heads dim
                res = torch.einsum("...nhd,...nh->...hd", values, weights)
                return res.view(*res.shape[:-2], -1)
            elif self.mode == "max":
                features = (
                    features.masked_fill(~mask, -100000)
                    .max(-2)
                    .values.masked_fill(~(mask.any(-2)), 0)
                )
            elif self.mode == "abs-max":
                values, indices = features.abs().masked_fill(~mask, -100000).max(-2)
                features = features.gather(dim=-2, index=indices.unsqueeze(1)).squeeze(
                    1
                )
            elif self.mode in ("sum", "mean", "first", "last"):
                features = features.masked_fill(~mask, 0).sum(-2)
                if self.mode == "mean":
                    features = features / mask.float().sum(-2).clamp_min(1.0)
            elif self.mode == "softmax":
                weights = (
                    (features.detach() * self.alpha)
                    .masked_fill(~mask, -100000)
                    .softmax(-2)
                )
                features = torch.einsum(
                    "...nd,...nd->...d", weights, features.masked_fill(~mask, 0)
                )
            elif self.mode == "softmax-abs":
                weights = (
                    (features.detach().abs() * self.alpha)
                    .masked_fill(~mask, -100000)
                    .softmax(-2)
                )
                features = torch.einsum(
                    "...nd,...nd->...d", weights, features.masked_fill(~mask, 0)
                )
            return features


@registry.model("ecci-qualifier")
class QualifierModel(Base):
    def __init__(
        self,
        transformer: PreTrainedModel,
        loss: _Loss,
        classification_head: BaseClassificationHead,
        optimizer_params: Dict[str, Any],
        metrics: Dict[str, Metric],
        dropout: Optional[float] = None,
        batch_size: int = 32,
        label="label",
        **kwargs,
    ):
        if "tokenizer" in kwargs:
            transformer.resize_token_embeddings(len(kwargs["tokenizer"]))

        super().__init__(
            transformer=transformer,
            loss=loss,
            classification_head=classification_head,
            optimizer_params=optimizer_params,
            metrics=metrics,
            dropout=dropout,
            batch_size=batch_size,
            label=label,
            **kwargs,
        )

    def _step(self, batch, compute_loss: bool = True):
        y = self.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state

        logits = self.classification_head(
            self.pooler(y, (batch["span_start"], batch["span_end"]))
        ).squeeze()

        loss = 0

        if compute_loss:
            loss = self.criterion(
                logits,
                self.cast(batch[self.label]),
            )

        pred = self.to_prediction(logits)

        return pred, logits, loss


@registry.model("sentence-classification")
class SentencerModel(Base):
    def __init__(
        self,
        transformer: PreTrainedModel,
        loss: _Loss,
        classification_head: BaseClassificationHead,
        optimizer_params: Dict[str, Any],
        metrics: Dict[str, Metric],
        dropout: Optional[float] = None,
        batch_size: int = 32,
        label="label",
        **kwargs,
    ):
        super().__init__(
            transformer=transformer,
            loss=loss,
            classification_head=classification_head,
            optimizer_params=optimizer_params,
            metrics=metrics,
            dropout=dropout,
            batch_size=batch_size,
            label=label,
            **kwargs,
        )

    def _step(self, batch, compute_loss: bool = True):
        y = self.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state

        logits = self.classification_head(y).mean(dim=1).squeeze()

        loss = 0

        if compute_loss:
            loss = self.criterion(
                logits,
                self.cast(batch[self.label]),
            )

        pred = self.to_prediction(logits)

        return pred, logits, loss
