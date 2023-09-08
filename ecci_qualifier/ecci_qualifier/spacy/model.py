from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import numpy as np
from optimum.bettertransformer import BetterTransformer
from spacy.language import Language
from spacy.tokens import Span

from edsml.spacy.component.models import SpacyBaseModel


class SpacyEcciQualifierModel(SpacyBaseModel):
    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        model_parts: Dict[str, Any],
        preprocess_size: int,
        batch_size: int,
        span_getters: Callable,
        annotation_setters: Callable,
        use_better_transformer: True,
    ):
        """
        Component used to apply any wrapped model to a stream of spaCy documents.

        Parameters
        ----------
        nlp : Language
            A spaCy language
        model : Union[str, Path, ModelWrapper]
            Either a path to a wrapped model, or the model itself
        preprocess_size : int
            Size (in number of examples) of the data given to the model
        batch_size : int
            Batch size for inference
        use_better_transformer : bool
            Whether to use `optimum.bettertransformer` for faster inference
        """

        super().__init__(
            nlp=nlp,
            model_parts=model_parts,
            preprocess_size=preprocess_size,
            batch_size=batch_size,
            span_getters=span_getters,
            annotation_setters=annotation_setters,
        )

        if use_better_transformer:
            self.model.transformer = BetterTransformer.transform(self.model.transformer)

    def set_extensions(self):
        if not Span.has_extension("to_keep"):
            Span.set_extension("to_keep", default=None)

    def predict(self, data, batch_size: Optional[int] = None, **kwargs):
        """
        Method called to predict

        Parameters
        ----------
        data :
            data generated by the `span_getter`
        batch_size : int
            batch size
        """
        if all(not bool(item) for item in data.values()):
            # no data
            return defaultdict(lambda: [])
        self.datamodule.setup("predict", data=data, batch_size=batch_size)

        preds = self.trainer.predict(
            self.model,
            dataloaders=self.datamodule.predict_dataloader(),
        )

        all_preds = []
        for pred in preds:
            all_preds.extend(
                list(np.atleast_1d(pred.cpu().numpy()))  # Handling 0-d array
            )

        return dict(to_keep=all_preds)
