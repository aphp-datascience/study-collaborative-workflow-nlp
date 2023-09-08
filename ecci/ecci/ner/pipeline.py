import spacy
from edsnlp.pipelines.ner.comorbidities.helpers import get_all_pipes
from spacy.tokens import Doc, Span


def get_pipes():
    """
    Get the dictionary of comorbidity pipes.
    Format is `pipe_name`: `full_pipe_name`.
    Example : `diabetes` : `eds.comorbidities.diabetes`
    """
    pipes = get_all_pipes()
    return {pipe.split(".")[-1]: pipe for pipe in pipes}


def get_nlp(qualifiers: bool = True, no_pipes: bool = False):
    """
    Build the Spacy language
    """
    pipes = get_pipes()

    qualifiers_list = ["negation", "hypothesis", "family"]
    for q in qualifiers_list:
        if not Span.has_extension(q):
            Span.set_extension(q, default=False)

    if not Span.has_extension("to_keep"):
        Span.set_extension("to_keep", default=False)

    doc_extensions = ["note_id", "cse", "session"]
    for e in doc_extensions:
        if not Doc.has_extension(e):
            Doc.set_extension(e, default=None)

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    nlp.add_pipe(
        "eds.normalizer",
        config=dict(
            pollution=dict(
                information=True,
                bars=True,
                biology=True,
                doctors=True,
                web=True,
                coding=True,
                spaces=True,
            ),
        ),
    )

    if qualifiers:
        span_groups = list(pipes.keys())
        for q in qualifiers_list:
            nlp.add_pipe(f"eds.{q}", config=dict(on_ents_only=span_groups))

    if no_pipes:
        return nlp

    for c in pipes.values():
        nlp.add_pipe(c)

    return nlp
