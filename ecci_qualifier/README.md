## SETUP

### Virtual environment

```bash
python -m venv .venv
```

### Poetry

#### (Optional) Installing poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### (Optional) Setup

```bash
poetry config virtualenvs.create false --local
poetry config virtualenvs.in-project true --local
```

#### Install dependencies and package

```bash
poetry install
```

#### Publish

:warning: You might have to increase the REQUESTS_TIMEOUT of poetry, which is hardcoded to 15s at

> `~/.local/share/pypoetry/venv/lib/python3.7/site-packages/poetry/utils/commands.py`

Simply set `REQUESTS_TIMEOUT = 150`

```bash
poetry publish --build --repository gitlab --username <YourUsername>
```

## USAGE

```
pip install ecci-qualifier --extra-index-url file:///export/home/share/datascientists/models_catalogue/simple
```

The model is ready to use in a spaCy pipeline with e.g. edsnlp. It will populate the `_.to_keep` extension

```python
import spacy

nlp = spacy.blank("eds")

nlp.add_pipe("eds.sentences")
nlp.add_pipe(
    "eds.normalizer",
    config=dict(
        accents=True,
        lowercase=True,
    ),
)

nlp.add_pipe(
    "eds.matcher",
    config=dict(
        attr="NORM",
        regex=dict(
            diabete="diabete",
            avc="avc",
        ),
    ),
)

nlp.add_pipe(
    "eds.ecci-qualifier",
)


text = """
Le patient est suivi depuis 2020 à l'hôpital. Il est atteint d'un diabete de type II.
Je vous l'envoie pour une suspicion d'AVC survenu la semaine dernière.
"""

doc = nlp(text)

for ent in doc.ents:
    print(ent, ent._.to_keep)

# Out: diabete 1
# Out: AVC 0
```
