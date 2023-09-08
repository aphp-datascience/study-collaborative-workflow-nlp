# Usage

## Using the model via spaCy / EDSNLP

### Setting things up

Since the model is a **qualification module**, it works on **exisiting entities**.  Thus, to use it with spaCy, you first need a pipe that extracts entities. For instance, you could use the [`eds-matcher`](https://aphp.github.io/edsnlp/latest/pipelines/core/matcher/) pipe from EDS-NLP:

```python
nlp = spacy.blank("eds")

nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe(
    "eds.matcher",
    config=dict(
        attr="NORM",
        regex=dict(diseases=["diabete", "hta", "avc", "lymphome", "cancer"]),
    ),
)
```

Then, the model can be added to the `nlp` object:

```python
nlp.add_pipe("eds.charlson-qualifier")
```

!!! info
    On each entity, the qualifier will add a `to_keep` extension which can take the values:

    - 0 if the entitiy has to be discarded
    - 1 else

Finally, the full pipeline can be executed on a piece of text:

```python
text = "Le patient est sûrement diabétique"
doc = nlp(text)
```

### A complete example

```python
text = """

SERVICE LYMPHOME ET CANCER

Le patient est suivi depuis 2020 à l'hôpital. Il est atteint d'un diabète de type II.
Je vous l'envoie pour une suspicion d'AVC survenu la semaine dernière.
Son facteur de risque cardiovasculaire principal est l'HTA.

Conseils pour les fumeurs: Il faut arrêter.
"""
```

On the text above, and given the configuration of our `eds.matcher`, we are expecting the following extractions:

| Entity     | Is a relevant entity ? | Remark                                                  |
| ---------- | ---------------------- | ------------------------------------------------------- |
| `LYMPHOME` | ❌                      | irrelevant entity since it refers to the care site name |
| `CANCER`   | ❌                      | irrelevant entity since it refers to the care site name |
| `diabète`  | ✅                      |                                                         |
| `AVC`      | ❌                      | irrelevant entity since it is hypothetical              |
| `HTA`      | ✅                      |                                                         |


Let us check the model's output:

```python
doc = nlp(text)

for ent in doc.ents:
    print(ent, ent._.to_keep)

# Out: LYMPHOME 0
# Out: CANCER 0
# Out: diabete 1
# Out: AVC 0
# Out: HTA 1
```

We can see that for this example, the model correcly identified relevant and irrelevant entities.

### Comparison with a rule-based approach

EDS-NLP provides three rule-based qualification pipes to handle [negation](https://aphp.github.io/edsnlp/latest/pipelines/qualifiers/negation/), [hypothesis](https://aphp.github.io/edsnlp/latest/pipelines/qualifiers/hypothesis/) and [family](https://aphp.github.io/edsnlp/latest/pipelines/qualifiers/family/). Let us check their output on the same example:

```python
nlp.add_pipe("eds.negation")
nlp.add_pipe("eds.hypothesis")
nlp.add_pipe("eds.family")

doc = nlp(text)
for ent in doc.ents:
    print(ent, ent._.negation, ent._.family, ent._.hypothesis)

# Out: LYMPHOME False False False
# Out: CANCER False False False
# Out: diabete False False False
# Out: AVC False False True
# Out: HTA False False True
```


We can see the following issues:

- `LYMPHOME` and `CANCER` aren't qualified as irrelevant
- `HTA` is considered hypothetical du to the preceding "risque" mention

## Scaling up: using the model on multiple texts

!!! warning "Long inference time"
    The model can take a fairly long time to run on CPU. You should prefer to use it on GPU

We can simply use an existing helper from `edsnlp` to run our pipeline on a Pandas DataFrame.
First let us make a dummy DataFrame:

```python
import pandas as pd

note = pd.DataFrame(
    data=dict(
        note_text=[text],
        note_id=[0],
    )
)
```

Then simply run:

```python
from edsnlp.processing.simple import pipe

note_nlp = pipe(note=note, nlp=nlp, extensions=["to_keep"])
```

## On GPU

We recommand to use `edstoolbox` to run jobs on GPU. Please read the [corresponding documentation](https://datasciencetools-pages.eds.aphp.fr/edstoolbox/cli/slurm/) to have a better understanding of it. Here is a simple tutorial on how to do it:

=== "1. Install `edstoolbox`"
    ```bash
    pip install edstoolbox
    ```

=== "2. Define your job confguration file"
    For instance:
    ```toml title="slurm.cfg"
    [slurm]
    gpu_type = "t4"
    log_path = "./logs"
    mem = 42G
    job_duration = "24:00:00"
    n_gpu = 4
    n_cpu = 5
    ```

=== "3. Define a `script.py` file"
    To run the model on GPU, you should put your code logic in a single `script.py` file. For instance:

    ```python title="script.py"

    import spacy
    import pandas as pd
    from edsnlp.processing.simple import pipe

    def apply_pipe(df):
        nlp = spacy.blank("eds")

        nlp.add_pipe("eds.sentences")
        nlp.add_pipe("eds.normalizer")
        nlp.add_pipe( # (1)!
            "eds.matcher",
            config = dict(
                attr = "NORM",
                regex = dict(
                    diseases = ["diabete", "hta", "avc", "lymphome", "cancer"]
                )
            )
        )
        nlp.add_pipe("eds.charlson-qualifier")

        note_nlp = pipe(note=df, nlp=nlp, extensions = ["to_keep"])

        return note_nlp

    if __name__ == "__main__":

        df = pd.read_csv("path/to/my/notes.csv")
        note_nlp = apply_pipe(df)
        nlp_nlp.to_csv("path/to/save/outputs.csv")
    ```

    1. Add any pipes you want here

=== "4. Run it !"
    ```bash
    eds-toolbox slurm submit --config slurm.cfg -c "python script.py"
    ```
