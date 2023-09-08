# Training

The model was developped in parallel with the development of 18 NER pipelines available on EDS-NLP.

## The training data
Those pipelines extracts 16 conditions from the Charlson Comorbidity Index, along with alcohol and tobacco consumption. As such, the training data consists of entities representing one of those 18 conditions.  A total of **4907** entities were used. **4416** went into the training set and **491** into the validation set.

From each entity, a training example was build by extracting a snippet comprised of the sentence containing the entity plus one sentence before and one sentence after.

<p align="center">
  <img src="../assets/stats/number_of_words.png" width="70%">
</p>

The gold `to_keep` single label was build from aggregating 4 gold attributes:


| Attribute    | Definition                              | % in the training dataset | Example                              |
| ------------ | --------------------------------------- | ------------------------- | ------------------------------------ |
| `negation`   | is the entity negated ?                 | 15.3 %                    | "Pas de diabète."                    |
| `hypothesis` | is the entity hypothetical ?            | 10.1 %                    | "Diabète probable."                  |
| `family`     | is the entity refering to the patient ? | 2.1 %                     | "Le père du patient est diabétique"  |
| `irrelevant` | is the entity relevant?                 | 1.1 %                     | "Contactez service-diabetes@aphp.fr" |

## The training procedure

The command to train the model is

```bash
cd scripts/train
./script.sh train_final # (1)!
```

1. `train_final` can be replaced by the name of any valid training configuration located in the `scripts/configs/` directory. Here, `train_final.cfg` will be used.

## Train the model on your own data

In order to be able to run the training procedure, you will need to:

- provide a dataset
- adapt the configuration file

### 1. Provide a dataset

The simplest way to provide a dataset is as a CSV file. The file contains one entity per row, with data such as the snippet containing the entity, and its potition in the snippet.

The follwing columns are expected:

- `entity_id`: unique row identifier.
- `note_text`: snippet text.
- `offset_begin`: position (in number of characters) of the start of the entity in the corresponding `note_text`.
- `offset_end`: position (in number of characters) of the end of the entity in the corresponding `note_text`.
- `to_keep`: boolean value set to `True` if the entity is relevant, `False` else.

### 2. Adapt the configuration file

Finally, a few modifications should be made to the configuration file. For instance, on the `train_final.cfg` file:

??? "Modifications of the `train_final.cfg`"
    ```toml hl_lines="22 23 34 113"
    [script]
    name = "tune"
    model = ${model}
    trainer = ${trainer}
    data = ${data}
    seed = 42

    [task]
    @task = "LogitsToPrediction"
    num_classes = 1
    multilabel = false

    [data]
    @data = "base-datamodule"
    dataset = ${dataset_loader}
    tokenizer = ${tokenizer}
    batch_size = 32
    padding = true

    [dataset_loader]
    @data = "dataset-from-DF"
    data = "/export/home/share/datascientists/ecci/final/all_entities.csv" # (1)!
    save_path = "/data/scratch/tpetitjean/ML/datasets/ecci" # (2)!

    [classification_head]
    @classification_head = "simple"
    dropout = ${transformer:embeddings.dropout.p}
    hidden_size = ${transformer:pooler.dense.out_features}
    num_classes = ${task.num_classes}
    bias_init = 0.9445

    [transformer]
    @model = "transformer"
    path = "/data/scratch/tpetitjean/ML/models/camembert-base" # (3)!

    [tokenizer]
    @model = "tokenizer"
    path = ${transformer.path}

    [loss]
    @loss = "PytorchLoss"
    name = "BCEWithLogitsLoss"
    pos_weight = 0.39

    [model]
    @model = "ecci-qualifier"
    transformer = ${transformer}
    classification_head = ${classification_head}
    loss = ${loss}
    optimizer_params = ${optimizer_params}
    total_steps = ${trainer:max_steps}
    metrics = ${metrics}
    task = ${task}
    label_dtype = "float"
    tokenizer = ${tokenizer}

    [trainer]
    @trainer = "PytorchLightningTrainer"
    max_steps = 1200
    logger = ${logger}
    callbacks = [${checkpoint}, ${early_stopping}, ${lr_monitor}]

    [trainer.cpu]
    accelerator = "auto"
    devices = "auto"
    strategy = null

    [trainer.gpu]
    accelerator = "gpu"
    devices = -1
    strategy = "dp"
    auto_select_gpus = true

    [optimizer_params]
    total_steps = ${trainer:max_steps}

    [optimizer_params.head]
    lr = 0.0005
    warmup_rate = 0

    [optimizer_params.transformer]
    lr = 5e-05
    warmup_rate = 0.1

    [metrics]

    [metrics.f1]
    @metric = "TorchMetric"
    name="F1Score"
    task="binary"
    dist_sync_on_step=True

    [metrics.precision]
    @metric = "TorchMetric"
    name="Precision"
    task="binary"
    dist_sync_on_step=True

    [metrics.recall]
    @metric = "TorchMetric"
    name="Recall"
    task="binary"
    dist_sync_on_step=True

    [checkpoint]
    @callback = "PytorchLightningCallback"
    name = "ModelCheckpoint"
    monitor = "valid/f1"
    mode = "max"
    save_top_k = 1
    verbose = true
    every_n_epochs = 1
    dirpath = "/data/scratch/tpetitjean/ML/checkpoints_test" # (4)!
    auto_insert_metric_name = false
    save_weights_only = true
    filename = "tune-head:simple-t_lr:5e-05-h_lr:0.0005-valid_f1:{valid/f1:.4f}"

    [early_stopping]
    @callback = "PytorchLightningCallback"
    name = "EarlyStopping"
    monitor = "valid/f1"
    patience = 3
    mode = "max"

    [lr_monitor]
    @callback = "PytorchLightningCallback"
    name = "LearningRateMonitor"
    logging_interval = "epoch"

    [logger]
    @logger = "TensorBoard"
    save_dir = "~/tensorboard_data"
    name = ${script.name}
    default_hp_metric = false
    ```

    1. Insert here the path of your custom dataset
    2. Insert here the path where the formatted dataset will be saved
    3. Insert here the path to your `BERT` model
    4. Insert here the path where checkpoints will be saved
