# Running everything

## 1. Training
### 1.1. Full training (all data)

#### 1.1.1. Using `camembert-eds`

```bash
cd ~/ecci/ecci_qualifier/scripts/train
./script.sh training_camembert_eds
```

#### 1.1.2. Using `camembert-base`

```bash
cd ~/ecci/ecci_qualifier/scripts/train
./script.sh training_camembert_base
```

### 1.2. Per CSE

#### 1.2.1. Intial data preparation

```python
import pandas as pd

t = pd.read_csv("/export/home/share/datascientists/ecci/train/texts.csv")
e = pd.read_csv("/export/home/share/datascientists/ecci/final/all_entities.csv")

e = e.merge(
    t[["note_id","cse"]],
    on = "note_id",
    how="inner",
)

for cse, df in e.groupby("cse"):
    df.to_csv(f"/export/home/share/datascientists/ecci/final/{cse}_entities.csv", index=False)
```

#### 1.2.2. Using `camembert-eds`

```bash
cd ~/ecci/ecci_qualifier/scripts/tune/simple/collaborative
./script.sh --model 'camembert-eds'
```

#### 1.2.3. Using `camembert-base`

```bash
cd ~/ecci/ecci_qualifier/scripts/tune/simple/collaborative
./script.sh --model 'camembert-base'
```

## 2. Running NER + Qualification

```bash
cd ~/ecci/analysis/pipelines
```

### 2.1. Using `camembert-eds`

```bash
eds-toolbox slurm submit --config slurm.cfg --exit-when-done -c "export BERT_MODEL=eds && python 2_apply_all_pipes.py"
eds-toolbox slurm submit --config slurm.cfg --exit-when-done -c "export BERT_MODEL=eds && export CSE=cse180032 && python 2_apply_all_pipes.py"
eds-toolbox slurm submit --config slurm.cfg --exit-when-done -c "export BERT_MODEL=eds && export CSE=cse200055 && python 2_apply_all_pipes.py"
eds-toolbox slurm submit --config slurm.cfg --exit-when-done -c "export BERT_MODEL=eds && export CSE=cse200093 && python 2_apply_all_pipes.py"
```

### 2.2. Using `camembert-base`

```bash
eds-toolbox slurm submit --config slurm.cfg --exit-when-done -c "export BERT_MODEL=base && python 2_apply_all_pipes.py"
eds-toolbox slurm submit --config slurm.cfg --exit-when-done -c "export BERT_MODEL=base && export CSE=cse180032 && python 2_apply_all_pipes.py"
eds-toolbox slurm submit --config slurm.cfg --exit-when-done -c "export BERT_MODEL=base && export CSE=cse200055 && python 2_apply_all_pipes.py"
eds-toolbox slurm submit --config slurm.cfg --exit-when-done -c "export BERT_MODEL=base && export CSE=cse200093 && python 2_apply_all_pipes.py"
```

**Warning: those commands overwrite any previous run**

## 3. Getting all statistics and metrics

```bash
cd ~/ecci/analysis/pipelines
```

** Depending on the previous step, metrics will be computed for `base` or `eds`

```bash
python 3_get_stats.py --excel
export CSE="cse180032" && python 3_get_stats.py --excel
export CSE="cse200055" && python 3_get_stats.py --excel
export CSE="cse200093" && python 3_get_stats.py --excel
```

Results are available in `~/ecci/analysis/export`
