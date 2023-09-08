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

> - For `poetry < 1.4`: `~/.local/share/pypoetry/venv/lib/python3.7/site-packages/poetry/utils/commands.py`
> - For `poetry >= 1.4`: `~/.local/share/pypoetry/venv/lib/python3.7/site-packages/poetry/utils/constants.py`

Simply set `REQUESTS_TIMEOUT = 150`

```bash
poetry publish --build --repository gitlab --username <YourUsername>
```
