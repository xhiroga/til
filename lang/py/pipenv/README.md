# Pipenv

## Create

```shell script
brew install pipenv

# create .venv in project root, alternative to $HOME/.local/share/virtualenvs/
export PIPENV_VENV_IN_PROJECT=true

pipenv --python 3.8.1

# show interpreter path
pipenv --py

pipenv install --dev --pre black
pipenv install --dev isort
```

## Usage

```shell script
pipenv run isort -rc -vb .    # all files under current
```


## References

- https://sourcery.ai/blog/python-best-practices/
