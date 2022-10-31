#/bin/bash

mkdir -p ~/.config/git
echo '.DS_Store' >> ~/.config/git/ignore

rm -rf .venv
poetry config virtualenvs.create false
poetry install
