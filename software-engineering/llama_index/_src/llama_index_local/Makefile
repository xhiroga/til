.PHONY: poetry-install;

llama_index:
	git -C $@ pull || git clone https://github.com/xhiroga/llama_index $@

poetry-install:
	poetry install
	.venv/bin/pip install -e ./llama_index

issue_1194: llama_index poetry-install;
	.venv/bin/python issue_1194.py
