.PHONY: poetry-install;

langchain:
	git -C $@ pull || git clone https://github.com/xhiroga/langchain $@

poetry-install:
	poetry -c langchain install
	poetry install
