FETCH_DATA_PY_URL := https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/tutorial/text_analytics/data/languages/fetch_data.py

all: data/languages

data/languages:
	mkdir -p data/languages
	cd data/languages; wget -O fetch_data.py $(FETCH_DATA_PY_URL); python fetch_data.py
