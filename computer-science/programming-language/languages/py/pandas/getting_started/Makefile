TITANIC_CSV_URL := https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv

.PHONY: all
all: data/titanic.csv

data:
	mkdir -p data

data/titanic.csv: data;
	curl -o data/titanic.csv $(TITANIC_CSV_URL)
