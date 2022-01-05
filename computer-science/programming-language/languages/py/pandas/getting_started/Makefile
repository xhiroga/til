TITANIC_CSV_URL := https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv
AIR_QUALITY_NO2_URL := https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2.csv

.PHONY: all
all: data data/titanic.csv data/air_quality_no2.csv

data:
	mkdir -p data

data/titanic.csv: data;
	curl -o data/titanic.csv $(TITANIC_CSV_URL)

data/air_quality_no2.csv: data;
	curl -o data/air_quality_no2.csv $(AIR_QUALITY_NO2_URL)
