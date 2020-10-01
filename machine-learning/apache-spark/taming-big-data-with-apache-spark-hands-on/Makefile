MATERIALS = SparkCourse
NOTEBOOK = notebook
DATA = ml-100k

$(DATA): $(MATERIALS) $(NOTEBOOK)
	wget http://files.grouplens.org/datasets/movielens/ml-100k.zip -O $(DATA).zip
	unzip $(DATA).zip -d $(MATERIALS)
	unzip $(DATA).zip -d $(NOTEBOOK)
	# rm $(DATA).zip

$(MATERIALS):
	wget http://media.sundog-soft.com/Udemy/SparkCourse.zip -O $(MATERIALS).zip
	unzip $(MATERIALS).zip
	# rm $(MATERIALS).zip

$(NOTEBOOK):
	mkdir -p $(NOTEBOOK)

