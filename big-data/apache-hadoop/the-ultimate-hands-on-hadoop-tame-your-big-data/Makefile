MATERIALS = HadoopSlides
DATA = ml-100k
PLATFORM = HDP_2.6.5_deploy-scripts_180624d542a25

$(DATA):
	wget http://files.grouplens.org/datasets/movielens/ml-100k.zip -O $(DATA).zip
	unzip $(DATA).zip
	# rm $(DATA).zip

$(MATERIALS):
	wget http://media.sundog-soft.com/hadoop/HadoopSlides.zip -O $(MATERIALS).zip
	mkdir -p $(MATERIALS)
	unzip $(MATERIALS).zip -d $(MATERIALS)
	# rm $(MATERIALS).zip

$(PLATFORM):
	# before wget, you should agree to policy. https://www.cloudera.com/downloads/hortonworks-sandbox/hdp.html
	wget https://archive.cloudera.com/hwx-sandbox/hdp/hdp-2.6.5/$(PLATFORM).zip -O $(PLATFORM).zip
	unzip $(PLATFORM).zip
	# rm $(PLATFORM).zip

docker-deploy:
	sh docker-deploy-hdp265.sh
