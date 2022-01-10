MECAB_IPADIC_REPO := git@github.com:neologd/mecab-ipadic-neologd.git
NEKO_TXT_URL := https://nlp100.github.io/data/neko.txt
AI_JA_ZIP_URL := https://nlp100.github.io/data/ai.ja.zip
NEWS_AGGREGATOR_DATA_SET := https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip

.PHONY: all
all: data data/neko.txt data/neko.txt.mecab data/ai.ja.zip data/ai.ja

data:
	mkdir -p data

data/neko.txt: data;
	wget $(NEKO_TXT_URL) -O $@

data/neko.txt.mecab: data/neko.txt;
	mecab data/neko.txt -o $@

data/ai.ja.zip: data;
	wget $(AI_JA_ZIP_URL) -O $@

data/ai.ja: data/ai.ja.zip;
	unzip data/ai.ja.zip -d $@

data/ai.ja.txt.parsed: data/ai.ja;
	cabocha -f1 data/ai.ja/ai.ja.txt > data/ai.ja.txt.parsed

data/NewsAggregatorDataset.zip:
	wget $(NEWS_AGGREGATOR_DATA_SET) -O $@

data/NewsAggregatorDataset: data/NewsAggregatorDataset.zip;
	unzip data/NewsAggregatorDataset.zip -d $@

# bin:
# 	mkdir -p bin

# bin/mecab-ipadic-neologd: bin;
# 	git clone --depth 1 $(MECAB_IPADIC_REPO) $@

# bin/mecab-ipadic-neologd/build: bin/mecab-ipadic-neologd;
# 	cd bin/mecab-ipadic-neologd; ./bin/install-mecab-ipadic-neologd -n
