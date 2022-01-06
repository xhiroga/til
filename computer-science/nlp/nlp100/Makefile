MECAB_IPADIC_REPO := git@github.com:neologd/mecab-ipadic-neologd.git
NEKO_TXT_URL := https://nlp100.github.io/data/neko.txt
AI_JA_ZIP_URL := https://nlp100.github.io/data/ai.ja.zip

.PHONY: all
all: bin bin/mecab-ipadic-neologd bin/mecab-ipadic-neologd/build data data/neko.txt data/neko.txt.mecab

bin:
	mkdir -p bin

bin/mecab-ipadic-neologd: bin;
	git clone --depth 1 $(MECAB_IPADIC_REPO) $@

# bin/mecab-ipadic-neologd/build: bin/mecab-ipadic-neologd;
# 	cd bin/mecab-ipadic-neologd; ./bin/install-mecab-ipadic-neologd -n

data:
	mkdir -p data

data/neko.txt: data;
	wget $(NEKO_TXT_URL) -O $@

data/neko.txt.mecab: data/neko.txt bin/mecab-ipadic-neologd;
	mecab data/neko.txt -o $@
