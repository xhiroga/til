MECAB_IPADIC_REPO := git@github.com:neologd/mecab-ipadic-neologd.git
NEKO_TXT_URL := https://nlp100.github.io/data/neko.txt

.PHONY: all
all: bin bin/mecab-ipadic-neologd bin/mecab-ipadic-neologd/build files files/neko.txt files/neko.txt.mecab

bin:
	mkdir -p bin

bin/mecab-ipadic-neologd: bin;
	git clone --depth 1 $(MECAB_IPADIC_REPO) $@

# bin/mecab-ipadic-neologd/build: bin/mecab-ipadic-neologd;
# 	cd bin/mecab-ipadic-neologd; ./bin/install-mecab-ipadic-neologd -n

files:
	mkdir -p files

files/neko.txt: files;
	wget $(NEKO_TXT_URL) -O $@

files/neko.txt.mecab: files/neko.txt bin/mecab-ipadic-neologd;
	cat files/neko.txt | mecab > $@
