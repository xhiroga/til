NEKO_TXT_URL := https://nlp100.github.io/data/neko.txt

.PHONY: all
all: files files/neko.txt

files:
	mkdir -p files

files/neko.txt:
	wget $(NEKO_TXT_URL) -O $@
