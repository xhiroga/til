DOWNLOAD_URL := https://www.aozora.gr.jp/cards/000148/files/752_ruby_2438.zip
DOWNLOAD_DIR := ./data
EXTRACTED_DIR := 752_ruby_2438

data:
	mkdir -p data

data/bocchan.txt: data;
	mkdir -p $(DOWNLOAD_DIR)
	wget $(DOWNLOAD_URL) -P $(DOWNLOAD_DIR)
	unzip $(DOWNLOAD_DIR)/$(EXTRACTED_DIR).zip -d $(DOWNLOAD_DIR)
	mv $(DOWNLOAD_DIR)/$(EXTRACTED_DIR) $(DOWNLOAD_DIR)
