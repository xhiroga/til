# THIS makefile save chromedriver in current path, but the script use chromedriver in PATH
CHROME_URL = https://chromedriver.storage.googleapis.com/87.0.4280.20/chromedriver_mac64.zip
CHROME_ZIP = chromedriver_mac64.zip
CHROME = chromedriver

.PHONY: all;
all: $(CHROME)

$(CHROME): $(CHROME_ZIP);
	unzip $(CHROME_ZIP)

$(CHROME_ZIP):
	wget $(CHROME_URL) -O $(CHROME_ZIP)
