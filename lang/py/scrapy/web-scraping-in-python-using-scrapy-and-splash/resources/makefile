LECTURE_11 = css_html_file
LECTURE_11_URL = https://www.dropbox.com/s/bqrcr7a7vln0qwq/css_html_file.zip?dl=0
LECTURE_14 = xpath_html_file
LECTURE_14_URL = https://www.dropbox.com/s/zk2dbqr2b1talms/xpath_html_file.zip?dl=0
WORLDOMETERS = worldometers
WORLDOMETERS_URL = https://www.dropbox.com/sh/p54jjns5rrchp4x/AACuEfYg8agYXLWX8IuTTRpia?dl=0
IMDB = imdb
IMDB_URL = https://www.dropbox.com/sh/a4q7z0sxsrnmjtb/AAAGsi17Enb2LXzWbjWdZL2Pa?dl=0
BOOKSTOSCRAPE = bookstoscrape
BOOKSTOSCRAPE_URL = https://www.dropbox.com/sh/a8anjg5z1oinxhv/AABWn-1nH-gJ7FMc6aoTrLy0a?dl=0
SPLASH = livecoin
SPLASH_URL = https://www.dropbox.com/sh/ge4wjx9jbuzc7fo/AACeX57YEwGJhK8zRVTIBYAWa?dl=0
SELENIUM = silkdeal
SELENIUM_URL = https://www.dropbox.com/sh/bpjinzf7a9li46x/AAC6xsAEQwgR3XbZ6JdNFxIua?dl=0
PIPELINE = imdb-pipeline
PIPELINE_URL = https://www.dropbox.com/sh/qd0zio6pq0i5vqk/AABN_me2Zk7tPbl4jEj7Ym8ma?dl=0


.PHONY: all;
all: $(LECTURE_11) $(LECTURE_14) $(WORLDOMETERS) $(BOOKSTOSCRAPE) SPLASH;

$(LECTURE_11): $(LECTURE_11).zip
	unzip $(LECTURE_11).zip -d $(LECTURE_11)

$(LECTURE_11).zip:
	wget $(LECTURE_11_URL) -O $(LECTURE_11).zip

$(LECTURE_14): $(LECTURE_14).zip
	unzip $(LECTURE_14).zip -d $(LECTURE_14)

$(WORLDOMETERS).zip:
	wget $(WORLDOMETERS_URL) -O $(WORLDOMETERS).zip

$(WORLDOMETERS): $(WORLDOMETERS).zip
	unzip $(WORLDOMETERS).zip -x / -d $(WORLDOMETERS)
	unzip $(WORLDOMETERS)/$(WORLDOMETERS).zip -x / -d $(WORLDOMETERS)

$(BOOKSTOSCRAPE).zip:
	wget $(BOOKSTOSCRAPE_URL) -O $(BOOKSTOSCRAPE).zip

$(BOOKSTOSCRAPE): $(BOOKSTOSCRAPE).zip
	unzip $(BOOKSTOSCRAPE).zip -x / -d $(BOOKSTOSCRAPE)
	unzip $(BOOKSTOSCRAPE)/$(BOOKSTOSCRAPE).zip -x / -d $(BOOKSTOSCRAPE)

$(SPLASH).zip:
	wget $(SPLASH_URL) -O $(SPLASH).zip

$(SPLASH): ${SPLASH}.zip
	unzip $(SPLASH).zip -x / -d $(SPLASH)
	unzip $(SPLASH)/$(SPLASH).zip -x / -d $(SPLASH)

$(SELENIUM).zip:
	wget $(SELENIUM_URL) -O $(SELENIUM).zip

$(SELENIUM): ${SELENIUM}.zip
	unzip $(SELENIUM).zip -x / -d $(SELENIUM)
	unzip $(SELENIUM)/$(SELENIUM).zip -x / -d $(SELENIUM)

$(PIPELINE).zip:
	wget $(PIPELINE_URL) -O $(PIPELINE).zip

$(PIPELINE): ${PIPELINE}.zip
	unzip $(PIPELINE).zip -x / -d $(PIPELINE)
