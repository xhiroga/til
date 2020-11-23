LECTURE_11 = resources/css_html_file
LECTURE_11_URL = https://www.dropbox.com/s/bqrcr7a7vln0qwq/css_html_file.zip?dl=0
LECTURE_14 = resources/xpath_html_file
LECTURE_14_URL = https://www.dropbox.com/s/zk2dbqr2b1talms/xpath_html_file.zip?dl=0
WORLDOMETERS = resources/worldmeters
WORLDOMETERS_URL = https://www.dropbox.com/sh/p54jjns5rrchp4x/AACuEfYg8agYXLWX8IuTTRpia?dl=0
IMDB = resources/imdb
IMDB_URL = https://www.dropbox.com/sh/a4q7z0sxsrnmjtb/AAAGsi17Enb2LXzWbjWdZL2Pa?dl=0
BOOKSTOSCRAPE = resources/bookstoscrape
BOOKSTOSCRAPE_URL = https://www.dropbox.com/sh/a8anjg5z1oinxhv/AABWn-1nH-gJ7FMc6aoTrLy0a?dl=0


.PHONY: all $(LECTURE_11) $(LECTURE_14) $(WORLDOMETERS)
all: $(LECTURE_11) $(LECTURE_14) $(WORLDOMETERS)

$(LECTURE_11): $(LECTURE_11).zip
	unzip $(LECTURE_11).zip -d $(LECTURE_11)

$(LECTURE_11).zip:
	wget $(LECTURE_11_URL) -O $(LECTURE_11).zip

$(LECTURE_14): $(LECTURE_14).zip
	unzip $(LECTURE_14).zip -d $(LECTURE_14)

$(WORLDOMETERS).zip:
	wget $(WORLDOMETERS_URL) -O $(WORLDOMETERS).zip

$(WORLDOMETERS): $(WORLDOMETERS).zip
	unzip $(WORLDOMETERS).zip -d $(WORLDOMETERS)

$(BOOKSTOSCRAPE).zip:
	wget $(BOOKSTOSCRAPE_URL) -O $(BOOKSTOSCRAPE).zip

$(BOOKSTOSCRAPE): $(BOOKSTOSCRAPE).zip
	unzip $(BOOKSTOSCRAPE).zip -d $(BOOKSTOSCRAPE)

