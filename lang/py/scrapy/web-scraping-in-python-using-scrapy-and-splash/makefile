LECTURE_11 = resources/css_html_file
LECTURE_11_URL = https://www.dropbox.com/s/bqrcr7a7vln0qwq/css_html_file.zip?dl=0
LECTURE_14 = resources/xpath_html_file
LECTURE_14_URL = https://www.dropbox.com/s/zk2dbqr2b1talms/xpath_html_file.zip?dl=0

.PHONY: all $(LECTURE_11) $(LECTURE_14)
all: $(LECTURE_11) $(LECTURE_14)

$(LECTURE_11): $(LECTURE_11).zip
	unzip $(LECTURE_11).zip -d $(LECTURE_11)

$(LECTURE_11).zip:
	wget $(LECTURE_11_URL) -O $(LECTURE_11).zip

$(LECTURE_14): $(LECTURE_14).zip
	unzip $(LECTURE_14).zip -d $(LECTURE_14)

$(LECTURE_14).zip:
	wget $(LECTURE_14_URL) -O $(LECTURE_14).zip
