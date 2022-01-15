STEAM = steam
STEAM_URL = https://www.dropbox.com/sh/lms10j4jr5gysu3/AAAqwDF32crdPBx7930g8AlZa?dl=0
INNER_ZIP = inner

.PHONY: all;
all: $(STEAM);

$(STEAM).zip:
	wget $(STEAM_URL) -O $(STEAM).zip

$(STEAM): ${STEAM}.zip
	unzip $(STEAM).zip -x / -d $(INNER_ZIP)
	unzip $(INNER_ZIP)/$(STEAM).zip -x / -d $(STEAM)