ZIP_URL := https://github.com/ppgee/cocos-pnp/releases/download/playable-ads-adapter-1.3.10/playable-3x.zip
MARKER := extensions/playable-ads-adapter/.installed

.PHONY: cocos-pnp-build
extensions/playable-ads-adapter: $(MARKER)

cocos-pnp-build:
	node cocos-pnp-build.js --version 3 --buildPath build

$(MARKER):
	@mkdir -p temp
	curl -L -o temp/playable-3x.zip $(ZIP_URL)
	unzip temp/playable-3x.zip -d temp
	rm -rf extensions/playable-ads-adapter
	mv temp/playable-ads-adapter extensions/
	rm -rf temp
	touch $(MARKER)
