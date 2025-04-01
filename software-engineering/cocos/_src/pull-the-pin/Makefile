ZIP_URL := https://github.com/ppgee/cocos-pnp/releases/download/playable-ads-adapter-1.3.10/playable-3x.zip
MARKER := extensions/playable-ads-adapter/.installed

DEFAULT_PROJECT_NAME := project-$(shell date +'%Y%m%d-%H%M%S')
PROJECT_NAME ?= $(DEFAULT_PROJECT_NAME)

.PHONY: cocos-pnp-build clean-projects clone-project
extensions/playable-ads-adapter: $(MARKER)

cocos-pnp-build:
	node cocos-pnp-build.js --version 3 --buildPath build

clean-projects:
	rm -rf projects

# プロジェクトを指定された名前 (PROJECT_NAME) またはデフォルト名で projects/ 以下にクローンする
# 使用例:
#   make clone-project                   # デフォルト名 (例: projects/project-20250401-113000) でクローン
#   make clone-project PROJECT_NAME=my-app # projects/my-app にクローン
#   PROJECT_NAME=my-app make clone-project # 環境変数で指定してもOK
clone-project:
	@mkdir -p projects/$(PROJECT_NAME)
	rsync -av --exclude=build --exclude mcp-server --exclude projects --exclude .git --exclude .vscode . projects/$(PROJECT_NAME)/

$(MARKER):
	@mkdir -p temp
	curl -L -o temp/playable-3x.zip $(ZIP_URL)
	unzip temp/playable-3x.zip -d temp
	rm -rf extensions/playable-ads-adapter
	mv temp/playable-ads-adapter extensions/
	rm -rf temp
	touch $(MARKER)
