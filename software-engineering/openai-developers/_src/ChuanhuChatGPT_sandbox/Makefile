ChuanhuChatGPT:
	git -C $@ pull || git clone https://github.com/GaiZhenbiao/ChuanhuChatGPT $@

ChuanhuChatGPT/config.json: ChuanhuChatGPT;
	op inject --force -i config.json.tpl -o $@

docker-build: ChuanhuChatGPT;
	docker build -t ghcr.io/xhiroga/chuanhu-chat-gpt ./ChuanhuChatGPT

docker-compose-up: ChuanhuChatGPT ChuanhuChatGPT/config.json docker-build;
	docker-compose up
