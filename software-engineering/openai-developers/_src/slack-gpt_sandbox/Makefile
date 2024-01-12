slack-gpt:
	git -C $@ pull || git clone https://github.com/pyama86/slack-gpt $@

.env:
	op inject --force -i .env.tpl -o $@

docker-compose-up: slack-gpt .env;
	docker-compose up --build
