.PHONY: docker-compose-up;

GPT3Discord:
	git clone https://github.com/Kav-K/GPT3Discord $@ || git -C $@ pull

.env:
	op inject --force -i .env.tpl -o $@

docker-build:
	docker build -t ghcr.io/xhiroga/gpt3discord ./GPT3Discord

containers/gpt3discord/share:
	mkdir -p $@

docker-compose-up: GPT3Discord .env containers/gpt3discord/share;
	docker-compose -f docker-compose-m1.yml up
