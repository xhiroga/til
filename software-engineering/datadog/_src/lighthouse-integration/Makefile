all: docker-build run;

docker-build:
	docker build . -t xhiroga/datadog-lighthouse-ingegration;

run:
	export $$(cat .env) && docker run -e DD_API_KEY -v $$(pwd)/lighthouse.d:/conf.d/lighthouse.d -it --rm xhiroga/datadog-lighthouse-ingegration bash
