docker_tag = xhiroga/jekyll-test

docker-run: docker-build;
	docker run -it --rm --volume="$$(pwd):/srv/jekyll" -p 4000:4000 $(docker_tag) jekyll serve --incremental --trace

docker-build:
	docker build . -t $(docker_tag);
