docker_tag = xhiroga/solidity

docker-run: docker-build;
	docker run -it --rm --volume="$$(pwd):/app" $(docker_tag)

docker-build:
	docker build . -t $(docker_tag);
