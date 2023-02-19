docker_tag = xhiroga/aws_batch

docker-run: docker-build;
	docker run -it --rm $(docker_tag) -l /usr/local/init.sh

# チュートリアルでは`docker commit`で作成していたが、再現性のためにDockerfile化している。
docker-build:
	docker build . -t $(docker_tag);
