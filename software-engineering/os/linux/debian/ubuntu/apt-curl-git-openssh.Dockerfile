# docker build -f ubuntu:apt-curl-git-openssh.Dockerfile -t ubuntu:apt-curl-git-openssh .
FROM ubuntu:latest

RUN apt update && apt upgrade -y
RUN apt install -y sudo
RUN sudo apt install -y curl git openssh-client
