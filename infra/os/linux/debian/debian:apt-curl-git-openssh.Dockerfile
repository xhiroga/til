# docker build -f debian:apt-curl-git-openssh.Dockerfile -t debian:apt-curl-git-openssh .
FROM debian:latest

RUN apt update && apt upgrade -y
RUN apt install -y sudo
RUN sudo apt install -y curl git openssh-client
