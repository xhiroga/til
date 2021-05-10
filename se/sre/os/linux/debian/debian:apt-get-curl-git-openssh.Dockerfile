# docker build -f debian:apt-get-curl-git-openssh.Dockerfile -t debian:apt-get-curl-git-openssh .
FROM debian:latest

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y sudo
RUN sudo apt-get install -y curl git openssh-client
