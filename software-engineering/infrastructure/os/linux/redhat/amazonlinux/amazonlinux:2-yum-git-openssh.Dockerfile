# docker build -f amazonlinux:2-yum-git-openssh.Dockerfile -t amazonlinux:2-yum-git-openssh .
FROM amazonlinux:2022

RUN yum update -y
RUN yum install -y git openssh openssh-clients
