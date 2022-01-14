# docker build -f centos:8-yum-git-openssh.Dockerfile -t centos:8-yum-git-openssh .
FROM centos:8

RUN yum update -y
RUN yum install -y git openssh openssh-clients
