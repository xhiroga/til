# docker build -f alpine:apk-curl-git-openssh.Dockerfile -t alpine:apk-curl-git-openssh .
FROM alpine:latest

RUN apk add git curl openssh
