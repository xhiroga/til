FROM python:alpine

RUN apk add git
RUN pip install awscli
RUN pip install boto3

ADD init.sh /usr/local/
ENTRYPOINT [ "/bin/ash" ]
