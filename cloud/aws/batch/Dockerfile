FROM python:alpine

RUN apk add git
RUN pip install awscli
RUN pip install boto3

# ashにおけるSystem profile。ある場合は起動時に読み込む。
ADD profile /etc/

ADD init.sh /usr/local/

# command
# docker build -t aws_batch .
# docker images
# docker run -it aws_batch /bin/ash -l
# docker run -it aws_batch /bin/ash -l /usr/local/init.sh

# チュートリアルでは`docker commit`で作成していたが、再現性のためにDockerfile化している。