FROM python:alpine

RUN mkdir /root/work
WORKDIR /root/work # 指定したディレクトリに切り替える。その際にディレクトリがなければ新規作成する。
COPY src/ ./

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["app.py"]

# command
# docker build -t server .
# docker run -p 5000:8080 -it server