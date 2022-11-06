FROM python:3.11.0-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get -y update && apt-get upgrade -qqy && apt-get -y install \
    bash \
    curl \
    gcc \
    git \
    jq

# Apply https://github.com/pypa/setuptools/pull/3108
RUN pip install --upgrade setuptools

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH /root/.local/bin:$PATH
