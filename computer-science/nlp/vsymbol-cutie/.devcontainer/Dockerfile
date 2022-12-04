FROM python:3.7.6-slim
# tensorflow==1.12.0 does not support python:3.7.
# However, python3.6 container cannot find any tensorflow versions and python3.8 does not support tensorflow==1.x
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get -y update && apt-get upgrade -qqy && apt-get -y install \
    bash \
    curl \
    gcc \
    git \
    libfreetype6-dev \
    pkg-config \
    libpng-dev \
    # libpng12-dev から名前が変わったらしい。
    build-essential

# Apply https://github.com/pypa/setuptools/pull/3108
RUN pip install --upgrade setuptools
RUN /usr/local/bin/python -m pip install --upgrade pip
