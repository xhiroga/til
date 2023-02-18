FROM ubuntu
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:ethereum/ethereum \
    && apt-get update \
    && apt-get install -y ethereum curl vim \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
