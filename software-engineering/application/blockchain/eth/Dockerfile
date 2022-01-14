# docker build -t ethereum .
# docker run -it ethereum

FROM ubuntu
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:ethereum/ethereum \
    && apt-get update \
    && apt-get install -y ethereum curl vim \
    && rm -rf /var/lib/apt/lists/*
RUN curl https://gist.githubusercontent.com/hiroga-cc/1e7ae56f94ea301e9b585663d275ccc4/raw/cdbfa8ee0726725dc60bbd6591edefc932b10568/.vimrc > ~/.vimrc
CMD ["bash"]