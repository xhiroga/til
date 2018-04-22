# docker build -t ubuntu_test .
# docker run -it --security-opt seccomp:unconfined ubuntu_test

FROM ubuntu
RUN apt-get update && apt-get install -y sudo curl binutils build-essential strace sysstat vim && rm -rf /var/lib/apt/lists/*
RUN curl https://gist.githubusercontent.com/hiroga-cc/1e7ae56f94ea301e9b585663d275ccc4/raw/cdbfa8ee0726725dc60bbd6591edefc932b10568/.vimrc > ~/.vimrc
CMD ["bash"]