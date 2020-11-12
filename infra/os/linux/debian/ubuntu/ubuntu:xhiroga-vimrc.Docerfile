# docker build -t ubuntu:xhiroga-vimrc -f ubuntu:xhiroga-vimrc.Docerfile .

FROM ubuntu
RUN apt update && apt upgrade -y
RUN apt install -y sudo curl binutils build-essential strace sysstat vim && rm -rf /var/lib/apt/lists/*
RUN curl https://gist.githubusercontent.com/xhiroga/1e7ae56f94ea301e9b585663d275ccc4/raw/cdbfa8ee0726725dc60bbd6591edefc932b10568/.vimrc > ~/.vimrc
CMD ["bash"]
