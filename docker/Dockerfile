FROM ubuntu:20.04
RUN apt-get update && apt-get -y install cmake protobuf-compiler 
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN apt-get -y install fontconfig-config
RUN apt-get -y install git
RUN apt-get -y install wget
RUN apt-get update && apt-get -qq install libgomp1 -y
RUN pip3 -q install pip --upgrade

COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

RUN mkdir -p src
ADD . src
WORKDIR /src

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

WORKDIR /src
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
