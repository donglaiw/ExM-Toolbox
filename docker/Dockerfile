FROM ubuntu:latest
RUN apt-get update && apt-get -y install cmake protobuf-compiler #update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN apt-get -y install git
RUN pip3 -q install pip --upgrade

COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

RUN mkdir src
ADD . src
WORKDIR /src

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

RUN cd $HOME; \
    git clone https://github.com/SuperElastix/SimpleElastix; \
    mkdir build; \
    cd build; \
    cmake ../SimpleElastix/SuperBuild; \
    make -j `nproc`; \
    cd $HOME/build/SimpleITK-build/Wrapping/Python; \
    python3 Packaging/setup.py install; \
    cd $HOME; \
    git clone https://bitbucket.org/e_sabidussi/simpleelastix-workshop.git --depth 1
#WORKDIR /
#RUN git clone https://github.com/SuperElastix/SimpleElastix
#RUN mkdir /build
#RUN cd /build
#RUN cd build/ cmake ../SimpleElastix/SuperBuild
#RUN cd build/ make -j4
#RUN cd /build;cmake -D PYTHON_EXECUTABLE=`which python3` ../SimpleElastix/SuperBuild
#RUN cd /build;make -j `nproc`
#RUN cd /build/SimpleITK-build/Wrapping/Python/Packaging;python3 setup.py install

WORKDIR /src
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
