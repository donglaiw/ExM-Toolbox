### About

Link: https://drive.google.com/file/d/1uAjpC5M0skeW0Wc29D9yvv-ksBe7VWrL/view?usp=sharing

This link contains a .zip folder holding a Dockerfile, requirements.txt file and a source directory holding larvae data and a Jupyter notebook. 

### Running on a Local Machine
To run it on your local machine, 

1. Download Docker and create a Docker account. 
2. [**Skip to this step if you already have Docker downloaded**] Download the .zip file in this repository, unzip it, and navigate to the resulting folder in your terminal. 
3. Make sure Docker is running in the background and run the following commands:

```docker build -t example .```

```docker run -p 8888:8888 example```

Finally, copy and paste one of the URLs into your browser. This will launch you into the Jupyter notebook.

### Running on a Cluster
To run it on a cluser, you'll need to download Singularity. Because Docker requires root priveleges, Singularity is preferred on shared machines. 

First, we install the Singularity dependencies. After logging into the MIT, BC, or Harvard server, run:

```conda install yum update -y &&      conda yum groupinstall -y 'Development Tools' &&      conda yum install -y      openssl-devel      libuuid-devel         libseccomp-devel      wget      squashfs-tools      cryptsetup```

Then, 

```conda install go```

Next, we clone the Singularity GitHub and install it using the following two commands:

```git clone https://github.com/sylabs/singularity.git && cd singularity && git checkout v3.5.2```

```g./mconfig --without-setuid --prefix=/home/dave/singularity && make -C ./builddir && make -C ./builddir install```

[**Skip to this step if you already have Singularity downloaded**] Finally, download the .zip file in this repository, unzip it, and navigate to the resulting folder. Run it with Singularity using the following commands:

```singularity build dockertest```

```singularity shell dockertest```
