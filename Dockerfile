

## { Mitura start}
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

################################################################################
# Prevent apt-get from prompting for keyboard choice
#  https://superuser.com/questions/1356914/how-to-install-xserver-xorg-in-unattended-mode
ENV DEBIAN_FRONTEND=noninteractive

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list



# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    # build-essential \
    #cuda-11.3\
    #nvidia-cuda-toolkit-11-3\
 && rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y wget
RUN apt-get install -y manpages-dev
RUN apt-get install -y g++
RUN apt-get install -y gcc



# RUN add-apt-repository --remove ppa:fkrull/deadsnakes
# RUN apt-get update
# RUN apt-get remove --purge python


RUN apt autoremove python3 -y



# RUN add-apt-repository ppa:jonathonf/gcc-7.1
# RUN apt-get update
# RUN apt-get install gcc-7 g++-7


# RUN apt-get install -y software-properties-common
# RUN add-apt-repository ppa:ubuntu-toolchain-r/test
# RUN apt install -y gcc-7 g++-7



RUN mkdir /app
WORKDIR /app

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda-11.3

## { Mitura end}



################################################################################
# Remove documentation to save hard drive space
#  https://askubuntu.com/questions/129566/remove-documentation-to-save-hard-drive-space
COPY etc/dpkg/dpkg.cfg.d/01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc

################################################################################
# - update apt, set up certs, run netselect to get fast mirror
# - reduce apt gravity
# - and disable caching
#   from https://blog.sleeplessbeastie.eu/2017/10/02/how-to-disable-the-apt-cache/
# Note: in each RUN command "apt-get update" must be called before apt-get is used.
RUN echo 'APT::Install-Recommends "0" ; APT::Install-Suggests "0" ;' >> /etc/apt/apt.conf && \
    echo 'Dir::Cache::pkgcache "";\nDir::Cache::srcpkgcache "";' | tee /etc/apt/apt.conf.d/00_disable-cache-files

################################################################################
# get packages
# - update (without update, old cached server addresses may be used and apt-get install may fail)
# - install things
#   - basic  tools
#   - slicer dependencies
#   - awesome window manager
#   - install ca-certs to prevent - fatal: unable to access 'https://github.com/novnc/websockify/': server certificate verification failed. CAfile: none CRLfile: none
RUN apt-get update -q -y && \
    apt-get install -q -y \
        vim net-tools curl \
        libgl1-mesa-glx \
        xserver-xorg-video-dummy \
        libxrender1 \
        libpulse0 \
        libpulse-mainloop-glib0  \
        libnss3  \
        libxcomposite1 \
        libxcursor1 \
        libfontconfig1 \
        libxrandr2 \
        libasound2 \
        libglu1 \
        x11vnc \
        awesome \
        jq \
        git && \
    apt-get install -q -y --reinstall ca-certificates
RUN apt-get install -y nautilus
RUN apt install -y jupyter-core
RUN apt install -y unzip

################################################################################
# set up user
ENV NB_USER sliceruser
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
            --gecos "Default user" \
            --uid ${NB_UID} \
            ${NB_USER}

WORKDIR ${HOME}

##picai specific
RUN mkdir ${HOME}/data
RUN mkdir ${HOME}/labels
RUN mkdir ${HOME}/preprocess
RUN mkdir ${HOME}/output

RUN git clone https://github.com/DIAGNijmegen/picai_labels.git ${HOME}/labels

##picai specific end
################################################################################
# Download and unpack Slicer

# Current preview release (Slicer-4.13 2022-02-15, rev30607)

## Mitura modif to newer
ARG SLICER_DOWNLOAD_URL=https://download.slicer.org/bitstream/6286ce04e8408647b39f803a

# Use local package:
#ADD $SLICER_ARCHIVE.tar.gz ${HOME}/

# Download package:
RUN curl -JL "$SLICER_DOWNLOAD_URL" | tar xz -C /tmp && mv /tmp/Slicer* Slicer

################################################################################
# these go after installs to avoid trivial invalidation
ENV VNCPORT=49053
ENV JUPYTERPORT=8888
ENV DISPLAY=:10

COPY xorg.conf .

################################################################################
# Set up remote desktop access - step 1/2

# Build rebind.so (required by websockify)
RUN set -x && \
    apt-get update -q -y && \
    apt-get install -y build-essential --no-install-recommends && \
    mkdir src && \
    cd src && \
    git clone https://github.com/novnc/websockify websockify && \
    cd websockify && \
    make && \
    cp rebind.so /usr/lib/ && \
    cd .. && \
    rm -rf websockify && \
    cd .. && \
    rmdir src && \
    apt-get purge -y --auto-remove build-essential

# Set up launcher for websockify
# (websockify must run in  Slicer's Python environment)
COPY websockify ./Slicer/bin/
RUN chmod +x ${HOME}/Slicer/bin/websockify

################################################################################
# Need to run Slicer as non-root because
# - mybinder requirement
# - chrome sandbox inside QtWebEngine does not support root.
RUN chown ${NB_USER} ${HOME} ${HOME}/Slicer
RUN chown ${NB_USER} ${HOME} ${HOME}/data
RUN chown ${NB_USER} ${HOME} ${HOME}/labels
RUN chown ${NB_USER} ${HOME} ${HOME}/preprocess
RUN chown ${NB_USER} ${HOME} ${HOME}/output

USER ${NB_USER}

RUN mkdir /tmp/runtime-sliceruser
ENV XDG_RUNTIME_DIR=/tmp/runtime-sliceruser

################################################################################
# Set up remote desktop access - step 2/2

# First upgrade pip
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install --upgrade pip



##picai specific

#download Pi cai data
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install zenodo_get==1.3.4
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m zenodo_get --retry=8 10.5281/zenodo.6517397

# unzip and remove zipped files


##picai specific end


# Now install websockify and jupyter-server-proxy (fixed at tag v1.6.0)
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install --upgrade websockify && \
    cp /usr/lib/rebind.so /home/sliceruser/Slicer/lib/Python/lib/python3.9/site-packages/websockify/ && \
    /home/sliceruser/Slicer/bin/PythonSlicer -m pip install notebook jupyterhub jupyterlab && \
    /home/sliceruser/Slicer/bin/PythonSlicer -m pip install -e \
        git+https://github.com/lassoan/jupyter-desktop-server#egg=jupyter-desktop-server \
        git+https://github.com/jupyterhub/jupyter-server-proxy@v1.6.0#egg=jupyter-server-proxy

####### ## { Mitura start} ading libraries through pip 
COPY requirements-dev.txt /tmp/





#RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install pyradiomics

RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install --no-cache-dir -r /tmp/requirements-dev.txt

RUN /home/sliceruser/Slicer/bin/PythonSlicer -m ipykernel install --user


ENV PYTHONPATH "${PYTHONPATH}:/home/sliceruser/Slicer/bin/PythonSlicer"


## { Mitura end}


################################################################################
# Install Slicer extensions

COPY start-xorg.sh .
COPY install.sh .
RUN ./install.sh ${HOME}/Slicer/Slicer && \
    rm ${HOME}/install.sh

################################################################################
EXPOSE $VNCPORT $JUPYTERPORT
COPY run.sh .
ENTRYPOINT ["/home/sliceruser/run.sh"]

CMD ["sh", "-c", "./Slicer/bin/PythonSlicer -m jupyter notebook --port=$JUPYTERPORT --ip=0.0.0.0 --no-browser --NotebookApp.default_url=/lab/"]

################################################################################
# Install Slicer application startup script

COPY .slicerrc.py .



#install prepared preprocessing and evaluation ready libraries
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install git+https://github.com/DIAGNijmegen/picai_prep
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install git+https://github.com/DIAGNijmegen/picai_eval


RUN rm /home/sliceruser/picai_public_images_fold0.zip 
RUN rm /home/sliceruser/picai_public_images_fold1.zip 
RUN rm /home/sliceruser/picai_public_images_fold2.zip 
RUN rm /home/sliceruser/picai_public_images_fold3.zip 
RUN rm /home/sliceruser/picai_public_images_fold4.zip



################################################################################
# Build-time metadata as defined at http://label-schema.org
ARG BUILD_DATE
ARG IMAGE
ARG VCS_REF
ARG VCS_URL
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name=$IMAGE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url=$VCS_URL \
      org.label-schema.schema-version="1.0"







##/home/sliceruser/Slicer/Slicer

