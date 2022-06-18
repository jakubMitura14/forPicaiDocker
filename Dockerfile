

## { Mitura start}
#FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

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
RUN apt-get install -y nodejs
RUN apt-get install -y libssl-dev



## installing github CLI - https://github.com/cli/cli/blob/trunk/docs/install_linux.md
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
RUN sudo apt update
RUN sudo apt install gh

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
RUN apt install -y zip
RUN apt install -y p7zip-full
RUN apt install -y apt-utils
RUN apt install -y octave
RUN apt install -y kmod
RUN apt install -y octave
RUN apt install -y zlib1g
RUN apt-get install -y python-dev

#from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu-jupyter.Dockerfile

ARG CUDA=11.3
ARG CUDNN=8.1.0.77-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=7.2.2-1
ARG LIBNVINFER_MAJOR_VERSION=7

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update 
RUN  apt-get update && apt-get install -y --no-install-recommends build-essential 
RUN  apt-get update && apt-get install -y --no-install-recommends cuda-command-line-tools-11.3 
RUN  apt-get update && apt-get install -y --no-install-recommends libcublas-11.3 
RUN  apt-get update && apt-get install -y --no-install-recommends cuda-nvrtc-11.3 
RUN  apt-get update && apt-get install -y --no-install-recommends libcufft-11.3 
RUN  apt-get update && apt-get install -y --no-install-recommends libcurand-11.3 
RUN  apt-get update && apt-get install -y --no-install-recommends libcusolver-11.3 
RUN  apt-get update && apt-get install -y --no-install-recommends libcusparse-11.3 
RUN  apt-get update && apt-get install -y --no-install-recommends curl 
#RUN  apt-get update && apt-get install -y --no-install-recommends libcudnn8  #=8.1.0.77-1+cuda11.3

RUN  apt-get update && apt-get install -y --no-install-recommends libfreetype6-dev 
RUN  apt-get update && apt-get install -y --no-install-recommends libhdf5-serial-dev 
RUN  apt-get update && apt-get install -y --no-install-recommends libzmq3-dev 
RUN  apt-get update && apt-get install -y --no-install-recommends pkg-config 
RUN  apt-get update && apt-get install -y --no-install-recommends software-properties-common 
RUN  apt-get update && apt-get install -y --no-install-recommends  unzip


RUN apt-get install -y bzip2
RUN apt-get install -y cmake

# from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb
# RUN mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/7fa2af80.pub
# RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
# RUN apt-get update
# RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
# RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}



# Install TensorRT if not building for PowerPC
# NOTE: libnvinfer uses cuda11.1 versions
# RUN  apt-get update && \
#         apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub && \
#         echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /"  > /etc/apt/sources.list.d/tensorRT.list && \
#         apt-get update && \
#         apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda11.0 \
#         libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda11.0 \
#         && apt-get clean \
#         && rm -rf /var/lib/apt/lists/*; 

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda-11.0/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8


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
RUN mkdir ${HOME}/data/preprocess
RUN mkdir ${HOME}/preprocess/monai_persistent_Dataset
RUN mkdir ${HOME}/output
RUN mkdir ${HOME}/piCaiCode
RUN mkdir ${HOME}/build
RUN mkdir ${HOME}/lightning_logs
RUN mkdir ${HOME}/preprocess/standarizationModels
RUN mkdir ${HOME}/preprocess/Bias_field_corrected
# COPY bashrc .
# COPY bashrc /etc/bash.bashrc
# RUN chown ${NB_USER} /etc/bash.bashrc
#RUN chmod a+rwx /etc/bash.bashrc



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
RUN chown ${NB_USER} ${HOME} ${HOME}/labels
RUN chown ${NB_USER} ${HOME} ${HOME}/data/preprocess
RUN chown ${NB_USER} ${HOME} ${HOME}/output
RUN chown ${NB_USER} ${HOME} ${HOME}/piCaiCode
RUN chown ${NB_USER} ${HOME} ${HOME}/build
RUN chown ${NB_USER} /var/lib/dpkg
RUN chown ${NB_USER} ${HOME} ${HOME}/lightning_logs
RUN chown ${NB_USER} ${HOME} ${HOME}/preprocess/monai_persistent_Dataset
RUN chown ${NB_USER} ${HOME} ${HOME}/preprocess/standarizationModels
RUN chown ${NB_USER} ${HOME} ${HOME}/preprocess/Bias_field_corrected








#used for logging lightning



USER ${NB_USER}

RUN mkdir /tmp/runtime-sliceruser
ENV XDG_RUNTIME_DIR=/tmp/runtime-sliceruser

################################################################################
# Set up remote desktop access - step 2/2

# First upgrade pip
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install --upgrade pip
# from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu-jupyter.Dockerfile
# RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip --no-cache-dir install --upgrade \
#     "pip<20.3" \
#     setuptools 


# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}




RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install --no-cache-dir jupyter matplotlib
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
# Pin jedi; see https://github.com/ipython/ipython/issues/12740
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2
#RUN jupyter serverextension enable --py jupyter_http_over_ws

#RUN apt-get autoremove -y && apt-get remove -y wget



##picai specific
#krowa

#download Pi cai data
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install zenodo_get==1.3.4


##picai specific end




# Now install websockify and jupyter-server-proxy (fixed at tag v1.6.0)
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install --upgrade websockify && \
    cp /usr/lib/rebind.so /home/sliceruser/Slicer/lib/Python/lib/python3.9/site-packages/websockify/ && \
    /home/sliceruser/Slicer/bin/PythonSlicer -m pip install notebook jupyterhub jupyterlab && \
    /home/sliceruser/Slicer/bin/PythonSlicer -m pip install -e \
    git+https://github.com/lassoan/jupyter-desktop-server#egg=jupyter-desktop-server \
    git+https://github.com/jupyterhub/jupyter-server-proxy@v1.6.0#egg=jupyter-server-proxy

####### ## { Mitura start} ading libraries through pip 

# RUN apt-get install -y software-properties-common
# RUN add-apt-repository ppa:ubuntu-toolchain-r/test
# RUN apt install -y gcc-7 g++-7

COPY requirements-dev.txt /tmp/

ENV PATH=$PATH:'/home/sliceruser/Slicer/lib/Python/bin'



#RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install pyradiomics

RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# tensorflow

# RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
# RUN Miniconda3-latest-Linux-x86_64.sh
# RUN conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/



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
# perform some standarization and bias field correction




##############3 code from picai hosts
# #install prepared preprocessing and evaluation ready libraries
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install git+https://github.com/DIAGNijmegen/picai_prep
RUN /home/sliceruser/Slicer/bin/PythonSlicer -m pip install git+https://github.com/DIAGNijmegen/picai_eval
#host segmentations and baselines
RUN git clone https://github.com/DIAGNijmegen/AbdomenMRUS-prostate-segmentation.git ${HOME}/picaiHost/picaiHostSegmentation
RUN git clone https://github.com/DIAGNijmegen/picai_baseline.git ${HOME}/picaiHost/picaiHostBaseline






# # #### download picai files and do some introductory preprocessing
# RUN /home/sliceruser/Slicer/bin/PythonSlicer -m zenodo_get --retry=8 10.5281/zenodo.6517397

# #prepare  csv containing metadata and paths
COPY processMetaData.py .
# RUN /home/sliceruser/Slicer/bin/PythonSlicer processMetaData.py
# # we already unpacked files now we can remove zips
# RUN rm /home/sliceruser/picai_public_images_fold0.zip 
# RUN rm /home/sliceruser/picai_public_images_fold1.zip 
# RUN rm /home/sliceruser/picai_public_images_fold2.zip 
# RUN rm /home/sliceruser/picai_public_images_fold3.zip 
# RUN rm /home/sliceruser/picai_public_images_fold4.zip
COPY standardize.py .
#RUN /home/sliceruser/Slicer/bin/PythonSlicer standardize.py



#login to github cli 
# COPY mytoken.txt .
# RUN gh auth login --with-token < mytoken.txt
# RUN git config --global user.name "Jakub Mitura"
# RUN git config --global user.email "jakub.mitura14@gmail.com"
# RUN git config -l






#copy main repository inside image
RUN git clone https://github.com/jakubMitura14/piCaiCode.git ${HOME}/piCaiCode



#USER root

#for simple elastix from https://github.com/Emanoel-sabidussi/SimpleElastixWorkshop


#RUN /home/sliceruser/Slicer/bin/PythonSlicer testBaselin.py

#Pi-Cai specific end

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

