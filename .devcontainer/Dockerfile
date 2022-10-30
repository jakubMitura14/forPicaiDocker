
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04





ENV DEBIAN_FRONTEND=noninteractive
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    wget\
    manpages-dev\
    g++\
    gcc\
    nodejs\
    libssl-dev\
    unzip\
    #cuda-11.3\
    #nvidia-cuda-toolkit-11-3\
    && rm -rf /var/lib/apt/lists/*
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda




RUN apt update
RUN apt update && apt upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.9
RUN apt install -y python3-pip
# RUN apt-get -y purge python3.9
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

RUN update-alternatives --set python /usr/bin/python3.9


RUN python3.9 -m pip install --upgrade pip

# set up user
ENV NB_USER sliceruser
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# RUN python3.9 -m pip install pyzmq
RUN apt-get install python-dev -y
RUN apt-get install python3.9-dev -y
RUN apt install -y libssl1.1
RUN apt install -y libssl-dev
RUN apt install -y libmysqlclient-dev
# RUN python3.9 -m pip install --no-cache-dir jupyter matplotlib
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
# Pin jedi; see https://github.com/ipython/ipython/issues/12740
# RUN python3.9 -m  pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2
RUN python3.9 -m  pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ENV JAXLIB_VERSION=0.3.16
RUN apt-get install -y python3-dev default-libmysqlclient-dev build-essential
#RUN    pip install pyradiomics
RUN python3.9 -m  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN python3.9 -m  pip install mysqlclient
# libmysqlclient \
#                         libmysqlclient-dev \
#                             mysql-server
#mysql-server \
RUN python3.9 -m  pip install \
    #itk \
    pytorch-ignite==0.4.8 \
    PyWavelets==1.3.0 \
    scipy==1.8.1 \
    nibabel==3.2.2 \
    pillow!=8.3.0 \  
    scikit-image==0.19.2 \
    tqdm>=4.47.0 \
    lmdb==1.3.0 \
    flake8>=3.8.1 \
    flake8-bugbear \
    flake8-comprehensions \
    flake8-executable \
    flake8-pyi \
    pylint!=2.13 \
    mccabe \
    pep8-naming \
    pycodestyle \
    pyflakes==2.4.0 \
    black \
    isort \
    pytype>=2020.6.1 \
    types-pkg_resources \
    mypy>=0.790 \
    ninja==1.10.2.3 \
    #torchvision==0.12.0 \
    opencv-python==4.5.5.64 \
    psutil==5.9.1 \
    Sphinx==3.5.3 \
    recommonmark==0.6.0 \
    sphinx-autodoc-typehints==1.11.1 \
    sphinx-rtd-theme==0.5.2 \
    # cucim==22.2.1; platform_system == "Linux" \
    # imagecodecs; platform_system == "Linux" \
    # tifffile; platform_system == "Linux" \
    pandas==1.4.2 \
    requests==2.27.1 \
    einops==0.4.1 \
    transformers==4.19.2 \
    mlflow==1.26.1 \
    matplotlib!=3.5.0 \
    types-PyYAML \
    pyyaml==6.0 \
    fire==0.4.0 \
    jsonschema==4.6.0 \
    kymatio==0.2.1 \
    juliacall==0.8.0 \
    jupyterlab_github==3.0.1 \
    torchio==0.18.77 \
    git+https://github.com/sat28/githubcommit.git \
    gdown==4.4.0 \
    seaborn==0.11.2 \
    # PyTorchUtils==0.0.3 \
    # optuna==2.10.0 \
    jupyterlab-git==0.37.1 \
    comet-ml==3.31.5 \
    # voxelmorph==0.2 \
    # tensorflow==2.9.1 \
    # tensorflow_addons==0.17.0 \
    ipywidgets==7.7.0 \
    h5py==3.7.0 \
    itk-elastix==0.14.1 \
    zenodopy==0.2.0 \
    evalutils==0.3.0 \
    nnunet==1.7.0 \
    SimpleITK \
    git+https://github.com/balbasty/nitorch#egg=nitorch[all] \
    threadpoolctl==3.1.0 \
    batchgenerators==0.24 \
    # zenodo_get==1.3.4 \
    git+https://github.com/DIAGNijmegen/picai_baseline \
    git+https://github.com/DIAGNijmegen/picai_prep \
    git+https://github.com/DIAGNijmegen/picai_eval \
    #picai_prep==2.0.1 \
    #https://github.com/DIAGNijmegen/picai_prep/archive/refs/tags/v1.3.2.zip \
    KevinSR==0.1.19 \
    dask==2022.6.0 \
    intensity-normalization[ants] \
    # numba==0.55.2 \
    # pyro-ppl==1.8.1 \
    # warp-lang==0.2.3 \
    report_guided_annotation \
    SimpleITK-SimpleElastix \
    geomloss[full] \
    torch-optimizer \ 
    optax \ 
    ott-jax \ 
    evojax[extra] \
    jraph \
    flax \
    toolz \
    ipyparallel \
    itk  \
    SimpleITK \
    openpyxl==3.0.10 \
    requests \
    pytorch-lightning==1.7.7 \
    matplotlib 

#    pytorch-lightning==1.7.7 \


# RUN python3.9 -m  pip install picai_prep==2.0.1





RUN python3.9 -m  pip install 'monai[nibabel, skimage, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, lmdb, psutil, cucim, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs]'
# RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
# RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html \

RUN python3.9 -m  pip install optuna

RUN git clone https://github.com/google-research/scenic.git ${HOME}/build
RUN cd ${HOME}/build/scenic
RUN python3.9 -m  pip install ${HOME}/build
RUN python3.9 -m pip install --upgrade jax jaxlib

# RUN python3.9 -m pip install pytorch-lightning-bolts 
#ENV CUDAToolkit_ROOT=/usr/local/cuda-11.3 


# RUN apt-get install python3.9-dev -y
RUN python3.9 -m pip install --no-cache-dir 'monai[nibabel, skimage, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, lmdb, psutil, cucim, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs]'

#host segmentations and baselines
RUN git clone https://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@github.com/jakubMitura14/piCaiCode.git ${HOME}/data/piCaiCode

RUN git clone https://github.com/DIAGNijmegen/AbdomenMRUS-prostate-segmentation.git ${HOME}/externalRepos/picaiHostSegmentation
RUN git clone https://github.com/DIAGNijmegen/picai_baseline.git ${HOME}/externalRepos/picaiHostBaseline
RUN git clone https://github.com/brudfors/UniRes.git ${HOME}/externalRepos/uniRes
RUN git clone https://github.com/neel-dey/Atlas-GAN.git ${HOME}/externalRepos/conditionalAtlasGAN
RUN git clone https://github.com/SuperElastix/SimpleElastix ${HOME}/externalRepos/elastix


# RUN python3.9 -m pip install --no-cache-dir intensity-normalization

#USER root

# ENV PATH="/usr/lib/python3.9/lib-dynload:${PATH}"
# ENV PATH="/usr/local/lib/python3.9/dist-packages:${PATH}"
# ENV PYTHONPATH=$PYTHONPATH:'/usr/lib/python3.9'
# ENV PYTHONHOME=$PYTHONHOME:'/usr/lib/python3.9'


RUN mkdir ${HOME}/scripts

#RUN git clone https://github.com/SuperElastix/SimpleElastix ${HOME}/externalRepos/elastix

# RUN cd ${HOME}/externalRepos/elastix && \
#     mkdir buildd && \
#     cd buildd && \
#     cmake ${HOME}/externalRepos/elastix/SuperBuild && \
#     make -j10 


# RUN cp ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python/_SimpleITK.so ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python/Packaging
# RUN cp ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python/_SimpleITK.so /home/sliceruser/

# RUN python3.9 ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python/Packaging/setup.py install

# RUN python3.9 -m pip install SimpleITK-SimpleElastix

#maybe remove if registering?
# RUN python3.9 -m pip install --no-cache-dir itk  \
#                                             SimpleITK \
#                                             openpyxl==3.0.10 \
#                                             requests \
#                                             pytorch-lightning==1.7.7 \
#                                             matplotlib

# ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python/_SimpleITK.so

# RUN cd ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python && \
#     python3 Packaging/setup.py install




# git clone https://github.com/SuperElastix/elastix.git ${HOME}/externalRepos/elastixBase
# cd ${HOME}/externalRepos/elastixBase

# https://github.com/SuperElastix/elastix.git


RUN ["chmod", "+x", "/home/sliceruser/externalRepos/elastix"]
#from https://installati.one/ubuntu/22.04/libpangoxft-1.0-0/
RUN apt update
RUN apt -y install libtbb-dev
#from https://askubuntu.com/questions/409310/error-while-loading-shared-libraries-cannot-open-shared-object-file-no-such-fi
RUN apt -y install libgtk-3-dev
RUN apt update

RUN apt-get -y install libidn11-dev
RUN apt-get -y install libglu1-mesa-dev
RUN apt-get -y install libpangox-1.0-dev
RUN apt-get -y install libpangoxft-1.0-0

RUN mkdir ${HOME}/elastixBase && \
cd ${HOME}/elastixBase && \
wget https://github.com/SuperElastix/elastix/releases/download/5.0.1/elastix-5.0.1-linux.tar.bz2&& \
tar -xf elastix-5.0.1-linux.tar.bz2

RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so ${HOME}/elastixBase/elastix-5.0.1-linux/bin/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 ${HOME}/elastixBase/elastix-5.0.1-linux/bin/libANNlib-5.0.so.1
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so ${HOME}/elastixBase/elastix-5.0.1-linux/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 ${HOME}/elastixBase/elastix-5.0.1-linux/libANNlib-5.0.so.1
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so /usr/local/lib/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 /usr/local/lib/libANNlib-5.0.so.1

RUN ldconfig
RUN apt-get install -q -y vim nano 




EXPOSE $VNCPORT $JUPYTERPORT

# ENTRYPOINT ["/home/sliceruser/run.sh"]

# CMD ["sh", "-c", "./Slicer/bin/PythonSlicer -m jupyter notebook --port=$JUPYTERPORT --ip=0.0.0.0 --no-browser --NotebookApp.default_url=/lab/"]

#login to github cli 
# COPY mytoken.txt .
# RUN gh auth login --with-token < mytoken.txt
RUN git config --global user.name "Jakub Mitura"
RUN git config --global user.email "jakub.mitura14@gmail.com"
RUN git config -l

#instal git for large files
RUN build_deps="curl" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ${build_deps} ca-certificates && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git-lfs && \
    git lfs install && \
    DEBIAN_FRONTEND=noninteractive apt-get purge -y --auto-remove ${build_deps} && \
    rm -r /var/lib/apt/lists/*

RUN git config --global --add safe.directory /home/sliceruser/locTemp/picai_baseline


#CMD ["/bin/bash"]
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


# USER root 
COPY start.sh .
RUN ["chmod", "+x", "start.sh"]
RUN chown ${NB_USER} start.sh

# RUN git clone https://github.com/jakubMitura14/picai_baseline.git 
# RUN git switch OPI


# CMD /bin/bash start.sh
#ENTRYPOINT [ "/bin/bash","-l", "-c" ]
ENTRYPOINT [ "/bin/bash", "start.sh","-l", "-c" ]



#  RUN      python3.9 -m zenodo_get --retry=8 10.5281/zenodo.6517397
