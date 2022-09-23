

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



## installing github CLI - https://github.com/cli/cli/blob/trunk/docs/install_linux.md
# RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
# RUN sudo apt update
# RUN sudo apt install gh
#RUN apt autoremove python3 -y


RUN mkdir /app
WORKDIR /app

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda-11.3
RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
RUN echo 'APT::Install-Recommends "0" ; APT::Install-Suggests "0" ;' >> /etc/apt/apt.conf && \
    echo 'Dir::Cache::pkgcache "";\nDir::Cache::srcpkgcache "";' | tee /etc/apt/apt.conf.d/00_disable-cache-files

#consider using venv https://pythonspeed.com/articles/activate-virtualenv-dockerfile/

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
    nautilus\
    jupyter-core\
    zip\
    p7zip-full\
    apt-utils\
    octave\
    kmod\
    zlib1g\
    python-dev\
    bzip2\
    cmake\
    cuda-command-line-tools-11.3 \
    libcublas-11.3 \
    cuda-nvrtc-11.3\
    libcufft-11.3 \
    libcurand-11.3 \
    libcusolver-11.3 \
    libcusparse-11.3 \
    libfreetype6-dev \
    curl\
    libzmq3-dev \
    pkg-config\
    software-properties-common\
    libhdf5-serial-dev\
    libtbb2 \
    #elastix=4.9.0-1\
    git && \
    apt-get install -q -y --reinstall ca-certificates

RUN add-apt-repository ppa:deadsnakes/ppa

#from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu-jupyter.Dockerfile

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





# Install some basic utilities
RUN mkdir \
    ${HOME}/data \
    ${HOME}/labels \
    ${HOME}/data/preprocess \
    ${HOME}/data/orig \
    ${HOME}/data/preprocess/monai_persistent_Dataset \
    ${HOME}/output \
    ${HOME}/data/piCaiCode \
    ${HOME}/build \
    ${HOME}/data/lightning_logs \
    ${HOME}/data/preprocess/standarizationModels \
    ${HOME}/data/preprocess/Bias_field_corrected \
    ${HOME}/data/metadata \
    ${HOME}/data/dummyData



RUN git clone https://github.com/DIAGNijmegen/picai_labels.git ${HOME}/labels


################################################################################
# these go after installs to avoid trivial invalidation
ENV VNCPORT=49053
ENV JUPYTERPORT=8888
ENV DISPLAY=:10

#COPY xorg.conf .

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
    apt install -y python3.8 && \
    apt install -y python3-pip && \
    #easy_install -y pip && \
    apt-get purge -y --auto-remove build-essential




RUN /sbin/ldconfig -v
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates gnupg2 less sudo \
 # Spicy build and test dependencies.
 && apt-get install -y --no-install-recommends git cmake ninja-build ccache bison flex g++ libfl-dev zlib1g-dev libssl-dev jq locales-all make \
 # Spicy doc dependencies.
 && apt-get install -y --no-install-recommends python3 python3-pip python3-sphinx python3-sphinx-rtd-theme python3-setuptools python3-wheel doxygen \
 && pip3 install "btest>=0.66" pre-commit \
 # Install a recent CMake.
 && mkdir -p /opt/cmake \
 && curl -L https://github.com/Kitware/CMake/releases/download/v3.18.0/cmake-3.18.0-Linux-x86_64.tar.gz | tar xzvf - -C /opt/cmake --strip-components 1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/cmake/bin:${PATH}"
ENV PATH="/usr/lib/python3.8:${PATH}"


################################################################################
# Need to run Slicer as non-root because
# - mybinder requirement
# - chrome sandbox inside QtWebEngine does not support root.
# RUN chown ${NB_USER} ${HOME} ${HOME}/Slicer
RUN chown ${NB_USER} ${HOME} ${HOME}/labels
RUN chown ${NB_USER} ${HOME} ${HOME}/data/preprocess
RUN chown ${NB_USER} ${HOME} ${HOME}/data/orig
RUN chown ${NB_USER} ${HOME} ${HOME}/output
RUN chown ${NB_USER} ${HOME} ${HOME}/data/piCaiCode
RUN chown ${NB_USER} ${HOME} ${HOME}/build
RUN chown ${NB_USER} /var/lib/dpkg
RUN chown ${NB_USER} ${HOME} ${HOME}/data/lightning_logs
RUN chown ${NB_USER} ${HOME} ${HOME}/data/preprocess/monai_persistent_Dataset
RUN chown ${NB_USER} ${HOME} ${HOME}/data/preprocess/standarizationModels
RUN chown ${NB_USER} ${HOME} ${HOME}/data/preprocess/Bias_field_corrected
RUN chown ${NB_USER} ${HOME} ${HOME}/data/metadata/





# First upgrade pip
RUN python3.8 -m pip install --upgrade pip
RUN apt-get update  
RUN apt-get install python3-dev -y

RUN python3.8 -m pip install --no-cache-dir jupyter matplotlib
RUN python3.8 -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2
RUN python3.8 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

RUN python3.8 -m pip install \
    #itk \
    #pytorch-ignite==0.4.8 \
    PyWavelets==1.3.0 \
    scipy==1.8.1 \
    nibabel==3.2.2 \
    pillow!=8.3.0 \  
    #tensorboard==2.9.0 \
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
    #tensorboardX==2.5 \
    types-PyYAML \
    pyyaml==6.0 \
    fire==0.4.0 \
    jsonschema==4.6.0 \
    kymatio==0.2.1 \
    juliacall==0.8.0 \
    jupyterlab_github==3.0.1 \
    torchio==0.18.77 \
    pytorch-lightning==\
    git+https://github.com/sat28/githubcommit.git \
    gdown==4.4.0 \
    seaborn==0.11.2 \
    PyTorchUtils==0.0.3 \
    optuna==2.10.0 \
    jupyterlab-git==0.37.1 \
    comet-ml==3.31.5 \
    voxelmorph==0.2 \
    tensorflow==2.9.1 \
    tensorflow_addons==0.17.0 \
    ipywidgets==7.7.0 \
    h5py==3.7.0 \
    #itk-elastix==0.14.1 \
    zenodopy==0.2.0 \
    evalutils==0.3.0 \
    nnunet==1.7.0 \
    #SimpleITK \
    git+https://github.com/balbasty/nitorch#egg=nitorch[all] \
    threadpoolctl==3.1.0 \
    batchgenerators==0.24 \
    zenodo_get==1.3.4 \
    git+https://github.com/DIAGNijmegen/picai_baseline \
    git+https://github.com/DIAGNijmegen/picai_prep \
    git+https://github.com/DIAGNijmegen/picai_eval \
    #https://github.com/DIAGNijmegen/picai_prep/archive/refs/tags/v1.3.2.zip \
    KevinSR==0.1.19 \
    dask==2022.6.0 \
    intensity-normalization[ants] \
    numba==0.55.2 \
    pyro-ppl==1.8.1 \
    warp-lang==0.2.3 \
    report_guided_annotation \
    geomloss[full] \
    torch-optimizer \
    gspread \
    torchmetrics \
    ray[tune] \
    ray[air]\
    horovod[pytorch] \
    GPy \
    sklearn \
    pytorch-lightning-bolts \
    git+https://github.com/ray-project/ray_lightning \
    fairscale
    # git+https://github.com/ray-project/ray_lightning

#ENV CUDAToolkit_ROOT=/usr/local/cuda-11.3 


# RUN apt-get install python3.8-dev -y
RUN python3.8 -m pip install 'monai[nibabel, skimage, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, lmdb, psutil, cucim, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs]'

#host segmentations and baselines
RUN git clone https://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@github.com/jakubMitura14/piCaiCode.git ${HOME}/data/piCaiCode

RUN git clone https://github.com/DIAGNijmegen/AbdomenMRUS-prostate-segmentation.git ${HOME}/externalRepos/picaiHostSegmentation
RUN git clone https://github.com/DIAGNijmegen/picai_baseline.git ${HOME}/externalRepos/picaiHostBaseline
RUN git clone https://github.com/brudfors/UniRes.git ${HOME}/externalRepos/uniRes
RUN git clone https://github.com/neel-dey/Atlas-GAN.git ${HOME}/externalRepos/conditionalAtlasGAN
RUN git clone https://github.com/SuperElastix/SimpleElastix ${HOME}/externalRepos/elastix




#USER root

# ENV PATH="/usr/lib/python3.8/lib-dynload:${PATH}"
# ENV PATH="/usr/local/lib/python3.8/dist-packages:${PATH}"
# ENV PYTHONPATH=$PYTHONPATH:'/usr/lib/python3.8'
# ENV PYTHONHOME=$PYTHONHOME:'/usr/lib/python3.8'


RUN mkdir ${HOME}/scripts

#RUN git clone https://github.com/SuperElastix/SimpleElastix ${HOME}/externalRepos/elastix

# RUN cd ${HOME}/externalRepos/elastix && \
#     mkdir buildd && \
#     cd buildd && \
#     cmake ${HOME}/externalRepos/elastix/SuperBuild && \
#     make -j10 


# RUN cp ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python/_SimpleITK.so ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python/Packaging
# RUN cp ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python/_SimpleITK.so /home/sliceruser/

# RUN python3.8 ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python/Packaging/setup.py install

# RUN python3.8 -m pip install SimpleITK-SimpleElastix

#maybe remove if registering?
RUN python3.8 -m pip install itk
RUN python3.8 -m pip install SimpleITK

# ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python/_SimpleITK.so

# RUN cd ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python && \
#     python3 Packaging/setup.py install




# git clone https://github.com/SuperElastix/elastix.git ${HOME}/externalRepos/elastixBase
# cd ${HOME}/externalRepos/elastixBase

# https://github.com/SuperElastix/elastix.git


RUN ["chmod", "+x", "/home/sliceruser/externalRepos/elastix"]
#from https://installati.one/ubuntu/22.04/libpangoxft-1.0-0/
RUN apt update
RUN apt install libtbb-dev
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


# ENV LD_LIBRARY_PATH=${HOME}/elastixBase/lib/libANNlib-5.0.so
# ENV LD_LIBRARY_PATH=${HOME}/elastixBase/lib/libANNlib-5.0.so.1

# ENV PATH="${HOME}/elastixBase/lib/libANNlib-5.0.so:${PATH}" 
# ENV PATH="${HOME}/elastixBase/lib/libANNlib-5.0.so.1:${PATH}" 


# cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so ${HOME}/libANNlib-5.0.so
# cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 ${HOME}/libANNlib-5.0.so.1

# ${HOME}/elastixBase/elastix-5.0.1-linux/bin/elastix

#https://github.com/SuperElastix/elastix/releases/download/5.0.1/elastix-5.0.1-linux.tar.bz2 && \

#RUN ["chmod", "+x", "/home/sliceruser/elastixBase"]

# LD_LIBRARY_PATH=${HOME}/elastixBase/lib/libANNlib-5.0.so LD_LIBRARY_PATH=${HOME}/elastixBase/lib/libANNlib-5.0.so.1 ${HOME}/elastixBase/elastix-5.0.1-linux/bin/elastix
#  RUN      python3.8 -m zenodo_get --retry=8 10.5281/zenodo.6517397

# USER root
# RUN chown ${NB_USER} ${HOME} ${HOME}/data/metadata/
# RUN ["chmod", "+x", "/home/sliceruser/data/metadata"]
# USER ${NB_USER}


USER ${NB_USER}

# RUN /home/sliceruser/install.sh ${HOME}/Slicer/Slicer 
# #RUN ["chmod", "+x", "/home/sliceruser/Slicer/Slicer"]

EXPOSE $VNCPORT $JUPYTERPORT

# ENTRYPOINT ["/home/sliceruser/run.sh"]

# CMD ["sh", "-c", "./Slicer/bin/PythonSlicer -m jupyter notebook --port=$JUPYTERPORT --ip=0.0.0.0 --no-browser --NotebookApp.default_url=/lab/"]

#login to github cli 
# COPY mytoken.txt .
# RUN gh auth login --with-token < mytoken.txt
RUN git config --global user.name "Jakub Mitura"
RUN git config --global user.email "jakub.mitura14@gmail.com"
RUN git config -l

#https://itsfoss.com/solve-open-shared-object-file-quick-tip/

# USER root
# # ENV PYTHONPATH=$PYTHONPATH:'/home/sliceruser/Slicer/bin/PythonSlicer'
# # ENV PYTHONHOME=$PYTHONHOME:'/home/sliceruser/Slicer/bin/PythonSlicer'


# # set PYTHONHOME=C:\Python33
# # set PYTHONPATH=C:\Python33\lib

# RUN update-alternatives --install /usr/bin/python python /home/sliceruser/Slicer/bin/PythonSlicer 3
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt install -y python3.11
# # RUN apt install -y python3.11-dev
# # RUN apt install -y python3.11-gdbm
# USER ${NB_USER}

#RUN    pip install SimpleITK-SimpleElastix


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


USER root 
COPY start.sh .
RUN ["chmod", "+x", "start.sh"]
RUN chown ${NB_USER} start.sh
USER ${NB_USER}


# CMD /bin/bash start.sh
# ENTRYPOINT [ "/bin/bash","-l", "-c" ]
ENTRYPOINT [ "/bin/bash", "start.sh","-l", "-c" ]






