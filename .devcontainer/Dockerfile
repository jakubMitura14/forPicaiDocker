

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
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
RUN sudo apt update
RUN sudo apt install gh
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
RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
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
    #elastix=4.9.0-1\
    git && \
    apt-get install -q -y --reinstall ca-certificates

RUN add-apt-repository ppa:deadsnakes/ppa

#from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu-jupyter.Dockerfile

# ARG CUDA=11.3
# ARG CUDNN=8.1.0.77-1
# ARG CUDNN_MAJOR_VERSION=8
# ARG LIB_DIR_PREFIX=x86_64
# ARG LIBNVINFER=7.2.2-1
# ARG LIBNVINFER_MAJOR_VERSION=7

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-get update 
# RUN  apt-get update && apt-get install -y --no-install-recommends cuda-command-line-tools-11.3 
# RUN  apt-get update && apt-get install -y --no-install-recommends libcublas-11.3 
# RUN  apt-get update && apt-get install -y --no-install-recommends cuda-nvrtc-11.3 
# RUN  apt-get update && apt-get install -y --no-install-recommends libcufft-11.3 
# RUN  apt-get update && apt-get install -y --no-install-recommends libcurand-11.3 
# RUN  apt-get update && apt-get install -y --no-install-recommends libcusolver-11.3 
# RUN  apt-get update && apt-get install -y --no-install-recommends libcusparse-11.3 
# RUN  apt-get update && apt-get install -y --no-install-recommends curl 
# #RUN  apt-get update && apt-get install -y --no-install-recommends libcudnn8  #=8.1.0.77-1+cuda11.3

# RUN  apt-get update && apt-get install -y --no-install-recommends libfreetype6-dev 
# RUN  apt-get update && apt-get install -y --no-install-recommends libhdf5-serial-dev 
# RUN  apt-get update && apt-get install -y --no-install-recommends libzmq3-dev 
# RUN  apt-get update && apt-get install -y --no-install-recommends pkg-config 
# RUN  apt-get update && apt-get install -y --no-install-recommends software-properties-common 
# RUN  apt-get update && apt-get install -y --no-install-recommends  unzip



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
# RUN mkdir {${HOME}/data, ${HOME}/labels,${HOME}/data/preprocess, ${HOME}/data/orig
# , ${HOME}/data/preprocess/monai_persistent_Dataset, ${HOME}/output,${HOME}/data/piCaiCode,${HOME}/build
# , ${HOME}/data/lightning_logs, ${HOME}/data/preprocess/standarizationModels
# , ${HOME}/data/preprocess/Bias_field_corrected, ${HOME}/data/metadata}



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
    ${HOME}/data/metadata



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
#ARG SLICER_DOWNLOAD_URL=https://download.slicer.org/bitstream/6286ce04e8408647b39f803a

# Use local package:
#ADD $SLICER_ARCHIVE.tar.gz ${HOME}/

# Download package:
#RUN curl -JL "$SLICER_DOWNLOAD_URL" | tar xz -C /tmp && mv /tmp/Slicer* Slicer

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
    apt install -y python3.9 && \
    apt install -y python3-pip && \
    #easy_install -y pip && \
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





COPY managePicaiFiles.sh .
COPY uniRes_elastix.sh .
COPY requirements-dev.txt .
# COPY start-xorg.sh .
# COPY install.sh .
# COPY run.sh .
# COPY .slicerrc.py .

RUN ["chmod", "+x", "/home/sliceruser/managePicaiFiles.sh"]
RUN ["chmod", "+x", "/home/sliceruser/uniRes_elastix.sh"]
RUN ["chmod", "+x", "/home/sliceruser/requirements-dev.txt"]
# RUN ["chmod", "+x", "/home/sliceruser/start-xorg.sh"]
# RUN ["chmod", "+x", "/home/sliceruser/install.sh"]
# RUN ["chmod", "+x", "/home/sliceruser/run.sh"]
# RUN ["chmod", "+x", "/home/sliceruser/.slicerrc.py"]



# USER ${NB_USER}
# RUN mkdir /tmp/runtime-sliceruser
# ENV XDG_RUNTIME_DIR=/tmp/runtime-sliceruser



# First upgrade pip
RUN python3 -m pip install --upgrade pip
#ENV PATH=$PATH:'/home/sliceruser/Slicer/bin/PythonSlicer'
#COPY processMetaData.py .
#COPY standardize.py .


## needed for picai host prostate segmentation algorithm
#### from https://github.com/DIAGNijmegen/AbdomenMRUS-prostate-segmentation/blob/main/Dockerfile

#RUN cd ${HOME}/externalRepos



# # Extend the nnUNet installation with custom trainers
# COPY --chown=${NB_USER}:${NB_USER} nnUNetTrainerV2_focalLoss.py /tmp/nnUNetTrainerV2_focalLoss.py
# RUN SITE_PKG=`   pip show nnunet | grep "Location:" | awk '{print $2}'` && \
#     mv /tmp/nnUNetTrainerV2_focalLoss.py "$SITE_PKG/nnunet/training/network_training/nnUNet_variants/loss_function/nnUNetTrainerV2_focalLoss.py"

# COPY --chown=${NB_USER}:${NB_USER} nnUNetTrainerV2_Loss_FL_and_CE.py /tmp/nnUNetTrainerV2_Loss_FL_and_CE.py
# RUN SITE_PKG=`   pip show nnunet | grep "Location:" | awk '{print $2}'` && \
#     mv /tmp/nnUNetTrainerV2_Loss_FL_and_CE.py "$SITE_PKG/nnunet/training/network_training/nnUNetTrainerV2_Loss_FL_and_CE.py"





# from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu-jupyter.Dockerfile
# # Options:
# #   tensorflow
# #   tensorflow-gpu
# #   tf-nightly
# #   tf-nightly-gpu
# # Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# # Installs the latest version by default.
# ARG TF_PACKAGE=tensorflow
# ARG TF_PACKAGE_VERSION=
# RUN    pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}



RUN pip install --no-cache-dir jupyter matplotlib
# RUN    pip install --no-cache-dir "intensity-normalization[ants]"
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
# Pin jedi; see https://github.com/ipython/ipython/issues/12740
RUN pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2


# Now install websockify and jupyter-server-proxy (fixed at tag v1.6.0)
# RUN pip install --upgrade websockify && \
#     cp /usr/lib/rebind.so /home/sliceruser/Slicer/lib/Python/lib/python3.9/site-packages/websockify/ && \
#        pip install notebook jupyterhub jupyterlab && \
#        pip install -e \
#     git+https://github.com/lassoan/jupyter-desktop-server#egg=jupyter-desktop-server \
#     git+https://github.com/jupyterhub/jupyter-server-proxy@v1.6.0#egg=jupyter-server-proxy

####### ## { Mitura start} ading libraries through pip 

# RUN apt-get install -y software-properties-common
# RUN add-apt-repository ppa:ubuntu-toolchain-r/test
# RUN apt install -y gcc-7 g++-7

#RUN    pip install pyradiomics
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#RUN    pip install requirements-dev.txt
#RUN ipykernel install --user
#ENV PYTHONPATH "${PYTHONPATH}:/home/sliceruser/Slicer/bin/PythonSlicer"

RUN pip install \
    #itk \
    pytorch-ignite==0.4.8 \
    PyWavelets==1.3.0 \
    scipy==1.8.1 \
    nibabel==3.2.2 \
    pillow!=8.3.0 \  
    tensorboard==2.9.0 \
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
    pytorch-lightning==1.6.4 \
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
    itk-elastix==0.14.1 \
    zenodopy==0.2.0 \
    evalutils==0.3.0 \
    nnunet==1.7.0 \
    SimpleITK \
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
    SimpleITK-SimpleElastix \
    geomloss[full]    

RUN    pip install 'monai[nibabel, skimage, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, lmdb, psutil, cucim, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs]'
RUN    pip install torch-optimizer

#for downloading Pi cai data
#RUN    pip install zenodo_get==1.3.4
#downloading from zenodo and preprocessing metadata
# RUN    zenodo_get --retry=8 10.5281/zenodo.6517397
#         # #prepare  csv containing metadata and paths
# RUN /home/sliceruser/Slicer/bin/PythonSlicer processMetaData.py
#         # # we already unpacked files now we can remove zips
# RUN rm /home/sliceruser/picai_public_images_fold0.zip 
# RUN rm /home/sliceruser/picai_public_images_fold1.zip 
# RUN rm /home/sliceruser/picai_public_images_fold2.zip 
# RUN rm /home/sliceruser/picai_public_images_fold3.zip 
# RUN rm /home/sliceruser/picai_public_images_fold4.zip
        #standarization - it can take sth like 90h on 10 cores cpu
        # /home/sliceruser/Slicer/bin/PythonSlicer standardize.py
#RUN /home/sliceruser/managePicaiFiles.sh
#standarization
 
#RUN /home/sliceruser/Slicer/bin/PythonSlicer standardize.py

#copy main repository inside image using token - as the repository is private

##############3 code from picai hosts
# #install prepared preprocessing and evaluation ready libraries
# RUN    pip install git+https://github.com/DIAGNijmegen/picai_prep
# RUN    pip install git+https://github.com/DIAGNijmegen/picai_eval
#host segmentations and baselines
RUN git clone https://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@github.com/jakubMitura14/piCaiCode.git ${HOME}/data/piCaiCode

RUN git clone https://github.com/DIAGNijmegen/AbdomenMRUS-prostate-segmentation.git ${HOME}/externalRepos/picaiHostSegmentation
RUN git clone https://github.com/DIAGNijmegen/picai_baseline.git ${HOME}/externalRepos/picaiHostBaseline
RUN git clone https://github.com/brudfors/UniRes.git ${HOME}/externalRepos/uniRes
RUN git clone https://github.com/neel-dey/Atlas-GAN.git ${HOME}/externalRepos/conditionalAtlasGAN
RUN git clone https://github.com/SuperElastix/SimpleElastix ${HOME}/externalRepos/elastix

#RUN git clone https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration.git ${HOME}/externalRepos

#install unires properly

# Install Slicer extensions
# RUN /home/sliceruser/uniRes_elastix.sh
# USER root

# RUN apt install -y python-is-python3
# # RUN chmod ugo+rwx /home/sliceruser/install.sh
# # RUN chmod ugo+rwx /home/sliceruser/start-xorg.sh
# # RUN chmod ugo+rwx /home/sliceruser/run.sh
# # # RUN chown -R ${NB_USER} ${HOME} ${HOME}/data
# # # RUN chmod -R ugo+rwx ${HOME}/data

#  USER ${NB_USER}
# RUN chown -R ${NB_USER}:${NB_USER} ${HOME}
# RUN chmod -R ugo+rwx ${HOME}

#chown -R $USER:$USER ${HOME}





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



ENTRYPOINT [ "/bin/bash", "-l", "-c" ]






