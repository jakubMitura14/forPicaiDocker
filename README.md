# forPicaiDocker
based on https://github.com/Slicer/SlicerDocker

After building it we will get id for example 198a491286ae. Can be run sth like docker run --init --gpus all --ipc host --privileged --net host -p 8888:8888 -p49053:49053 -v "$PWD":/home/sliceruser/work 198a491286ae
