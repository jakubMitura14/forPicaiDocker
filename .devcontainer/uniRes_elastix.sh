# cd externalRepos/uniRes
# /home/sliceruser/Slicer/bin/PythonSlicer -m pip install .
# NI_COMPILED_BACKEND="C" /home/sliceruser/Slicer/bin/PythonSlicer -m pip install --no-build-isolation .

cd ${HOME}/externalRepos/elastix
mkdir buildd
cd buildd
cmake ${HOME}/externalRepos/elastix/SuperBuild
make -j6

# cd ${HOME}/externalRepos/elastix/buildd/SimpleITK-build/Wrapping/Python
# ${BUILD_DIRECTORY}/SimpleITK-build/Wrapping/Python
# Wrapping/Python


# update-alternatives --install /usr/bin/python python /usr/bin/python

# update-alternatives --install /usr/bin/python/home/sliceruser/Slicer/lib/Python/bin python /usr/bin/python/home/sliceruser/Slicer/lib/Python/bin 

# externalRepos/elastix/buildd/SimpleITK-build/Wrapping/CMakeFiles

# http://127.0.0.1:8888/lab/tree/externalRepos/elastix/SuperBuild

# cd ${HOME}/externalRepos/elastix/buildd


# /usr/bin/python/home/sliceruser/Slicer/lib/Python/bin

# /home/sliceruser/Slicer/bin/PythonSlicer

# update-alternatives --install /usr/bin/python python /home/sliceruser/Slicer/bin/PythonSlicer 1

# /home/sliceruser/data/piCaiCode/preprocessing/registration/parameters.txt

# /home/sliceruser/Slicer/NA-MIC/Extensions-30822/SlicerElastix/lib/Slicer-5.0/elastix -f /home/sliceruser/data/orig/10019/10019_1000019_t2w.mha -m /home/sliceruser/data/orig/10019/10019_1000019_hbv.mha -out /home/sliceruser/data/exploration/ -p /home/sliceruser/data/piCaiCode/preprocessing/registration/parameters.txt

#  elastix -f fixedImage.ext -m movingImage.ext -out outputDirectory -p parameterFile.txt

#  #rigid
#  /home/sliceruser/Slicer/NA-MIC/Extensions-30822/SlicerElastix/lib/Slicer-5.0/elastix -f /home/sliceruser/data/orig/10019/10019_1000019_t2w.mha -m /home/sliceruser/data/orig/10019/10019_1000019_hbv.mha -out /home/sliceruser/data/exploration/ -p /home/sliceruser/data/piCaiCode/preprocessing/registration/parametersRigid.txt
# /home/sliceruser/data/orig/10019/10019_1000019_adc.mha


# /home/sliceruser/Slicer/NA-MIC/Extensions-30822/SlicerElastix/lib/Slicer-5.0/elastix -f /home/sliceruser/data/orig/10019/10019_1000019_t2w.mha -m /home/sliceruser/data/orig/10019/10019_1000019_adc.mha -out /home/sliceruser/data/exploration/ -p /home/sliceruser/data/piCaiCode/preprocessing/registration/parameters.txt
