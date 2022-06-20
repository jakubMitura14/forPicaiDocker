echo "managing zenodo files"
DIR="/home/sliceruser/data/orig"
#if data for originals exists from https://www.cyberciti.biz/faq/linux-unix-shell-check-if-directory-empty/
if [ -d "$DIR" ]
then
	if [ "$(ls -A $DIR)" ]; then
        echo "files already in orig subfolder will not download it from zenodo"
    #and if there are no files we will download and preprocess them there
    else
        echo "downloading zenodo"
        /home/sliceruser/Slicer/bin/PythonSlicer -m zenodo_get --retry=8 10.5281/zenodo.6517397
        # #prepare  csv containing metadata and paths
        /home/sliceruser/Slicer/bin/PythonSlicer processMetaData.py
        # # we already unpacked files now we can remove zips
        rm /home/sliceruser/picai_public_images_fold0.zip 
        rm /home/sliceruser/picai_public_images_fold1.zip 
        rm /home/sliceruser/picai_public_images_fold2.zip 
        rm /home/sliceruser/picai_public_images_fold3.zip 
        rm /home/sliceruser/picai_public_images_fold4.zip
        #standarization - it can take sth like 90h on 10 cores cpu
        #/home/sliceruser/Slicer/bin/PythonSlicer standardize.py
    fi
    else
        echo "no folder /home/sliceruser/data/orig"   
fi        