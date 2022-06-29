echo "managing zenodo files"
DIR="/home/sliceruser/data/orig"
#if data for originals exists from https://www.cyberciti.biz/faq/linux-unix-shell-check-if-directory-empty/
mkdir \
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
${HOME}/data/for_host_whole_gland_segm \
${HOME}/data/for_host_whole_gland_segm/nnunet \
${HOME}/data/for_host_whole_gland_segm/nnunet/input \
${HOME}/data/for_host_whole_gland_segm/nnunet/output \
${HOME}/data/for_host_whole_gland_segm/algorithm \ 
${HOME}/data/for_host_whole_gland_segm/algorithm/results \
${HOME}/data/for_host_whole_gland_segm/nnunet/output/transverse-whole-prostate-mri \




echo "downloading zenodo"
/home/sliceruser/Slicer/bin/PythonSlicer -m zenodo_get --retry=8 10.5281/zenodo.6517397
# #prepare  csv containing metadata and paths
/home/sliceruser/Slicer/bin/PythonSlicer processMetaData.py
# # we already unpacked files now we can remove zips
# rm /home/sliceruser/picai_public_images_fold0.zip 
# rm /home/sliceruser/picai_public_images_fold1.zip 
# rm /home/sliceruser/picai_public_images_fold2.zip 
# rm /home/sliceruser/picai_public_images_fold3.zip 
# rm /home/sliceruser/picai_public_images_fold4.zip

#standarization - it can take sth like 90h on 10 cores cpu
#/home/sliceruser/Slicer/bin/PythonSlicer standardize.py
