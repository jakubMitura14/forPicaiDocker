using Pkg
Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
#Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
using MedPipe3D
using MedEye3d
using Distributions
using Clustering
using IrrationalConstants
using ParallelStencil
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedPipe3D.visualizationFromHdf5, MedPipe3D.distinctColorsSaved
using CUDA
using HDF5,Colors
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using MedEval3D
using MedEval3D.BasicStructs
using MedEval3D.MainAbstractions
using MedEval3D
using MedEval3D.BasicStructs
using MedEval3D.MainAbstractions
using UNet
using Hyperopt,Plots
using MedPipe3D.LoadFromMonai
using Flux
using Distributed


monai=MedPipe3D.LoadFromMonai.getMonaiObject()
monai.config.print_config()
root_dir = "/workspaces/dockerForJulia/spleenData/Task09_Spleen/Task09_Spleen/"


resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
compressed_file = joinpath(root_dir, "Task09_Spleen.tar")
md5 = "410d4a301da4e5b2f6f86ec3ddba524e"
monai=MedPipe3D.LoadFromMonai.getMonaiObject()
monai.apps.download_and_extract(resource, compressed_file, root_dir, md5)

monai.config.print_config()


train_labels = map(fileEntry-> joinpath(root_dir,"Task09_Spleen,labelsTr",fileEntry),readdir(joinpath(root_dir,"Task09_Spleen","labelsTr"); sort=true))
train_images = map(fileEntry-> joinpath(root_dir,"Task09_Spleen","imagesTr",fileEntry),readdir(joinpath(root_dir,"Task09_Spleen","imagesTr"); sort=true))
zipped= collect(zip(train_images,train_labels))
tupl=zipped[1]
loaded = LoadFromMonai.loadByMonaiFromImageAndLabelPaths(tupl[1],tupl[2])
