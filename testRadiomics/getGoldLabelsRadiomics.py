import functools
import multiprocessing as mp
import os
from functools import partial
from zipfile import BadZipFile, ZipFile
import numpy as np
import pandas as pd
import SimpleITK as sitk
targetDir= '/home/data/prostateFold'
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
from preprocess_utils import z_score_norm
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter
from picai_baseline.unet.training_setup.preprocess_utils import z_score_norm
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample, PreprocessingSettings, crop_or_pad, resample_img
from report_guided_annotation import extract_lesion_candidates
from picai_eval import evaluate
import radiomics
from __future__ import print_function

import logging
import os

import pandas
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

######### get files from folder
files=[]
for subdir, dirs, files in os.walk(targetDir):
    files=files
# segmDir= '/home/data/picai_labels/csPCa_lesion_delineations/AI/Bosma22a'
segmDir= '/home/data/orig_unet_semi'

segmFiles=[]
for subdir, dirs, files in os.walk(segmDir):
    segmFiles=files
segmFiles

labelFiles= []
labelsPath= '/home/data/picai_labels/csPCa_lesion_delineations/human_expert/resampled'
for subdir, dirs, files in os.walk(labelsPath):
    labelFiles=files
mriFiles= []
mrisPath= '/home/data/orig'
for subdir, dirs, files in os.walk(mrisPath):
    mriFiles=mriFiles+files



segmFiles= list(filter(lambda name: '.nii.gz' in name  ,segmFiles ))
labelFiles= list(filter(lambda name: '.nii.gz' in name  ,labelFiles ))



labelFiles.sort(key= lambda el: int(el.split('_')[1].replace('.nii.gz','')))
segmFiles.sort(key= lambda el: int(el.split('_')[1].replace('.nii.gz','')))
# segmFiles.sort(key= lambda el: int(el.replace('uun_semi_super_','').replace('.nii.gz','')))

len(labelFiles)

def getFirtstEmpty(filtered,key):
    liist=list(filter(lambda el: key in el ,filtered))
    if(len(liist)>0):
        return liist[0]
    return ' '

def getMriFile(mriFiles, studyId):
    filtered = list(filter(lambda el: studyId in el ,mriFiles  ))
    # print(f"studyId {studyId}  filtered {filtered}")
    t2w = getFirtstEmpty(filtered,'t2w')
    adc = getFirtstEmpty(filtered,'adc')
    hbv = getFirtstEmpty(filtered,'hbv')
    return (t2w,adc,hbv)

def getZipped(i):
    currPath = labelFiles[i]
    imageSeg = sitk.ReadImage(join(segmDir,currPath))
    stemm= Path(currPath).stem
    studyId=stemm.split('_')[1].replace('.nii.gz','')#.replace(".nii","").replace("uun_semi_super_","")
    studyId=studyId.replace('.nii','')#.replace(".nii","").replace("uun_semi_super_","")
    segmm=list(filter(lambda el:studyId in el,segmFiles))
    if(len(segmm)>0):
        t2w,adc,hbv=getMriFile(mriFiles, studyId)
        return (segmm[0],currPath,t2w,adc,hbv,studyId)
    return (' ', ' ', ' ', ' ', ' ',' ')    

def getMriPath(name):
    patId = name.split('_')[0]
    return join(mrisPath,patId,name )


def getNotEmptyLabels(tupl):
    labPath = tupl[1]
    imageSeg = sitk.ReadImage(join( labelsPath,labPath))
    imageArr= sitk.GetArrayFromImage(imageSeg)
    imageArr = (imageArr >= 1).astype('uint8')
    return np.sum(imageArr)>0

def processLabels(path):
    lbl=sitk.ReadImage(path)
    imageArr= sitk.GetArrayFromImage(lbl)
    imageArr = (imageArr >= 1).astype('uint8')    
    lbl_new: sitk.Image = sitk.GetImageFromArray(imageArr)
    lbl_new.CopyInformation(lbl)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(path)
    writer.Execute(lbl_new)       

def processSegmentations(path):
    lbl=sitk.ReadImage(path)
    connectedComps= sitk.ConnectedComponent(lbl)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(path)
    writer.Execute(connectedComps)       




zipped = list(map( getZipped ,range(0,len(labelFiles)) ))
zipped= list(filter(lambda el: el[0]!=' ' ,zipped))
zipped= list(filter(lambda el: 'uunsemi_' in el[0] ,zipped))
zipped= list(filter(lambda el: el[2]!=' ' ,zipped))
zipped= list(filter(lambda el: el[3]!=' ' ,zipped))
zipped= list(filter(lambda el: el[4]!=' ' ,zipped))
zipped= list(filter(getNotEmptyLabels ,zipped))


# path=labelsToEval[0]
# lbl=sitk.ReadImage(path)
# imageArr= sitk.GetArrayFromImage(lbl)
# imageArr = (imageArr >= 1).astype('uint8')    
# lbl_new: sitk.Image = sitk.GetImageFromArray(imageArr)
# lbl_new.CopyInformation(lbl)
# writer = sitk.ImageFileWriter()
# writer.KeepOriginalImageUIDOn()
# writer.SetFileName(path)
# writer.Execute(lbl_new)    




labelsToEval= list(map(lambda el:join( labelsPath,el[1]) ,zipped))
list(map(processLabels,labelsToEval))
segmToEval= list(map(lambda el:join( segmDir,el[0]) ,zipped))
list(map(processSegmentations,segmToEval))

t2ws= list(map(lambda el:getMriPath(el[2]) ,zipped))
adcs= list(map(lambda el:getMriPath(el[3]) ,zipped))
hbvs= list(map(lambda el:getMriPath(el[4]) ,zipped))
ids= list(map(lambda el:el[5] ,zipped))
df = pd.DataFrame(columns=['ID','Image','Mask' ])
df['ID']=ids
df['Image']=t2ws
df['Mask']=labelsToEval
dataDir =  '/home/data/radiomicsOut'
inputCSV = os.path.join(dataDir, 'testCases.csv')
df.to_csv(inputCSV)

valid_metrics = evaluate(y_det=segmToEval,
                        y_true=labelsToEval,
                        y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
                        num_parallel_calls=os.cpu_count())

valid_metrics.auroc
valid_metrics.AP
valid_metrics.score



currPath = segmToEval[0]
imageSeg = sitk.ReadImage(currPath)
stemm= Path(currPath).stem
studyId=stemm.replace(".nii","").replace("nnunetOut_","")

stats = sitk.LabelShapeStatisticsImageFilter()
stats.SetComputeOrientedBoundingBox(True)
connectedComps=sitk.ConnectedComponent(imageSeg)
imageArr= sitk.GetArrayFromImage(connectedComps)
np.unique(imageArr)



# labelsToEval= list(map(lambda el:join( labelsPath,el[1]) ,zipped))
# segmToEval= list(map(lambda el:join( segmDir,el[0]) ,zipped))
# t2ws= list(map(lambda el:getMriPath(el[2]) ,zipped))
# adcs= list(map(lambda el:getMriPath(el[3]) ,zipped))
# hbvs= list(map(lambda el:getMriPath(el[4]) ,zipped))

###################3333 pyradiomics
import six
from radiomics import featureextractor, getTestCase
dataDir =  '/home/data/radiomicsOut'
imageName, maskName = labelsToEval[0],t2ws[0]   #getTestCase('brain1', dataDir)
outPath=dataDir
outputFilepath = os.path.join(outPath, 'radiomics_features.csv')
progress_filename = os.path.join(outPath, 'pyrad_log.txt')
rLogger = logging.getLogger('radiomics')
params = '/workspaces/forPicaiDocker/testRadiomics/Params.yaml' #os.path.join(dataDir, "examples", "exampleSettings", "Params.yaml")
handler = logging.FileHandler(filename=progress_filename, mode='w')
handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
rLogger.addHandler(handler)


# Initialize logging for batch log messages
logger = rLogger.getChild('batch')

# Set verbosity level for output to stderr (default level = WARNING)
radiomics.setVerbosity(logging.INFO)
flists = pandas.read_csv(inputCSV).T

extractor = featureextractor.RadiomicsFeatureExtractor(params)
results = pandas.DataFrame()






# labelsToEval= list(map(lambda el:join( labelsPath,el[1]) ,zipped))
# list(map(processLabels,labelsToEval))
# segmToEval= list(map(lambda el:join( segmDir,el[0]) ,zipped))
# t2ws= list(map(lambda el:getMriPath(el[2]) ,zipped))
# adcs= list(map(lambda el:getMriPath(el[3]) ,zipped))
# hbvs= list(map(lambda el:getMriPath(el[4]) ,zipped))
modality='hbv'
outputFilepath = os.path.join(outPath, f"radiomics_features_{modality}.csv")

outdf = pd.DataFrame()

label = 1 #flists[entry].get('Label', None)

for index in range(0,len(labelsToEval)):
    imageFilepath = hbvs[index]
    maskFilepath = labelsToEval[index]
    result = pandas.Series(extractor.execute(imageFilepath, maskFilepath, label))
    outdf=outdf.append(result, ignore_index = True)

outdf.to_csv(outputFilepath)

notDiagnosticCols= list(filter(lambda colname : 'diagnostics_' not in colname ,outdf.columns))

minMaxesDf= pd.DataFrame(columns=['name', 'min', 'max', 'std'])


def getColumnStats(colname,modality):
    arrr = outdf[colname].to_numpy()
    return (colname, np.min(arrr),np.max(arrr),np.std(arrr))

colStats = list(map(lambda colname: getColumnStats(colname,modality),notDiagnosticCols ))
minMaxesDf['name']=list(map(lambda el: el[0],colStats))
minMaxesDf['min']=list(map(lambda el: el[1],colStats))
minMaxesDf['max']=list(map(lambda el: el[2],colStats))
minMaxesDf['std']=list(map(lambda el: el[3],colStats))

outputFilepathStats = os.path.join(outPath, f"columnStats_{modality}.csv")
minMaxesDf.to_csv(outputFilepathStats)

modalities = [('t2w', t2ws), ('adc',adcs), ('hbv',hbvs )]
for modalityPair in modalities:
    modality=modalityPair[0]
    imageArr= modalities[1]
    outputFilepathStats = os.path.join(outPath, f"columnStats_{modality}.csv")
    minMaxess = pd.read_csv(outputFilepathStats)

    locRadiomDf = pd.DataFrame()
    for index in range(0,len(labelsToEval)):
        imageFilepath = imageArr[index]
        maskFilepath = segmToEval[index]

        result = pandas.Series(extractor.execute(imageFilepath, maskFilepath, 1))
        locRadiomDf=locRadiomDf.append(result, ignore_index = True)

    
    for row in list(minMaxess.iterrows()):
        row = row[1]






    segmToEval= list(map(lambda el:join( segmDir,el[0]) ,zipped))
t2ws= list(map(lambda el:getMriPath(el[2]) ,zipped))
adcs= list(map(lambda el:getMriPath(el[3]) ,zipped))
hbvs= list(map(lambda el:getMriPath(el[4]) ,zipped))



















################################3 simple itk label stats

imageSeg.GetSize()

stats.Execute(connectedComps)
l= stats.GetLabels()[0]
label_elongation=stats.GetElongation(l)
labelPhysSize = stats.GetPhysicalSize(l)
labelRoundness = stats.GetRoundness(l)



def get_sizes(path):    
    image = sitk.ReadImage(join(targetDir,path))
    targetSpac=(3.0, 0.5, 0.5)
    origSize= image.GetSize()
    orig_spacing=image.GetSpacing()
    currentSpacing = list(orig_spacing)
    print(f"origSize {origSize}")
    #new size of the image after changed spacing
    new_size = tuple([int(origSize[0]*(orig_spacing[0]/targetSpac[0])),
                    int(origSize[1]*(orig_spacing[1]/targetSpac[1])),
                    int(origSize[2]*(orig_spacing[2]/targetSpac[2]) )  ]  )


    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpac)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetSize(new_size)
    image= resample.Execute(image)



    arr = sitk.GetArrayFromImage(image)
    indicies=  np.argwhere(arr)
    xs= list(map(lambda el: el[0],indicies ))
    ys= list(map(lambda el: el[1],indicies ))
    zs= list(map(lambda el: el[2],indicies ))
    return (max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs)   )

sizes = list(map(get_sizes, files  ))    

xs= list(map(lambda el: el[0],sizes ))
ys= list(map(lambda el: el[1],sizes ))
zs= list(map(lambda el: el[2],sizes ))

maxX = max(xs)
maxY = max(ys)
maxZ = max(zs)