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
import pathlib

import logging
import os

import pandas
import SimpleITK as sitk
import shutil
import radiomics
from radiomics import featureextractor

######### get files from folder
files=[]
for subdir, dirs, files in os.walk(targetDir):
    files=files
#segmDir= '/home/data/picai_labels/csPCa_lesion_delineations/AI/Bosma22a'
segmDir= '/home/data/prostateFold'
segmDirConnected= '/home/data/prostateFoldB'

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

# len(labelFiles)

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
    currPath = segmFiles[i]
    # currPath = labelFiles[i]
    imageSeg = sitk.ReadImage(join(segmDir,currPath))
    stemm= Path(currPath).stem
    studyId=stemm.split('_')[1].replace('.nii.gz','')#.replace(".nii","").replace("uun_semi_super_","")
    studyId=studyId.replace('.nii','')#.replace(".nii","").replace("uun_semi_super_","")
    labbb=list(filter(lambda el:studyId in el,labelFiles))
    if(len(labbb)>0):
        t2w,adc,hbv=getMriFile(mriFiles, studyId)
        return (currPath,labbb[0],t2w,adc,hbv,studyId)
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

    imageArr= sitk.GetArrayFromImage(lbl)
    imageArr = (imageArr > 0).astype('uint8')    
    lbl_new: sitk.Image = sitk.GetImageFromArray(imageArr)
    lbl_new.CopyInformation(lbl)

    connectedComps= sitk.ConnectedComponent(lbl_new)
    # connectedComps=sitk.Cast(connectedComps, sitk.sitkInt64)
    writer = sitk.ImageFileWriter()
    newPath = path.replace(segmDir,segmDirConnected)
    
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(newPath)
    writer.Execute(connectedComps)
    return newPath       



zipped = list(map( getZipped ,range(0,len(segmFiles)) ))
#zipped=zipped[0:15]# krowa


zipped= list(filter(lambda el: el[0]!=' ' ,zipped))
# zipped= list(filter(lambda el: 'uunsemi_' in el[0] ,zipped))
zipped= list(filter(lambda el: el[2]!=' ' ,zipped))
zipped= list(filter(lambda el: el[3]!=' ' ,zipped))
zipped= list(filter(lambda el: el[4]!=' ' ,zipped))



labelsToEval= list(map(lambda el:join( labelsPath,el[1]) ,zipped))
segmToEval= list(map(lambda el:join( segmDir,el[0]) ,zipped))
segmToEvalOrig=segmToEval.copy()
segmToEval=list(map(processSegmentations,segmToEval))

t2ws= list(map(lambda el:getMriPath(el[2]) ,zipped))
adcs= list(map(lambda el:getMriPath(el[3]) ,zipped))
hbvs= list(map(lambda el:getMriPath(el[4]) ,zipped))
ids= list(map(lambda el:el[5] ,zipped))



def getValidScore(segmToEvall,labelsToEval):
    valid_metrics = evaluate(y_det=segmToEvall,
                            y_true=labelsToEval,
                            y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
                            num_parallel_calls=os.cpu_count())

    valid_metrics.auroc
    valid_metrics.AP
    print(f"auroc {valid_metrics.auroc} AP {valid_metrics.AP}  ")
    return valid_metrics.score, valid_metrics.auroc, valid_metrics.AP

# scorePrim = getValidScore(segmToEval,labelsToEval)
scorePrim = getValidScore(segmToEvalOrig,labelsToEval)
scorePrim


radiomTemp= '/home/data/radiomTemp'
dataDir =  '/home/data/radiomicsOut'

params = '/workspaces/forPicaiDocker/testRadiomics/Params.yaml' #os.path.join(dataDir, "examples", "exampleSettings", "Params.yaml")
extractor = featureextractor.RadiomicsFeatureExtractor(params)


def getSingleFileEval(index,imageArr):
    imageFilepath = imageArr[index]
    maskFilepath = segmToEval[index]
    # maskFilepath = segmToEval[index]
    id = ids[index]
    lbl=sitk.ReadImage(maskFilepath)
    imgg = sitk.ReadImage(imageFilepath)
    segArr= sitk.GetArrayFromImage(lbl)
    uniqq = np.unique(segArr)

    uniqq = list(filter(lambda el : el>0 ,uniqq))
    print(f" uniqq {uniqq} {np.sum(segArr)} lbl size {lbl.GetSize()} imgg size {imgg.GetSize()}  ")
    result=[{'id':' '}]
    for curr in uniqq:
        try:
            if( np.sum(np.where( segArr!=curr, 0,1 ))>4 ):
                print("***")
            # try:
                dictt = extractor.execute(imageFilepath, maskFilepath,label=int(curr))#, curr
                dictt['id']=id
                dictt['labNum']=curr
                dictt['imageFilepath']=imageFilepath
                dictt['maskFilepath']=maskFilepath
                dictt['isError']=False
                resultDict = pandas.Series(dictt)
                result= result+[resultDict]
            else:
                print("----")    
            # except:
                # print("error feature extracting")
                dictt={'id':id , 'labNum' : curr
                ,'imageFilepath' :imageFilepath
                ,'maskFilepath' :maskFilepath
                ,'isError':True }
                resultDict = pandas.Series(dictt)
                result= result+[resultDict]
        except Exception as inst:
            print(f"eeee {inst}")
        # locRadiomDf=locRadiomDf.append(result, ignore_index = True)
    return result

    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(findPathh,dirDictt=dirDict,keyWord=keyword,targetDir=targetDir)  ,list(dff.iterrows()))


def getSegmentationRadiom(imageArr):
    locRadiomDf = pd.DataFrame()
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(partial(getSingleFileEval,imageArr=imageArr)  ,list(range(0,len(labelsToEval))))
    flat_list=resList
    flat_list = [item for sublist in resList for item in sublist]
    # print(f"flat_list {flat_list}  ")
    flat_list=list(filter(lambda entry: entry['id']!=' '  ,flat_list))
    for entry in flat_list:
        locRadiomDf=locRadiomDf.append(entry, ignore_index = True)

    #for index in range(0,len(labelsToEval)):
        
        # imageFilepath = imageArr[index]
        # maskFilepath = segmToEval[index]
        # # maskFilepath = segmToEval[index]
        # id = ids[index]
        # lbl=sitk.ReadImage(maskFilepath)
        # imgg = sitk.ReadImage(imageFilepath)
        # segArr= sitk.GetArrayFromImage(lbl)
        # uniqq = np.unique(segArr)

        # uniqq = list(filter(lambda el : el>0 ,uniqq))

        # for curr in uniqq:
        #     # try:
        #     dictt = extractor.execute(imageFilepath, maskFilepath,label=int(curr))#, curr
        #     dictt['id']=id
        #     dictt['labNum']=curr
        #     dictt['imageFilepath']=imageFilepath
        #     dictt['maskFilepath']=maskFilepath
        #     result = pandas.Series(dictt)
        #     locRadiomDf=locRadiomDf.append(result, ignore_index = True)
            # except:
            #     pass
    return locRadiomDf

# dff =getSegmentationRadiom(t2ws)

def copyFiless(segmFile):
    newPath = join(radiomTemp,Path(segmFile).name)
    shutil.copyfile(segmFile,newPath )
    return newPath



# outputFilepathStats = os.path.join(dataDir, f"columnStats_t2w.csv")
# minMaxess = pd.read_csv(outputFilepathStats)
# row = list(dff.iterrows())[0]
# currentCol=minMaxess.iloc[0]['name']
# minMaxesDf=minMaxess
# minMaxesDf['name']


def standardizeFile(maskFilepath):
    if(exists(maskFilepath)):
        imageSeg = sitk.ReadImage(maskFilepath)
        imageSegArr= sitk.GetArrayFromImage(imageSeg)
        imageSegArr = (imageSegArr > 0)
        
        origPath = maskFilepath.replace('/home/data/radiomTemp',segmDir)
        origImage = sitk.ReadImage(origPath)
        origArr= sitk.GetArrayFromImage(origImage)
        
        origArr[np.logical_not(imageSegArr)]=0
        # imageSegArr[selector]=0
        lbl_new: sitk.Image = sitk.GetImageFromArray(origArr)
        lbl_new.CopyInformation(origImage)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(maskFilepath)
        writer.Execute(lbl_new)    
    return maskFilepath



def modifyFile(row,minMaxesDf,currentCol):
    row = row[1]
    maskFilepath= row['maskFilepath']
    maskFilepath = join(radiomTemp,Path(maskFilepath).name)

    if(exists(maskFilepath)):
        imageSeg = sitk.ReadImage(maskFilepath)
        imageSegArr= sitk.GetArrayFromImage(imageSeg)
        intr = minMaxesDf.loc[minMaxesDf['name'] == currentCol]
        min = intr['min'].to_numpy()[0]
        max = intr['max'].to_numpy()[0]
        labNum = row['labNum']
        #setting to zero cases when no extraction was possible - for example single voxel masks
        if(row['isError']):
            imageSegArr = np.where( imageSegArr!=labNum, imageSegArr,0 )#(imageSegArr==labNum)
        else:
            currValue = row[currentCol]
            if(currValue<min or currValue>max):
                print(" modyfing arr ")
                imageSegArr = np.where( imageSegArr!=labNum, imageSegArr,0 )#(imageSegArr==labNum)
            
        
        # imageSegArr[selector]=0
        lbl_new: sitk.Image = sitk.GetImageFromArray(imageSegArr)
        lbl_new.CopyInformation(imageSeg)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(maskFilepath)
        writer.Execute(lbl_new)    

# modifyFile(row,minMaxess,currentCol)


resDf= pd.DataFrame(columns=['modality','currentCol', 'score','auroc', 'ap'])
outputFilepathRes = os.path.join(dataDir, f"RadiomFeatureOnScore.csv")
# scorePrim = getValidScore(segmToEval,labelsToEval)
        # resDfpd.concat({'modality':modality,'currentCol':currentCol, 'score':score   },ignore_index=True)



modalities = [('adc',adcs),('t2w', t2ws), ('hbv',hbvs )]

def getFeaturesOfSegm():
    for modalityPair in modalities:
        modality=modalityPair[0]
        imageArr= modalityPair[1]
        
        outputFilepathStats = os.path.join(dataDir, f"columnStats_{modality}.csv")
        minMaxess = pd.read_csv(outputFilepathStats)
        featuresDf = getSegmentationRadiom(imageArr)
        
        outputSegmFeatures = os.path.join(dataDir, f"segmFeatures_{modality}.csv")
        featuresDf.to_csv(outputSegmFeatures)

# getFeaturesOfSegm()


# scorePrim = getValidScore(segmToEval,labelsToEval)
# scorePrim = getValidScore(segmToEvalOrig,labelsToEval)
# scorePrim

for modalityPair in modalities:
    modality=modalityPair[0]
    imageArr= modalityPair[1]
    outputFilepathStats = os.path.join(dataDir, f"columnStats_{modality}.csv")
    minMaxess = pd.read_csv(outputFilepathStats)
    outputSegmFeatures = os.path.join(dataDir, f"segmFeatures_{modality}.csv")
    featuresDf = pd.read_csv(outputSegmFeatures)
    rowws = list(featuresDf.iterrows())

    for currentCol in minMaxess['name'].to_numpy():
        shutil.rmtree(radiomTemp)
        pathlib.Path(radiomTemp).mkdir(parents=True, exist_ok=True) 
        tempPaths = list(map( copyFiless ,segmToEval  ))
        with mp.Pool(processes = mp.cpu_count()) as pool:
            pool.map( partial(modifyFile,minMaxesDf=minMaxess,currentCol=currentCol),rowws)
        tempPaths=list(map(standardizeFile,tempPaths))

        score,auroc,ap  = getValidScore(tempPaths,labelsToEval)
        print(f"currentCol {currentCol} score {score}  ")
        # curSeriess= pd.Series([modality, currentCol,score])
        curSeriess=pd.DataFrame([[modality, currentCol,score,auroc,ap ]],
                   columns=['modality','currentCol', 'score' ,'auroc', 'ap'])

        resDf=pd.concat([resDf,curSeriess])

        # resDfpd.concat({'modality':modality,'currentCol':currentCol, 'score':score   },ignore_index=True)
        print(f"rrrrrrrr   {resDf}")
        resDf.to_csv(outputFilepathRes)




# modalityPair= modalities[0]

# modality=modalityPair[0]
# imageArr= modalityPair[1]
# outputFilepathStats = os.path.join(dataDir, f"columnStats_{modality}.csv")
# minMaxess = pd.read_csv(outputFilepathStats)
# outputSegmFeatures = os.path.join(dataDir, f"segmFeatures_{modality}.csv")
# featuresDf = pd.read_csv(outputSegmFeatures)
# rowws = list(featuresDf.iterrows())

# currentCol=minMaxess['name'].to_numpy()[6]


# shutil.rmtree(radiomTemp)
# pathlib.Path(radiomTemp).mkdir(parents=True, exist_ok=True) 
# tempPaths = list(map( copyFiless ,segmToEval  ))
# list(map( partial(modifyFile,minMaxesDf=minMaxess,currentCol=currentCol),rowws))
# tempPaths=list(map(standardizeFile,tempPaths))

# score = getValidScore(tempPaths,labelsToEval)
# print(f"currentCol {currentCol} score {score}  ")
# resDf.append({'modality':modality,'currentCol':currentCol, 'score':score   },ignore_index=True)
# resDf.to_csv(outputFilepathRes)














# modalityPair=modalities[0]

# modality=modalityPair[0]
# imageArr= modalityPair[1]

# outputFilepathStats = os.path.join(dataDir, f"columnStats_{modality}.csv")
# minMaxess = pd.read_csv(outputFilepathStats)
# featuresDf = getSegmentationRadiom(imageArr)
# rowws = list(featuresDf.iterrows())

# def validateQuantity()

# for currentCol in minMaxess['name'].to_numpy():
#     shutil.rmtree(radiomTemp)
#     pathlib.Path(radiomTemp).mkdir(parents=True, exist_ok=True) 
#     tempPaths = list(map( copyFiless ,segmToEval  ))
#     list(map( partial(modifyFile,minMaxesDf=minMaxess,currentCol=currentCol),rowws))
#     score = getValidScore(tempPaths,labelsToEval)
#     print(f"currentCol {currentCol} score {score}  ")


#     # for segmFile in segmToEval:
#     #     shutil.copyfile(segmFile, join(segmFile,Path(segmFile).name))
    
#     # for index in range(0,len(labelsToEval)):
#     #     imageFilepath = imageArr[index]
#     #     maskFilepath = segmToEval[index]
#     #     id = ids[index]
#     #     lbl=sitk.ReadImage(maskFilepath)
#     #     segArr= sitk.GetArrayFromImage(lbl)
#     #     uniqq = np.unique(segArr)
#     #     uniqq = list(filter(lambda el : el>0 ,uniqq))

#     #     for curr in uniqq:
#     #         dictt = extractor.execute(imageFilepath, maskFilepath, curr)
#     #         dictt['id']=id
#     #         dictt['labNum']=curr
#     #         result = pandas.Series(dictt)
#     #         locRadiomDf=locRadiomDf.append(result, ignore_index = True)

    
#     # for row in list(minMaxess.iterrows()):
#     #     row = row[1]








#     segmToEval= list(map(lambda el:join( segmDir,el[0]) ,zipped))
# t2ws= list(map(lambda el:getMriPath(el[2]) ,zipped))
# adcs= list(map(lambda el:getMriPath(el[3]) ,zipped))
# hbvs= list(map(lambda el:getMriPath(el[4]) ,zipped))



















# ################################3 simple itk label stats

# imageSeg.GetSize()

# stats.Execute(connectedComps)
# l= stats.GetLabels()[0]
# label_elongation=stats.GetElongation(l)
# labelPhysSize = stats.GetPhysicalSize(l)
# labelRoundness = stats.GetRoundness(l)



# def get_sizes(path):    
#     image = sitk.ReadImage(join(targetDir,path))
#     targetSpac=(3.0, 0.5, 0.5)
#     origSize= image.GetSize()
#     orig_spacing=image.GetSpacing()
#     currentSpacing = list(orig_spacing)
#     print(f"origSize {origSize}")
#     #new size of the image after changed spacing
#     new_size = tuple([int(origSize[0]*(orig_spacing[0]/targetSpac[0])),
#                     int(origSize[1]*(orig_spacing[1]/targetSpac[1])),
#                     int(origSize[2]*(orig_spacing[2]/targetSpac[2]) )  ]  )


#     resample = sitk.ResampleImageFilter()
#     resample.SetOutputSpacing(targetSpac)
#     resample.SetOutputDirection(image.GetDirection())
#     resample.SetOutputOrigin(image.GetOrigin())
#     resample.SetTransform(sitk.Transform())
#     resample.SetDefaultPixelValue(image.GetPixelIDValue())
#     resample.SetInterpolator(sitk.sitkBSpline)
#     resample.SetSize(new_size)
#     image= resample.Execute(image)



#     arr = sitk.GetArrayFromImage(image)
#     indicies=  np.argwhere(arr)
#     xs= list(map(lambda el: el[0],indicies ))
#     ys= list(map(lambda el: el[1],indicies ))
#     zs= list(map(lambda el: el[2],indicies ))
#     return (max(xs)-min(xs),max(ys)-min(ys),max(zs)-min(zs)   )

# sizes = list(map(get_sizes, files  ))    

# xs= list(map(lambda el: el[0],sizes ))
# ys= list(map(lambda el: el[1],sizes ))
# zs= list(map(lambda el: el[2],sizes ))

# maxX = max(xs)
# maxY = max(ys)
# maxZ = max(zs)