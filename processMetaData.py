import torch
import pandas as pd
import numpy as np
import torchio as tio
from torch.utils.data import DataLoader
import os
import SimpleITK as sitk
from zipfile import ZipFile
from zipfile import BadZipFile
import SimpleITK as sitk

#read metadata, and add columns for additional information
csvPath='/home/sliceruser/labels/clinical_information/marksheet.csv'
df = pd.read_csv(csvPath)
#initializing empty columns
df["reSampledPath"] = ""
df["adc"] = ""
df["cor"] = ""
df["hbv"] = ""
df["sag"] = ""
df["t2w"] = ""
df["t2w"] = ""
df["isAnythingInAnnotated"] = 0
df["isAnyMissing"] = False


rootdir = '/home/sliceruser/data/'

zipDir= '/home/sliceruser/picai_public_images_fold0.zip'
targetDir= '/home/sliceruser/data'
def unpackk(zipDir,targetDir):
    with ZipFile(zipDir, "r") as zip_ref:
        for name in zip_ref.namelist():
            #ignoring all corrupt files
            try:
                zip_ref.extract(name, targetDir)
            except BadZipFile as e:
                print(e)
    
    
    # with ZipFile(zipDir, 'r') as zipObj:
    #    # Extract all the contents of zip file in different directory
    #    zipObj.extractall(targetDir)
        
unpackk( '/home/sliceruser/picai_public_images_fold0.zip', targetDir)      
unpackk( '/home/sliceruser/picai_public_images_fold1.zip', targetDir)      
unpackk( '/home/sliceruser/picai_public_images_fold2.zip', targetDir)      
unpackk( '/home/sliceruser/picai_public_images_fold3.zip', targetDir)      
unpackk( '/home/sliceruser/picai_public_images_fold4.zip', targetDir)   

#create a dictionary of directories where key is the patient_id
rootdir = '/home/sliceruser/data/'

dirDict={}
for subdir, dirs, files in os.walk(rootdir):
    for subdirin, dirsin, filesin in os.walk(subdir):
        lenn= len(filesin)
        if(lenn>0):
            dirDict[subdirin.split("/")[4]]=filesin

labelsFiles=[]
labelsRootDir = '/home/sliceruser/labels/csPCa_lesion_delineations/human_expert/resampled/'
for subdir, dirs, files in os.walk(labelsRootDir):
    labelsFiles=files
    
#Constructing functions that when applied to each row will fill the necessary path data
listOfDeficientStudyIds=[]

def findPathh(row,dirDictt,keyWord,rootdir):
    patId=str(row['patient_id'])
    study_id=str(row['study_id'])
    #first check is such key present
    if(patId in dirDictt ):
        filtered = list(filter(lambda file_name:   (keyWord in file_name and  study_id  in  file_name  ), dirDictt[patId]  ))
        if(len(filtered)>0):
            return os.path.join(rootdir,  patId, filtered[0] )
        else:
            print(f"no {keyWord} in {study_id}")
            listOfDeficientStudyIds.append(study_id)
            return " " 
    listOfDeficientStudyIds.append(study_id)
    return " "
    

def addPathsToDf(dff, dirDictt, keyWord):
    return dff.apply(lambda row : findPathh(row,dirDictt ,keyWord,rootdir )   , axis = 1)

df['t2w'] =addPathsToDf(df,dirDict, 't2w')
df["adc"] = addPathsToDf(df,dirDict, 'adc')
df["cor"] = addPathsToDf(df,dirDict, 'cor')
df["hbv"] = addPathsToDf(df,dirDict, 'hbv')
df["sag"] = addPathsToDf(df,dirDict, 'sag')
#now  resampled labels are in separate directory


def findResampledLabel(row,labelsFiles):
    patId=str(row['patient_id'])
    study_id=str(row['study_id'])
    filtered = list(filter(lambda file_name:   (study_id  in  file_name  ), labelsFiles ))
    if(len(filtered)>0):
        return os.path.join(labelsRootDir, filtered[0])
    listOfDeficientStudyIds.append(study_id)    
    return " "
    
        
df["reSampledPath"] =  df.apply(lambda row : findResampledLabel(row,labelsFiles )   , axis = 1)  

def isAnythingInAnnotated(row):
    reSampledPath=str(row['reSampledPath'])
    if(len(reSampledPath)>1):
        image = sitk.ReadImage(reSampledPath)
        nda = sitk.GetArrayFromImage(image)
        return np.sum(nda)
    return 0

df["isAnythingInAnnotated"]= df.apply(lambda row : isAnythingInAnnotated(row), axis = 1)  

#marking that we have something lacking here
df["isAnyMissing"]=df.apply(lambda row : str(row['study_id']) in  listOfDeficientStudyIds  , axis = 1) 

df.to_csv(os.path.join('/home/sliceruser/labels', 'processedMetaData.csv')) 