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

files=[]
for subdir, dirs, files in os.walk(targetDir):
    files=files


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