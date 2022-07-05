import importlib.util
import sys
spec = importlib.util.spec_from_file_location("resampl", "/home/sliceruser/data/piCaiCode/preprocessing/Resampling.py")
resampl = importlib.util.module_from_spec(spec)
sys.modules["resampl"] = resampl
spec.loader.exec_module(resampl)

from functools import partial

import importlib.util
import sys
spec = importlib.util.spec_from_file_location("Three_chan_baseline", "/home/sliceruser/data/piCaiCode/Three_chan_baseline.py")
Three_chan_baseline = importlib.util.module_from_spec(spec)
sys.modules["Three_chan_baseline"] = Three_chan_baseline
spec.loader.exec_module(Three_chan_baseline)





# 5     1000005
# 12    1000012
# 19    1000019
# 21    1000021

# resampl.copyDirAndOrigin()

# # import sys
# # sys.path.append('{HOME}/data/piCaiCode/preprocessing/Resampling.py')
# # import Resampling
# import pandas as pd
# import SimpleITK as sitk
# from subprocess import Popen
# import subprocess
# import SimpleITK as sitk
# import pandas as pd
# import multiprocessing as mp
# import functools
# from functools import partial
# import sys
# import os.path
# from os import path as pathOs
# import comet_ml
# from comet_ml import Experiment



# df = pd.read_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv')
# row = list(df.iterrows())[1]
# elacticPath='/home/sliceruser/Slicer/NA-MIC/Extensions-30822/SlicerElastix/lib/Slicer-5.0/elastix'
# reg_prop='/home/sliceruser/data/piCaiCode/preprocessing/registration/parameters.txt'      





# def reg_adc_hbv_to_t2w(row,colName,elacticPath,reg_prop,t2wColName,experiment=None):
#     """
#     registers adc and hbv images to t2w image
#     first we need to create directories for the results
#     then we suply hbv and adc as moving images and t2w as static one - and register in reference to it
#     we do it in multiple threads at once and we waiteach time the process finished
#     """

#     row=row[1]
#     study_id=str(row['study_id'])
    
#     patId=str(row['patient_id'])
#     path=str(row[colName])
#     outPath = path.replace(".mha","_for_"+colName)
#     result=pathOs.join(outPath,"result.0.mha")
#     print("**********  ***********  ****************")
#     print(result)

    
    
# #     /home/sliceruser/data/orig/10005/10005_1000005_adc_stand_medianSpac_for_adc_med_spac/result.0.mha
    
    
#     #     /home/sliceruser/data/orig/10005/10005_1000005_adc_stand_medianSpac_for_adc_med_spac/result.0.mha
# #     /home/sliceruser/data/orig/10005/10005_1000005_adc_stand_medianSpac_for_adc_med_spac
# #     /home/sliceruser/data/orig/10005/10005_1000005_hbv_stand_medianSpac_for_hbv_med_spac/result.0.mha
    
#     print(pathOs.exists(result))
#     #returning faster if the result is already present
#     if(pathOs.exists(outPath)):
#         if(experiment!=None):
#             experiment.log_text(f"already registered {colName} {study_id}")
        
#         print("registered already present")
#         return result     
#     else:
#         if(len(path)>1):
#             if(experiment!=None):  
#                 print(f"new register {colName} {study_id}")
#                 experiment.log_text(f"new register {colName} {study_id}")

#             cmd='mkdir '+ outPath
#             p = Popen(cmd, shell=True)
#             p.wait()
#             cmd=f"{elacticPath} -f {row[t2wColName]} -m {path} -out {outPath} -p {reg_prop}"
#             print(cmd)
#             try:
#                 p = Popen(cmd, shell=True)
#             except:
#                 print("error in patId")
#             p.wait()
#             return result
#         else:
#             return ""    
#     return ""
# roww=row[1]

# roww['adc_med_spac']

# partial(reg_adc_hbv_to_t2w,colName='adc_med_spac',elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w_med_spac')(row)
