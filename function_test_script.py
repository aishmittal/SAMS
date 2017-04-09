#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 19:06:06 2017
TEST SCRIPT
@author: shubham
"""

import database_operations as dsop
import cnn_model_generator as cmg
import database_operations as loader
from keras.utils import np_utils
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop,SGD
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import load_model
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping




def miniCallback (epch,lgs,extra_arg=[1]):
    print("\n---------MiniCback-----------\n")
    print("Epoch:",epch)
    print("Logs:",lgs)
    if lgs["acc"]>5:
        extra_arg[0]=input("Enter Your DOB")
    print("Extra:",extra_arg)



#MY GLOBAL CONSTANTS
LOAD_PATH='/home/aishwarya/Documents/sams/data/images/webcam'#load data from here
STORE_PATH='/home/aishwarya/Documents/sams/data'#store data in this folder
DSETP=STORE_PATH+'/datasets'
MODLP=STORE_PATH+'/models'

DSETFN="ssd"
MFN="model_"+DSETFN# model filename
NEPOCH=10
DBS=500# data batch size
TBS=10# training batch size
DMY="no val"# dummmy

#load_path: 
#DSETP : dataset path where you want to save. You will pass this path. f
# dsop.createSingleBlockDataset(LOAD_PATH,DSETP,DSETFN,(50,50,3))
# md=dsop.loadMetaData(DSETP+'/'+DSETFN+'_metadata.txt')
# print(md[0],md[0]["shape"])

# print("PATH:",DSETP+"/"+DSETFN+".h5",md[0]["shape"])
#dsop.navigateDataset(DSETP+"/"+DSETFN+".h5",md[0]["shape"],0)

# dsop.partitionDataset(DSETP+"/"+DSETFN+".h5",DSETP+"/"+DSETFN+"_metadata.txt",(80,20))

# md=dsop.loadMetaData(DSETP+'/'+DSETFN+'_train_metadata.txt')
# model= cmg.getModelFrame(md[0]["shape"],int(md[0]["nb_classes"]),3)
# DBS= md[0]["dataset_shape"][0]
# MFN=MFN+"_"+str(NEPOCH)
#model_path, model_md_path=getTrainedModel(model,DSETP+"/"+DSETFN+"_train.h5",DSETP+"/"+DSETFN+"_train_metadata.txt",
#                              MODLP,MFN,NEPOCH,DBS,TBS)


MODEL_LOC=MODLP+"/"+MFN+".h5"
TD_LOC=DSETP+"/"+DSETFN+"_test.h5"
TD_MD_LOC=DSETP+"/"+DSETFN+"_test_metadata.txt"


# tu ye use kr lena for training the model so that you can use it for prediction later.
#It will return the path of the best model. Though the model will be saved at othercheck points also(e.g. here
#once the accuracy reaches 70 )
# model_path, model_md_path=cmg.getCustomOptimalTrainedModel(model,DSETP+"/"+DSETFN+"_train.h5",
#                                                            DSETP+"/"+DSETFN+"_train_metadata.txt",
#                                                      MODLP,MFN,70,2,
#                                                      0.8,15,0.2,TD_LOC,TD_MD_LOC,10000)
# print(model_path)



MODEL_LOC=MODLP+"/"+MFN+".h5"
TD_LOC=DSETP+"/"+DSETFN+"_test.h5"
TD_MD_LOC=DSETP+"/"+DSETFN+"_test_metadata.txt"


#It's your job to ensure that the test data sizes are appropriate i.e. same as the training data
#cmg.evaluateModel ("/home/shubham/Desktop/Test/test2/hfiles/models/model_gtech25_10_113.h5",TD_LOC,TD_MD_LOC)
cmg.evaluateModel ('/home/aishwarya/Documents/sams/data/models/model_ssd_10_509.h5',TD_LOC,TD_MD_LOC)
#cmg.evaluateModel (MODEL_LOC,TD_LOC,TD_MD_LOC)

img=dsop.cv.imread('2.jpg')

cmg.labelFaces('/home/aishwarya/Documents/sams/data/models/model_ssd_10_509.h5','/home/aishwarya/Documents/sams/data/models/model_ssd_10_509_metadata.txt',img)





"""
 [('/home/shubham/Desktop/Test/test2/hfiles/models/model_gtech5_10_71.h5', 0.83333333333333337, 71), ('/home/shubham/Desktop/Test/test2/hfiles/models/model_gtech5_10_90.h5', 0.89999999602635705, 90), ('/home/shubham/Desktop/Test/test2/hfiles/models/model_gtech5_10_112.h5', 0.93333333730697632, 112)]
 
 """





        
