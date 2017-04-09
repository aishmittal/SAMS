#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:41:19 2017

@author: shubham
"""

import numpy as np
import cv2 as cv
import base_file as bfl
face_cascade = cv.CascadeClassifier(bfl.HAAR_CASCADE_CLASSIFIER_LOCATION)
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
img_loaded = cv.imread('1.jpg')
np.set_printoptions(threshold=np.inf)
#cv.imshow('img',img)
#cv.destroyAllWindows()
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
##faces = face_cascade.detectMultiScale(gray, 1.3, 5)
##for (x,y,w,h) in faces:
##    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
##    roi_gray = gray[y:y+h, x:x+w]
##    roi_color = img[y:y+h, x:x+w]
##    eyes = eye_cascade.detectMultiScale(roi_gray)
##    for (ex,ey,ew,eh) in eyes:
##        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#cv.imshow('GRAy',gray)
#
#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#for (x,y,w,h) in faces:
#    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color = img[y:y+h, x:x+w]
#cv.imshow('img',img)
#cv.waitKey(0)
#cv.destroyAllWindows()
#img2=img.copy()
#face_list=[]
#i=-1
#for (x,y,w,h) in faces:
#   face_list.append(cv.resize(img2[ y:y+h,x:x+w],(50,50)))
#   i+=1
#   cv.imshow(str(i),face_list[i])
#    
#cv.waitKey(0)
#cv.destroyAllWindows()  
#    
    
    
    
def getFacesForPrediction (img,prediction_img_shape):
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    num_of_faces=faces.shape[0]
    pims=prediction_img_shape
    print("Num:",num_of_faces)
    print("Faces:",type(faces),faces)
    i=-1
    face_array=np.empty(shape=(num_of_faces,pims[0],pims[1],pims[2]),dtype=img.dtype)
    for (x,y,w,h) in faces:
        i+=1
        cv.rectangle(img,(x,y),(x+w,y+h),bfl.clr["red"],5)
        cv.putText(img,str(i)+"imag",(x,y),cv.FONT_HERSHEY_COMPLEX_SMALL,1,bfl.clr["yellow"])
        cv.putText(img,str(i)+"imag",(x,y+h),cv.FONT_HERSHEY_COMPLEX_SMALL,1,bfl.clr["blue"])
        face_array[i]=cv.resize(img[ y:y+h,x:x+w],(pims[0],pims[1]))
        #cv.imshow(str(i),face_array[i])
        
        print(face_array.shape)
        #cv.waitKey(0)
        #print(":",i,":",face_array)
    cv.destroyAllWindows()
    img=np.empty(shape=img.shape)
    cv.imshow('AfterIn',img)
    cv.waitKey(0)
    cv.destroyAllWindows() 
    return (face_array,faces)
    
    
    
getFacesForPrediction(img_loaded,(50,50,3)) 

cv.imshow('After',img_loaded)
cv.waitKey(0)
cv.destroyAllWindows() 

def normalizeAsDuringTraining (array,dtype='float32',axis_shift=100,scale_down_factor=255):
    assert(type(array)==type(np.array()))
    array=array.transpose(0,3,1,2)
    array=array.astype(dtype)
    array= (array-axis_shift)/scale_down_factor
    return array



def evaluateModel (model_loc,model_metadata_loc,image):
    
    face_array,faces=getFacesForPrediction(image,)

    print("X_Test.dtype",x_test.dtype)
    model= load_model(model_loc)
    pred_label=model.predict(x_test)
    wrong_count=0;

        
        
    
    
    
    
    
    
    
    
    
