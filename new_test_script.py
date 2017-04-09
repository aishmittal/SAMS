#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 17:09:52 2017

@author: shubham
"""

import database_operations as dsop
import cnn_model_generator as cmg
import database_operations as loader

img=dsop.cv.imread('group_1.png')


model_path="/home/shubham/Desktop/Test/test2/hfiles/bestmodels/model_gtech25_10_113.h5"
model_md_path="/home/shubham/Desktop/Test/test2/hfiles/bestmodels/model_gtech25_10_113_metadata.txt"




pred_results=cmg.labelFaces(model_path,model_md_path,img)
# returns a dictionary of the form
#{"image":image_with_bunding_boxes, "label_map":md[1],
#            "predicted_labels_and_confidences":labels_and_confidences,
#            "prediction_matrix":pred_labels,
#            "confusion_list":confusion_list}

print ("Prediction_results:\n",pred_results)



loader.cv.imshow("Img",pred_results["image"])
loader.cv.waitKey(0)
