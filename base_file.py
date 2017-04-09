#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 00:15:30 2017
GLOBAL FILE to be included in every file directly or indirectly to 
shre information about system/Global constants.
@author: shubham
"""

import os
base_path = os.path.dirname(os.path.realpath(__file__))
HAAR_CASCADE_CLASSIFIER_LOCATION=os.path.join(base_path,'haarcascade_frontalface_default.xml')





CT={"blank":0 , "not_sure":0.6, "present":0.8 ,  "similar":0.15 }#CONFIDENCE_THRESHOLD
# tweak_ May Add another level : can be resolved using closer look on rest of the scores

#Model Signatures
MODSIG_OLD=[          { "convs": (8,16,32,64,128),
                   "fcls":((100,0.5),(80,0.5),(40,0.5))},
                  
                  {"convs": (8,16,32,64,128,256),
                   "fcls": ((200,0.5),(150,0.5),(100,0.5))
                          }

 ]

MODSIG=[          { "convs": ((8,1),(16,1),(32,1),(64,1),(128,1)),
                   "fcls":((100,0.5),(80,0.5),(40,0.5))},   # indexing starts frm 0 
                  
                  {"convs": ((8,1),(16,1),(32,1),(64,1),(128,1),(256,1)),
                   "fcls": ((200,0.5),(150,0.5),(100,0.5))#1
                          },
                  
                  {"convs": ((16,2),(16,1),(32,2),(64,1),(128,1)),
                   "fcls": ((200,0.5),(150,0.5),(100,0.5))#2
                          },
                  
                  {"convs": ((16,2),(32,1),(32,1)),
                   "fcls": ((200,0.5),(150,0.5),(100,0.5))
                          }
 ]


clr = {"blue": (255,0,0), "white":(255,255,255),"green":(0,255,0),
       "red":(0,0,255),"sm_clr": (100,100,0),"yellow":(0,255,255)}