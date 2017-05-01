#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 23:19:09 2017

@author: shubham
"""

import cv2 as cv
import h5py as hf
import numpy as np
import os


def generateMetaData(file_loc,specs_dict,label_map=[]):
    with open(file_loc+".txt","w") as f:
        f.write("SPECS:\n")
        for items in specs_dict:
            f.write(str(items)+"->"+str(specs_dict[items])+"\n")
        f.write(":SPECS_END\n")
        f.write("LABEL_MAP:\n")
        for items in label_map:
            f.write(str(items[0])+"->"+str(items[1])+"\n")
        f.write(":LABEL_MAP_END")

def createSingleBlockDataset (load_path,store_path,filename,image_shape=(50,50,3)):
    """filename without h5(e.g. 'test_dataset')
        default image_shape=(50,50,3)
    """
    num_files=0
    num_labels=-1
    img_dtype=None
    img_shape=None
    current_file=None
    temp_array=None
    label_array=None
    custom_shape_flag=True
    custom_shape=image_shape
    
    with hf.File(store_path+"/"+filename+".h5",'w') as f:
        for path,sub_dirs,files in os.walk(load_path):
            num_labels+=1
            current_folder=str(path).strip()
            # print("Folder:",current_folder)#remove_
            for file in files:
                num_files+=1
                current_file=current_folder+"/"+str(file).strip()
                
                # print("---> ",current_file)#remove_
                
        img=cv.imread(current_file)
        #calculate length of vector
        len_vector=1
        if ((img is not None) and custom_shape_flag):
            img_dtype=img.dtype
            img_shape=custom_shape
            for items in img_shape:
                len_vector*=items
            # print (len_vector)
        elif (img is not None):
            img_dtype=img.dtype
            img_shape=img.shape
            for items in img.shape:
                len_vector*=items
        else:
            raise ValueError('Not a recognize image type: %s' % (current_file))
            return
        
        try:
            temp_array=np.empty(shape=(num_files,len_vector),dtype=img.dtype)
            label_array=np.empty(shape=(num_files,1),dtype='uint16')
        except MemoryError:
            print("Not Enough Space to load all the images simultaneously.")
        
        
        idx=0
        itr=0
        inner_itr=0
        label=0#current_label
        last_label=0
        last_folder_empty=True
        label_map=[]
        for path,sub_dirs,files in os.walk(load_path):
            current_folder=str(path).strip()
            print("Folder:",current_folder)
            
            if(last_folder_empty):
                label-=1
            label+=1
            last_folder_empty=True
            inner_itr=0
            for file in files:
                itr+=1
                if(inner_itr==0):
                    label_map.append((label,current_folder[len(load_path)+1:]))
                
                current_file=current_folder+"/"+str(file).strip()
                print("---> ",current_file)
                
                img=cv.imread(current_file)
                if img is not None:
                    img=cv.resize(img,(custom_shape[0],custom_shape[1]))
                    temp_array[idx]=img.ravel()
                    label_array[idx]=label
                    idx+=1
                else:
                    print("Itr:",itr,"->","Error in:",current_file)
                last_folder_empty=False
                
                inner_itr+=1
                last_label=label
        
        
        
        if(itr== idx):
            pass
        else:
            print("One or more files have not been added.")
            
        temp_array=temp_array[:idx][:]
        label_array=label_array[:idx][:]
        
        
            
        rng_state = np.random.get_state()
        np.random.shuffle(temp_array)
        np.random.set_state(rng_state)
        np.random.shuffle(label_array)
        
        specs={"shape":img_shape,"dtype":img_dtype,"dataset_shape":temp_array.shape,"label_array_shape":label_array.shape,
               "nb_classes":last_label+1}
        f.create_dataset('predictors',data=temp_array,compression='gzip')
        f.create_dataset('labels',data=label_array,compression='gzip')
        #f.create_dataset('map',data=label_map)
        #f.create_dataset('specs',data=specs)
        generateMetaData(store_path+"/"+filename+"_metadata",specs,label_map)
        
        f.visit(print) #remove_    
                
                
                
                
    
def navigateDataset(file_location="/example_folder/example_file.h5",image_shape=(50,50,3),num_sample=0):
    with hf.File(file_location,'r') as f:
         X=f['predictors']
         Y=f['labels']
         print('again_see_RAM')
         x=X[:].reshape(X.shape[0],image_shape[0],image_shape[1],image_shape[2])
         y=Y[:].reshape(Y.shape[0],1)
         # print(y)
         for i in range(x.shape[0]):
            if(i==num_sample):
                break
            cv.putText(x[i],str(y[i][0]),(round(image_shape[0]),round(image_shape[1]/2)), cv.FONT_HERSHEY_COMPLEX, 1,(0,0,255),1)
            cv.imshow('Images',x[i])
            # cv.waitKey(0)    



def loadMetaData(metadata_file):
    '''returns A tuple of a dictionary and a list:: (specs{}, label_map[])'''
    specs={}
    label_map=[]
    with open(metadata_file) as f:
        a= str(f.readline()).strip()
        #print(a)
        if (a=="SPECS:"):
            
            while(True):
                a=str(f.readline()).strip()
                if(a==":SPECS_END"):
                    break;
                l=a.split('->')
                m=l[1]
                if(l[1][0]=="("):
                    m=[]
                    l[1]=l[1][1:-1]
                    l[1]=l[1].split(', ')
                    for items in l[1]:
                        m.append(int(items))
                    m=tuple(m)
                specs[l[0]]=m
        else:
            print("Could not read specifications.\n++",a,"end++\n")
        a=str(f.readline()).strip()
        #print(a)
        if (a=="LABEL_MAP:"):
            
            while(True):
                a=str(f.readline()).strip()
                if(a==":LABEL_MAP_END"):
                    break;
                l=a.split("->")
                label_map.append((int(l[0]),l[1]))
        else:
            print("Could not read label_map.\n")
    return (specs,label_map)
                
                



def read_in_chunks(begin_at,batch_size,f): 
    
     
     X=f['predictors']
     Y=f['labels']
     print('again_see_RAM')
     begin_at=0
     num_s=10#to_set
     x=X[begin_at:begin_at+num_s].reshape(num_s,592,896,3)
     # y=Y[begin_at:begin_at+num_s].reshape(num_s,1)
     print(y)
     for i in range(x.shape[0]):
        cv.putText(x[i],str(y[i][0]),(60,90), cv.FONT_HERSHEY_COMPLEX, 1,(0,0,255),1)
        #cv.rectangle(x[i],(55,85),(100,100),(0,255,0),3)
        #cv.putText()
        cv.imshow('Win',x[i])
        # cv.waitKey(0)
     

            



def getNextBatch (batch_size,file_object,metadata,dtype='float32'):
    if 'count' not in getNextBatch.__dict__:
        getNextBatch.count=0
    
    if getNextBatch.count==0:
        getNextBatch.f=file_object
        getNextBatch.sp=0# starting point
        getNextBatch.specs=metadata[0]
        getNextBatch.max=getNextBatch.specs["dataset_shape"][0]
        getNextBatch.img_shp=getNextBatch.specs["shape"]
        # print("getShape",getNextBatch.img_shp)
    print("Current count:",getNextBatch.count,"f=",getNextBatch.f,"Actual:",file_object)
    if getNextBatch.f==file_object:
        try:
            X=getNextBatch.f['predictors']
            Y=getNextBatch.f['labels']
            begin_at=getNextBatch.sp
            if getNextBatch.max==(getNextBatch.sp+batch_size):
                x=X[begin_at:getNextBatch.max].reshape(batch_size,getNextBatch.img_shp[0],
                   getNextBatch.img_shp[1],getNextBatch.img_shp[2])
                y=Y[begin_at:getNextBatch.max].reshape(batch_size,1)
                getNextBatch.sp=0
                getNextBatch.count+=1
                
                return (x,y)
            elif getNextBatch.max<(getNextBatch.sp+batch_size):
                if batch_size>getNextBatch.max:
                    batch_size=getNextBatch.max
                x=np.concatenate((X[begin_at:getNextBatch.max],X[0:(-getNextBatch.max+begin_at+batch_size)]),axis=0).reshape(batch_size,
                                getNextBatch.img_shp[0],getNextBatch.img_shp[1],getNextBatch.img_shp[2])
                y=np.concatenate((Y[begin_at:getNextBatch.max],Y[0:(-getNextBatch.max+begin_at+batch_size)]),axis=0).reshape(batch_size,1)
                
                getNextBatch.sp=(-getNextBatch.max+begin_at+batch_size)
                getNextBatch.count+=1
                return (x,y)
            else:
                x=X[begin_at:begin_at+batch_size].reshape(batch_size,getNextBatch.img_shp[0],
                   getNextBatch.img_shp[1],getNextBatch.img_shp[2])
                y=Y[begin_at:begin_at+batch_size].reshape(batch_size,1)
                
                getNextBatch.sp=(begin_at+batch_size)
                getNextBatch.count+=1
                return (x,y)
        except IOError:
            print("Problem_with file object:",getNextBatch.f,"In getNextBatch()")
    else:
        getNextBatch.count=0
        return(getNextBatch(batch_size,file_object,metadata,dtype))


def partitionDataset (source,source_md,part_ratio=(60.0,20.0),
                      dest_folder_train=None,filename_train=None,
                      dest_folder_test=None,filename_test=None,
                      create_valdn=True,dest_folder_validation=None,filename_validation=None):
    """source= '/example_folder/examplefile.h5' (dataset to be partioned)
       source_md='/example_folder/examplefile_metadata.txt'
       dest_test='/example_folder/examplefile.h5' (test dataset will be saved here)
       
       part_ratio : (%age for training, test)
       create_val=False #if you don't want to ceate a validation dataset
       dest_validation
    """
    assert (sum(part_ratio)<=100.0)
    for items in part_ratio:
        assert(items>0)
    
    idx=-1
    char=source[idx]
    while char!= '/':
        idx-=1
        char=source[idx]
    dest_t=source[0:idx+1]
    split_list=source.split('/')
    split_list=split_list[len(split_list)-1].split(".")
    fn=split_list[0]#file name of original data to be partitioned
    
    if dest_folder_train ==None:
        dest_folder_train=dest_t
    if dest_folder_test ==None:
        dest_folder_test=dest_t
    if dest_folder_validation==None:
        dest_folder_validation=dest_t
    if filename_train==None:
        filename_train=fn+"_train"
    if filename_test== None:
        filename_test=fn+"_test"
    if filename_validation==None:
        filename_validation=fn+"_validation"
    
    md=loadMetaData(source_md)# be sure to cast the bjects to int which are supposed to be int.
    
    
    num_samples= md[0]['dataset_shape'][0]
    max_idx=num_samples-1
    assert(num_samples>=2)
    slice1=int (max_idx*part_ratio[0]*0.01)
    slice2=slice1+int(max_idx*part_ratio[1]*0.01)
    assert(slice2<=max_idx and slice2>slice1)
    
    flags=[True,True,create_valdn]
    if(sum(part_ratio)==100 ):
        slice2=max_idx

    
    train_size=slice1+1
    test_size=slice2-slice1
    valdn_size=max_idx-slice2
    sizes=[train_size,test_size,valdn_size]
    slices=[-1,slice1,slice2,max_idx]
    fnms=[dest_folder_train+"/"+filename_train,
          dest_folder_test+"/"+filename_test,
          dest_folder_validation+"/"+filename_validation]
    
    

    with hf.File(source,'r') as f:
        X=f['predictors']
        Y=f['labels']
    
        for i in range (len(fnms)):
            if(flags[i] and sizes[i]>0):
                x=X[slices[i]+1:slices[i+1]+1]
                y=Y[slices[i]+1:slices[i+1]+1]
                print(i,"from",slices[i]+1,"to", slices[i+1]+1)
                with hf.File(fnms[i]+".h5",'w') as f:
                    assert(sizes[i]==x.shape[0])
                    f.create_dataset('predictors',data=x,compression='gzip')
                    f.create_dataset('labels',data=y,compression='gzip')
                new_md=(md[0].copy(),md[1].copy())
                
                new_md[0]["label_array_shape"]=(y.shape[0],y.shape[1])
                new_md[0]["dataset_shape"]=(x.shape[0],x.shape[1])
                generateMetaData(fnms[i]+"_metadata",new_md[0],new_md[1])