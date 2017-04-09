#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 00:13:59 2017

@author: shubham
"""
#MY IMPORTS
import database_operations as loader
import base_file as bfl
import warnings

import numpy
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop,SGD
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import load_model
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping

K.set_epsilon(1e-6)

def getCustomModelFrame (img_shape,nb_classes,convs=None,
                         fcls=None):
    assert(len(img_shape)==3)
    input_img = Input(shape=(img_shape[2],img_shape[0],img_shape[1]))
    
    assert(len(convs)>=1 and len(fcls)>=1)
    
    if(len(convs)==1):
        c= Convolution2D(convs[0][0],3,3,activation='relu',border_mode='same')(input_img)#5
        for i in range(convs[0][1]-1):
             c = Convolution2D(convs[0][0], 3, 3, activation='relu', border_mode='same')(c)
                         
        joint_layer=MaxPooling2D((2,2),border_mode='same')(c)#8*4*4
        
    else:
        c = Convolution2D(convs[0][0], 3, 3, activation='relu', border_mode='same')(input_img)#1
        for i in range(convs[0][1]-1):
             c = Convolution2D(convs[0][0], 3, 3, activation='relu', border_mode='same')(c)
        
        c = MaxPooling2D((2, 2), border_mode='same')(c)#16*64*64
        
        convlayer=2
        
        while convlayer<=(len(convs)-1):
            c = Convolution2D(convs[convlayer-1][0], 3, 3, activation='relu', border_mode='same')(c)#2
            for i in range(convs[convlayer-1][1]-1):
                c = Convolution2D(convs[convlayer-1][0], 3, 3, activation='relu', border_mode='same')(c)
             
            c = MaxPooling2D((2, 2), border_mode='same')(c)#8*32*32
            convlayer+=1

        c= Convolution2D(convs[convlayer-1][0],3,3,activation='relu',border_mode='same')(c)#5
        for i in range(convs[convlayer-1][1]-1):
            c = Convolution2D(convs[convlayer-1][0], 3, 3, activation='relu', border_mode='same')(c)
        joint_layer=MaxPooling2D((2,2),border_mode='same')(c)
        
            
    
    
    FCN_layer= Flatten()(joint_layer)#d1
    
    if(len(fcls)==1):
        assert(fcls[0][0]>0 and fcls[0][1]>=0 and fcls[0][1]<=1)
        d=Dense(fcls[0][0], activation='sigmoid')(FCN_layer)#d2
        d=Dropout(fcls[0][1])(d)
    else:
        assert(fcls[0][0]>0 and fcls[0][1]>=0 and fcls[0][1]<=1)
        d=Dense(fcls[0][0], activation='sigmoid')(FCN_layer)#d2
        d=Dropout(fcls[0][1])(d)
        
        fcnl=2
        while fcnl<=len(fcls):
            assert(fcls[fcnl-1][0]>0 and fcls[fcnl-1][1]>=0 and fcls[fcnl-1][1]<=1)
            d=Dense(fcls[fcnl-1][0], activation='sigmoid')(d)#d3
            d=Dropout(fcls[fcnl-1][1])(d)
            fcnl+=1

    output=Dense(nb_classes,activation='softmax')(d)
    model=Model(input_img,output,name="model")
    print("type", type(model))
    
    #optimizer_fn= SGD(nesterov=True)
    optimizer_fn=RMSprop()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer_fn,
                  metrics=['accuracy'])
    return model


def getModelFrame (img_shape,nb_classes,model_number=0):
    
    signature= bfl.MODSIG[model_number]
    return getCustomModelFrame(img_shape,nb_classes,signature["convs"],
                               signature["fcls"])
    


def getOldModelFrame (img_shape,nb_classes):
    assert(len(img_shape)==3)
    input_img = Input(shape=(img_shape[2],img_shape[0],img_shape[1]))
    
    c = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(input_img)#1
    c = MaxPooling2D((2, 2), border_mode='same')(c)#16*64*64
    
    c = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(c)#2
    c = MaxPooling2D((2, 2), border_mode='same')(c)#8*32*32
    
    c = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(c)#2
    c = MaxPooling2D((2, 2), border_mode='same')(c)#8*32*32                
    
    c = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(c)#3
    c= MaxPooling2D((2, 2), border_mode='same')(c)#8*16*16
    
    c= Convolution2D(128,3,3,activation='relu',border_mode='same')(c)#5
    joint_layer=MaxPooling2D((2,2),border_mode='same')(c)#8*4*4
    
    FCN_layer= Flatten()(joint_layer)#d1
    d=Dense(100, activation='sigmoid')(FCN_layer)#d2
    d=Dropout(0.5)(d)
    
    
    d=Dense(80, activation='sigmoid')(d)#d3
    d=Dropout(0.5)(d)
    
    
    d=Dense(40, activation='sigmoid')(d)#d3
    d=Dropout(0.5)(d)
    #op_layer_num_neurons=input('Enter op layer neuron number: ')
    output=Dense(nb_classes,activation='softmax')(d)
    model=Model(input_img,output,name="model")
    print("type", type(model))
    
    #optimizer_fn= SGD(nesterov=True)
    optimizer_fn=RMSprop()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer_fn,
                  metrics=['accuracy'])
    return model

def getTrainedModel(model_to_train,dataset_loc,metadata_loc,
                    STORE_FOLDER, op_model_file_name,num_of_epoch=111,data_batch_size=None,training_batch_size=10,
                    create_metadata=True):             
    """
    dataset_loc: full paths required e.g. /home/user/dtop/test/file.h5
    same for metadata_loc
    "add kwargs: for md path"
    """
    
    
    with loader.hf.File(dataset_loc,'r') as f:
        num_epoch=num_of_epoch
        metadata=loader.loadMetaData(metadata_loc)
        nb_classes= int(metadata[0]["nb_classes"])
        next_batch=loader.getNextBatch
        #b_size=metadata[0]["dataset_shape"][0]
        if(data_batch_size==None):
            data_batch_size=metadata[0]["shape"][0]
        b_size=data_batch_size
        print("Batch_size:",b_size,"metadata[0]=",metadata[0])
        #wait=input("Enter 1 and proceed;")
        y=loader.np.zeros(shape=(b_size,nb_classes))
       
        x,y_tr=next_batch(b_size,f,metadata)
        x=x.astype('float32')
        x=x.transpose(0,3,1,2)
        x=(x-100)/255
        print("x:",x.shape)
        print("y:",y.shape)
        print(y_tr)
        #for j in range(len(y_tr)):
            #print()
            #y[j][y_tr[j][0]]=1
        y = np_utils.to_categorical(y_tr, nb_classes)
        print("Y:",y)
        history=model_to_train.fit(x,y,batch_size=training_batch_size,nb_epoch=num_epoch)
        generated_model_address=STORE_FOLDER+'/'+op_model_file_name+'.h5'
        model_to_train.save(generated_model_address)
        print("In getTrainedModel History= ",history)
        
        if create_metadata:
            specs=metadata[0]
            specs.update({"original_model_address":generated_model_address,
                   "dataset_trained_on":dataset_loc,
                   "num_of_epoch":num_of_epoch,
                   "data_batch_size":data_batch_size,
                   "training_batch_size":training_batch_size
                   })
            loader.generateMetaData(STORE_FOLDER+'/'+op_model_file_name+'_metadata',
                                    specs,metadata[1])
        
        
        
        return generated_model_address,STORE_FOLDER+'/'+op_model_file_name+'_metadata.txt'#metadata address
    



def normalizeAsDuringTraining (array,dtype='float32',axis_shift=100,scale_down_factor=255):
    assert(type(array)==type(loader.np.array([1])))
    array=array.transpose(0,3,1,2)
    array=array.astype(dtype)
    array= (array-axis_shift)/scale_down_factor
    return array





    
def evaluateModel (model_loc,test_dataset_loc,test_metadata_loc,use_whole_dataset=True,percentage_used=100):
    with loader.hf.File(test_dataset_loc,'r') as f:
         md=loader.loadMetaData(test_metadata_loc)
         
         if use_whole_dataset:    
             x,y=loader.getNextBatch(md[0]["dataset_shape"][0],f,md)
         else:
             assert(percentage_used<=100)
             num_samples= int((md[0]["dataset_shape"][0])*percentage_used*0.01)
             x,y=loader.getNextBatch(num_samples,f,test_metadata_loc)

         x_test=x.transpose(0,3,1,2)
         x_test=x_test.astype('float32')
         x_test=(x_test-100)/255
         print("X_Test.dtype",x_test.dtype)
         model= load_model(model_loc)
         pred_label=model.predict(x_test)
         wrong_count=0;
         for i in range(x.shape[0]):
            #cv.putText(x[i],str(y[i][0]),(60,90), cv.FONT_HERSHEY_COMPLEX, 1,(0,0,255),1)
            #cv.rectangle(x[i],(55,85),(100,100),(0,255,0),3)
            
            #print(pred_label.shape)
            #for itr in range (pred_label.shape[0]):
            pr=loader.np.argmax(pred_label[i])
            print("itr:",i," Pred:",pr,end=" ")
            print("Actual_LAbel:",y[i][0])
            if pr==y[i][0]:
                pass
            else:
                wrong_count+=1
         accuracy=(1-wrong_count/x.shape[0])*100
         print ("Wrong:",wrong_count," %age Acc:", accuracy)
         return accuracy
  
    
    
def getFacesForPrediction (img,prediction_img_shape):
    
    gray = loader.cv.cvtColor(img, loader.cv.COLOR_BGR2GRAY)
    face_cascade = loader.cv.CascadeClassifier(bfl.HAAR_CASCADE_CLASSIFIER_LOCATION)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#coordinates
    print('faces:\n',faces)
    num_of_faces=faces.shape[0]
    pims=prediction_img_shape
    print("Num:",num_of_faces)
    print("Faces:",type(faces),faces)
    i=-1
    face_array=loader.np.empty(shape=(num_of_faces,pims[0],pims[1],pims[2]),dtype=img.dtype)
    for (x,y,w,h) in faces:
        i+=1
        face_array[i]=loader.cv.resize(img[ y:y+h,x:x+w],(pims[0],pims[1]))
        #cv.imshow(str(i),face_array[i])
        
        print(face_array.shape)
        #cv.waitKey(0)
        #print(":",i,":",face_array)
    #cv.destroyAllWindows()
    loader.cv.destroyAllWindows() 
    return (face_array,faces)
    
    
def checkStatus (confidence_score):
    cs=confidence_score
    if (cs>=bfl.CT["blank"]) and (cs<bfl.CT["not_sure"]):
        return -1# No idea
    elif (cs>bfl.CT["not_sure"]) and cs<=bfl.CT["present"]:
        return 0# may be confused
    elif cs>bfl.CT["present"]:
        return 1#Recognized
        

def getLabelMap (list_of_tuples):
    dict_labels={}
    for items in list_of_tuples:
        dict_labels[items[0]]=items[1]
    return dict_labels


def labelFaces (model_loc,model_metadata_loc,image):
    md=loader.loadMetaData(model_metadata_loc)
    label_map= getLabelMap (md[1])
    print ("Metadata: ",md)
    img=image.copy()
    face_array,faces=getFacesForPrediction(img,md[0]["shape"])

    model= load_model(model_loc)
    face_array=normalizeAsDuringTraining(face_array)
    pred_labels=model.predict(face_array)
    
    confusion_list =[]# A list of list of the form [label_with_max_score, similar_label_1, similar_label_2,...]
    labels_and_confidences=[]# list of tuples like (0,0.98) where 0 is the predicted class label, with a confidence of 98%
    
    print("Pred_labels",pred_labels.shape)
    
    for i in range(face_array.shape[0]):
        max_score_label=loader.np.argmax(pred_labels[i])
        max_score=pred_labels[i][max_score_label]
        (x,y,w,h)=faces[i]
        if checkStatus(max_score)==-1:
            
            loader.cv.rectangle(img,(x,y),(x+w,y+h),bfl.clr["red"],5)
            loader.cv.putText(img,str(max_score_label)+":"+label_map[max_score_label],(x,y),
                              loader.cv.FONT_HERSHEY_COMPLEX_SMALL,1,bfl.clr["yellow"])
        elif checkStatus(max_score)==0:
            similar_list=[]
            for j in range (pred_labels.shape[1]):
                similarity_level=abs(pred_labels[i][max_score_label]-pred_labels[i][j])
                if  similarity_level< bfl.CT["similar"]:
                    similar_list.append(j)
            if len(similar_list)>0:
                confusion_list.append(similar_list)
            loader.cv.rectangle(img,(x,y),(x+w,y+h),bfl.clr["yellow"],5)
            loader.cv.putText(img,str(max_score_label)+":"+label_map[max_score_label],(x,y),
                              loader.cv.FONT_HERSHEY_COMPLEX_SMALL,1,bfl.clr["blue"])
            
        elif checkStatus(max_score)==1:
            
            loader.cv.rectangle(img,(x,y),(x+w,y+h),bfl.clr["green"],5)
            loader.cv.putText(img,str(max_score_label)+":"+label_map[max_score_label],(x,y),
                              loader.cv.FONT_HERSHEY_COMPLEX_SMALL,1,bfl.clr["yellow"])
        else:
            raise ValueError
        labels_and_confidences.append((max_score_label,max_score))
    
    
    return {"image":img, "label_map":md[1],
            "predicted_labels_and_confidences":labels_and_confidences,
            "prediction_matrix":pred_labels,
            "confusion_list":confusion_list}
            
                
            
                





def getTrainableData(dataset_loc,dataset_metadata_loc):
    with loader.hf.File(dataset_loc,'r') as f:
        metadata=loader.loadMetaData(dataset_metadata_loc)
        nb_classes= int(metadata[0]["nb_classes"])
        next_batch=loader.getNextBatch
        
        b_size=metadata[0]["dataset_shape"][0]#YOU were trapped here
        print("Batch_size:",b_size,"metadata[0]=",metadata[0])
        y=loader.np.zeros(shape=(b_size,nb_classes))
       
        x,y_tr=next_batch(b_size,f,metadata)
        x=normalizeAsDuringTraining(x)
        print("x:",x.shape)
        print("y:",y.shape)
        print(y_tr)
        y = np_utils.to_categorical(y_tr, nb_classes)
        return (x,y)
    
#function to aid saving models at intervals of accuracies or epoochs or mixed conditions
class lambdaHelper(Callback):
    def __init__(self,model_folder,model_filename,monitor='acc',
                 acc_begin_saving=60,update_delta_acc=5,
                  stop_delta_acc=2,
                 epoch_tolerance=100, verbose=0,erase=True):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.acc_begin= acc_begin_saving/100
        self.delta_update= update_delta_acc/100
        self.delta_stop = stop_delta_acc/100
        
        self.epoch_tolerance=epoch_tolerance
        self.verbose = verbose
        
        print("lamnda_values",
              self.acc_begin,self.delta_update,self.delta_stop,
              self.epoch_tolerance)
        
        #dummy=input("Press 1 and Enter")
        
        self.last_update_epoch=0
        self.current_epoch=0
        self.current_acc=0.0
        self.last_update_acc=0.0
        self.best_acc=0.0
        self.folder= model_folder
        self.fn= model_filename
        self.erase=erase
        
        self.last_model_loc=None
        self.best_model_loc=None
        self.first_save=True
        self.model_list=[]
    
    
    def checkStopCriteria (self):
        """Update all vars before calling"""
        depoch= self.current_epoch-self.last_update_epoch
        dacc=self.current_acc-self.last_update_acc
        print("In StopCriteria")
        print(self.current_epoch,self.last_update_epoch,
              self.current_acc,self.last_update_acc,
              depoch,dacc)
        if (depoch>=self.epoch_tolerance and dacc<=self.delta_stop):
            return True
        else:
            return False
        
        
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        self.current_acc=current
        current_epoch=epoch
        self.current_epoch=current_epoch
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        if current_epoch is None:
            warnings.warn("Early stopping requires 'epoch' available!", RuntimeWarning)

        if self.first_save==False and self.checkStopCriteria():
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            print("\n\nAbout TO STOP\n\n")
            #dummy=input("Press 1 and Enter")
            if(self.current_acc>self.best_acc):
                print("Best Model Saved Before Stopping")
                self.model.save(self.folder+"/"+self.fn+"_"+str(current_epoch)+".h5")
                self.best_model_loc=self.folder+"/"+self.fn+"_"+str(self.current_epoch)+".h5"
                self.last_model_loc=self.best_model_loc
                self.last_update_epoch=self.current_epoch
                self.best_acc=self.current_acc
                self.model_list.append((self.last_model_loc,self.current_acc,self.last_update_epoch))
                
            self.model.stop_training = True
            
        else:
            if (self.current_acc>=self.acc_begin and self.first_save):
                print("\n\n------FIRST SAVE----------------\n\n")
                print("Acc",self.current_acc)
                #dummy=input("Press 1 and Enter")
                
                self.first_save=False
                self.model.save(self.folder+"/"+self.fn+"_"+str(current_epoch)+".h5")
                self.last_update_acc=self.current_acc
                self.last_update_epoch=self.current_epoch
                self.best_acc=self.current_acc
                self.last_model_loc=self.folder+"/"+self.fn+"_"+str(self.current_epoch)+".h5"
                self.best_model_loc=self.folder+"/"+self.fn+"_"+str(self.current_epoch)+".h5"
                self.model_list.append((self.last_model_loc,self.last_update_acc,self.last_update_epoch))
                dummy=input("First Save")
                #dummy=input("Press 1 and Enter")
            elif((self.current_acc-self.last_update_acc)>=self.delta_update and self.first_save==False):
                print("\n\n------MODEL SAVE----------------\n\n","Acc",self.current_acc)
                #dummy=input("Press 1 and Enter")
                self.model.save(self.folder+"/"+self.fn+"_"+str(current_epoch)+".h5")
                self.last_update_acc=self.current_acc
                self.last_update_epoch=self.current_epoch
                self.last_model_loc=self.folder+"/"+self.fn+"_"+str(self.current_epoch)+".h5"
                if(self.best_acc<=self.current_acc):
                    print("Acc",self.current_acc)
                    
                    print("\n\n------NEW BEST SAVE----------------\n\n","Acc",self.current_acc)
                    #dummy=input("Press 1 and Enter")
                    self.best_model_loc=self.folder+"/"+self.fn+"_"+str(self.current_epoch)+".h5"
                    self.best_acc=self.current_acc
                self.model_list.append((self.last_model_loc,self.last_update_acc,self.last_update_epoch))
            else:
                pass
                





####################
    
    



def getCustomOptimalTrainedModel(model_to_train,dataset_loc,metadata_loc,
                    STORE_FOLDER, op_model_file_name,
                    accuracy_to_start_saving=60,
                    save_at_accuracy_increase_of=2.5,
                    accuracy_increase_cutoff=1,
                    cutoff_num_epochs_from_last_update=200,
                    val_partition_fraction=None,val_dataset_loc=None,val_metadata_loc=None,
                    num_of_epoch=1000000,data_batch_size=None,training_batch_size=10,
                    create_metadata=True):             
    """
    dataset_loc: full paths required e.g. /home/user/dtop/test/file.h5
    same for metadata_loc
    "add kwargs: for md path"
    """
    num_epoch=num_of_epoch
    metadata=loader.loadMetaData(metadata_loc)
    
    if(data_batch_size==None):
        data_batch_size=metadata[0]["shape"][0]
    
    
    x,y= getTrainableData(dataset_loc,metadata_loc)
    
    if(val_partition_fraction==None and val_dataset_loc==None):
        save_at_intervals=lambdaHelper(STORE_FOLDER,op_model_file_name,'acc',
                                   accuracy_to_start_saving,save_at_accuracy_increase_of,
                                   accuracy_increase_cutoff,cutoff_num_epochs_from_last_update
                                   )
        history=model_to_train.fit(x,y,batch_size=training_batch_size,nb_epoch=num_epoch,
                                   callbacks=[save_at_intervals])
    elif (val_dataset_loc!=None):
        save_at_intervals=lambdaHelper(STORE_FOLDER,op_model_file_name,'val_acc',
                                   accuracy_to_start_saving,save_at_accuracy_increase_of,
                                   accuracy_increase_cutoff,cutoff_num_epochs_from_last_update
                                   )
        assert(val_metadata_loc!=None)
        xv,yv= getTrainableData(val_dataset_loc,val_metadata_loc)
        history=model_to_train.fit(x,y,batch_size=training_batch_size,nb_epoch=num_epoch,
                                   validation_data=(xv,yv),
                                   callbacks=[save_at_intervals])
    else:
        save_at_intervals=lambdaHelper(STORE_FOLDER,op_model_file_name,'val_acc',
                                   accuracy_to_start_saving,save_at_accuracy_increase_of,
                                   accuracy_increase_cutoff,cutoff_num_epochs_from_last_update
                                   )
        history=model_to_train.fit(x,y,batch_size=training_batch_size,nb_epoch=num_epoch,
                                   validation_split=val_partition_fraction,
                                   callbacks=[save_at_intervals])
        
        
    
    
    
    
    
    
    
    print("____MODEL_LIST______\n",save_at_intervals.model_list)
    if(save_at_intervals.best_model_loc!=None):
        print("Using Best One")
        generated_model_address=save_at_intervals.best_model_loc#STORE_FOLDER+'/'+op_model_file_name+'.h5'
        generated_model_md_address= generated_model_address[0:-3]+"_metadata"
        print(generated_model_address)
    else:
        generated_model_address=STORE_FOLDER+"/"+op_model_file_name+".h5"
        generated_model_md_address=STORE_FOLDER+"/"+op_model_file_name+"_metadata"
        model_to_train.save(generated_model_address)
    print("In getTrainedModel History= ",history)
    
    if create_metadata:
        specs=metadata[0]
        specs.update({"original_model_address":generated_model_address,
               "dataset_trained_on":dataset_loc,
               "data_batch_size":data_batch_size,
               "training_batch_size":training_batch_size
               })
        loader.generateMetaData(generated_model_md_address,
                                specs,metadata[1])
    
    #remove_later
    plot_summary=True
    if plot_summary:
        import matplotlib.pyplot as plt
        plt.plot(history.history["val_acc"])
        plt.plot(history.history["acc"])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
            
            
    return generated_model_address,generated_model_md_address+'.txt'#metadata address

######################################################################################################################




def getOptimalTrainedModel(model_to_train,dataset_loc,metadata_loc,
                    STORE_FOLDER, op_model_file_name,
                    accuracy_increase_cutoff=1,
                    cutoff_num_epochs_from_last_update=200,
                    val_partition_fraction=None,val_dataset_loc=None,val_metadata_loc=None,
                    num_of_epoch=1000000,data_batch_size=None,training_batch_size=10,
                    create_metadata=True):             
    """
    dataset_loc: full paths required e.g. /home/user/dtop/test/file.h5
    same for metadata_loc
    "add kwargs: for md path"
    """
    num_epoch=num_of_epoch
    metadata=loader.loadMetaData(metadata_loc)
    
    if(data_batch_size==None):
        data_batch_size=metadata[0]["shape"][0]
    b_size=data_batch_size
    
    x,y= getTrainableData(dataset_loc,metadata_loc)
    
    generated_model_address=STORE_FOLDER+'/'+op_model_file_name+'.h5'
    
    if(val_partition_fraction==None and val_dataset_loc==None):
        save_at_intervals = ModelCheckpoint(generated_model_address,
                                 monitor='acc', save_best_only=True,
                             mode='max')
        early_termination=EarlyStopping(monitor='acc', min_delta=accuracy_increase_cutoff/100,
                                                        patience=cutoff_num_epochs_from_last_update, verbose=1, mode='max')
        history=model_to_train.fit(x,y,batch_size=training_batch_size,nb_epoch=num_epoch,
                                   callbacks=[save_at_intervals,early_termination])
    elif (val_dataset_loc!=None):
        save_at_intervals = ModelCheckpoint(generated_model_address,
                                 monitor='val_acc', save_best_only=True,
                             mode='max')
        early_termination=EarlyStopping(monitor='val_acc', min_delta=accuracy_increase_cutoff/100,
                                                        patience=cutoff_num_epochs_from_last_update, verbose=1, mode='max')
        assert(val_metadata_loc!=None)
        xv,yv= getTrainableData(val_dataset_loc,val_metadata_loc)
        history=model_to_train.fit(x,y,batch_size=training_batch_size,nb_epoch=num_epoch,
                                   validation_data=(xv,yv),
                                   callbacks=[save_at_intervals,early_termination])
    else:
        save_at_intervals = ModelCheckpoint(generated_model_address,
                                 monitor='val_acc', save_best_only=True,
                             mode='max')
        early_termination=EarlyStopping(monitor='val_acc', min_delta=accuracy_increase_cutoff/100,
                                                        patience=cutoff_num_epochs_from_last_update, verbose=1, mode='max')
        
        history=model_to_train.fit(x,y,batch_size=training_batch_size,nb_epoch=num_epoch,
                                   validation_split=val_partition_fraction,
                                   callbacks=[save_at_intervals,early_termination])
        
        
    
    
    
    
    
    
    #model_to_train.save(generated_model_address)
    print("In getTrainedModel History= ",history)
    
    if create_metadata:
        specs=metadata[0]
        specs.update({"original_model_address":generated_model_address,
               "dataset_trained_on":dataset_loc,
               "num_of_epoch":num_of_epoch,
               "data_batch_size":data_batch_size,
               "training_batch_size":training_batch_size
               })
        loader.generateMetaData(STORE_FOLDER+'/'+op_model_file_name+'_metadata',
                                specs,metadata[1])
    
        
        
    return generated_model_address,STORE_FOLDER+'/'+op_model_file_name+'_metadata.txt'#metadata address
    
