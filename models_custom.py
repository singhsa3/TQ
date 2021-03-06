#!/usr/bin/env python
# coding: utf-8

# In[1]:
#export
from tensorflow.tools.docs import doc_controls
import librosa, librosa.display
import matplotlib.pyplot as plt

import os
import glob
import cv2
import pickle5 as pickle
from PIL import Image as im
import argparse
import math
import sys
import time
import copy
import numpy as np
import keras
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization #, regularizers
from keras.layers.noise import GaussianNoise
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import models_custom 
from keras.models import load_model
import pandas as pd
import psutil
import models_custom as mcs
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization #, regularizers
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,Conv1D
from os.path import exists
from keras.layers import Bidirectional, GlobalMaxPool1D, Input,Dense, Dropout, Flatten, Activation, Attention 
from keras.layers import BatchNormalization,TimeDistributed, Reshape,RepeatVector,Permute, Multiply #, regularizers
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, MaxPooling1D
from keras.layers import LSTM 
from keras_self_attention import SeqSelfAttention
import tensorflow_addons as tfa


# In[6]:
def get_memory():
    import nvidia_smi
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)
    
    
#export
# Here I an creating real time labels
def create_label(therapist,emotion, pathG,balanced=1):
    with open(pathG+'/duration.pickle', 'rb') as f:
            dur=pickle.load(f)
    with open(pathG+'/fileM.pickle', 'rb') as handle:
        fileM= pickle.load(handle)
    wavM = [s.replace('pickle', 'wav') for s in fileM]
    #dur={key:val for key, val in dur.items() if val <= t}  
    fls =list(set(dur.keys()).intersection(set(wavM)))
    #print(len(fls))
    def data_balanced (Alemu):       
        emt= Alemu[Alemu.emotion==1]
        emtN=Alemu[Alemu.emotion==0]
        sz = np.min([emt.shape[0],emtN.shape[0]])
        Alemu=pd.concat([emt.sample(sz), emtN.sample(sz)])
        return Alemu.sample(frac=1)
    df= pd.read_csv(pathG+'/labelsConsolidated.csv')
    df = df[df['name'].isin(fls)]
    df= df[(df.therapist==therapist) & (df.emotion_type==emotion)]
    df['emotion'] = df['rating'].apply(lambda x: 1 if (x.lower()=="high" or x.lower()=="medium") else 0)
    if balanced==1:
        df=data_balanced (df)
    return df
import pandas as pd
# This was experimental, where for each time points, I was only taking vital statistics
def getZ(c): 
    c=np.array(c).T    
    #print("1st",c.shape)
    b = np.max(c)
    a = np.min(c)
    c =(c-a)/(b-a)
    c= pd.DataFrame(c)
    #c= c.astype('float64')
    #print(type(c))
    c=c.describe(percentiles=[.1, .25,.5,.75,.90])
    #print("2nd",c.shape)
    #print(c.shape)
    c=c.drop(['count'])
    c=np.array(c.T) 
    #print("last",c.shape)
    return  c

def getZTime(c):  # Describing for each featured along time scales
    #print("I am hereABC")
    #print(c.shape)
    c=np.array(c)    
    #print(c.shape)
    b = np.max(c)
    a = np.min(c)
    c =(c-a)/(b-a)
    c= pd.DataFrame(c)
    #c= c.astype('float64')
    #print(type(c))
    c=c.describe(percentiles=[.1, .25,.5,.75,.90])
    #print(c.shape)
    c=c.drop(['count'])
    c=np.array(c)  
    #print("done")
    return  c

# Normalizing data and creating an array.
# This used for image based as well as generalized array.
# Note: I am no longer droping any file. If file lenght is more then l95, I am simply truncating it.
def create_arr(fl,l95, height,width,img, scaleImg): 
    with open(fl,"rb") as f:
        x=pickle.load(f) 
    l1 = x[0].shape[0]
    try:
        if type(x) is not np.ndarray:
            x=x.cpu().detach().numpy()
        l95=int(np.ceil(l95/10))*10
        #print("l95 :"+str(l95))
        if l95>=l1:
            x=np.pad(x, ((0,0), (0,l95-l1), (0, 2)), 'constant') 
        else:
            x=x[:,:l95,:]
            x=np.pad(x, ((0,0), (0,0), (0, 2)), 'constant')
            #print(x.shape)
        c= x[0]
        b = np.max(c)
        a = np.min(c)
        c = (c-a)/(b-a)
      
        if img==1:
            if scaleImg>1:
                height=int(x.shape[1]/scaleImg)
                width=int(x.shape[2]/scaleImg)
            #print(height,width)
            #c= 255*c
            #c= c.astype(np.uint8)
            data=im.fromarray(x[0]) # Note this is not c
            data = data.resize((height,width), im.LANCZOS )
            arr=np.array(data)
            arr = arr.reshape(height,width,1)
            x=None
        elif scaleImg==-1:
            
            arr = getZTime(c)
            
            #print(arr.shape)
        else:
            if l95>5001: # If the length is >5000, I am just taking statistic                
                arr=getZ(c)
            else:
                arr =c             
        arr = arr.reshape(arr.shape[0],arr.shape[1],1)
        #print(arr.shape)
    except Exception as e: # work on python 2.x
        #print("oops")
        #print(str(e))
        arr= None     
    return arr


# In[11]:


from os.path import exists
# We are combining all
def imgconvert(df, l95,DATASET_PATHS,height, width, img,scaleImg=1):
    df=df.drop(columns=['level_0'])
    df=df.reset_index()
    arrlist0=[]
    for DATASET_PATH in DATASET_PATHS:
        fl= DATASET_PATH +"/"+df.iloc[0]['name2']           
        if exists(fl)==True:
            arrlist0.append(create_arr(fl,l95, height,width,img,scaleImg))
        else:
            print(fl+" Does not exists")
    arr= np.concatenate(tuple(arrlist0), axis=2)   
    arr= np.expand_dims(arr, axis=0)    
    idx=[]
    for i,row in df.iterrows():     
        arrlist2=[]        
        for DATASET_PATH in DATASET_PATHS:            
            fl = DATASET_PATH+"/"+row['name2'] 
            if exists(fl)==True:
                arrlist2.append(create_arr(fl,l95, height,width,img,scaleImg)) 
            else:
                print(fl+" Does not exists")
               
        try:
            #arr2=create_arr(fl,l95, height,width,img)            
            arr2 = np.concatenate(tuple(arrlist2), axis=2)
            #arr2 = arr2.reshape(1, height,width,len(DATASET_PATHS))
            arr2= np.expand_dims(arr2, axis=0)
            arr = np.vstack((arr,arr2))

                
            idx.append(i)
            x=None            
        except Exception as e: 
            #print("dhat teri")
            print(str(e))
    
    arr =np.delete(arr, (0), axis=0) # First row was dummy row
    labels=np.array(df.emotion.iloc[idx])
    return arr , labels 
# In[12]:

# Function to divide a data frame in chunks
def split_dataframe(df, chunk_size): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks
# This function takes the layers files and combines them into chunks. frm is the chunk that is passed in
# This is independent of labeled datasets
def getarr(frm,w,h,flt,fl):
    #print(w,h)
    arr=np.zeros(shape=(1,w,h,flt))
    
    lbl=[]
    for i,row in frm.iterrows():
        a=row["name2"]
        #fl=pathG+'/w2v2/three/'+a
        fl2=fl+'/'+a
        try:
            with open(fl2,'rb') as f:
                arr2= pickle.load(f)
                #print(arr2.shape)
                arr = np.vstack((arr,arr2))
                print(i,arr.shape, psutil.virtual_memory())
                lbl.append(row['name'])
                #print(b.shape)
        except Exception as e:
            print(str(e))
    arr =np.delete(arr, (0), axis=0) # First row was dummy row
    return arr, lbl
# This function takes in output of getarr function, deletes items based on the labeled dataset for a therapist/emotion
# The output is what will go into the deep learning models
def moddf(df,oldX,oldy ):
    lst=[]
    names=list(df['name'])
    for i,arr in enumerate(oldX):
        if oldy[i] not in names:    
            lst.append(i)
    #print(len(lst))
    if len(lst)>0:
        newywav=np.delete(np.array(oldy), lst, axis=0)    
        newX=np.delete(oldX, lst, axis=0)
    else:
        newywav=np.array(oldy)
        newX=oldX
    newy=[]
    for a in newywav:
        val =df[df.name== a].emotion.values[0]
        newy.append(val)

    newy=np.array(newy).astype(int) 
    #newy = to_categorical(newy)
    return newX, newy

    
def create_LSTM_Attn( img_rows, img_cols, filters,num_classes = 2,noise = 1,droprate=0.25):
    input_shape = ( img_rows, img_cols,filters)
    N = Input(shape=input_shape)
    O= Conv2D(1, kernel_size=(1, 1), strides=(1, 1),padding='same')(N) 
    #print("I am here O")
    #print(O.shape)
    act='ReLU'
    P = Reshape((O.shape[1],O.shape[2]*O.shape[3]))(O)
    #print("I am here P")
    #print(P.shape)
   
    lstm = LSTM(64, return_sequences=True,dropout=droprate)(P)
    lstm = LSTM(64, return_sequences=True,dropout=droprate) (lstm)

    
    sent_representation=SeqSelfAttention(attention_activation='sigmoid',
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,    
        kernel_regularizer=keras.regularizers.l2(1e-6),
        use_attention_bias=False,)(lstm)

    sent_representation = Flatten()(sent_representation)

    d1= Dense(25088,activation= act, use_bias=True)(sent_representation)
    d2= Dropout(droprate)(d1)
    d3= Dense(2048,activation= act, use_bias=True)(d2)
    d4= Dropout(droprate)(d3)
    d5= Dense(1, activation='sigmoid')(d4)
    
    model = Model(inputs=N, outputs=d5)
    model.compile(loss=keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(), tfa.metrics.F1Score(num_classes=1, average='macro',threshold=0.5)])
    return model

def create_conv1dlstmAttn( img_rows, img_cols, filters,num_classes = 2,noise = 1,droprate=0.25):
    input_shape = ( img_rows, img_cols,filters)
    inputs = Input(input_shape)
    x= Conv2D(1, kernel_size=(1, 1), strides=(1, 1),padding='same')(inputs) 

    act='ReLU'

    levels = 8
    
    x = Reshape((x.shape[1],x.shape[2]*x.shape[3]))(x)
    
    
    for level in range(4):    
        x = Conv1D(levels, 3,strides=1,padding='same')(x)    
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = MaxPooling1D(pool_size=3, strides=1, padding='same' )(x) 
        levels *= 2
   
    O = Reshape((x.shape[1],x.shape[2]))(x)

    P= LSTM(64,return_sequences=True,dropout=droprate)(O)

    lstm = LSTM(64, return_sequences=True,dropout=droprate)(P)
    #lstm = LSTM(64, return_sequences=True,dropout=0.25)(lstm)
    
    lstm=SeqSelfAttention(attention_activation='sigmoid', attention_width=5,
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,    
        kernel_regularizer=keras.regularizers.l2(1e-6),
        use_attention_bias=False,)(lstm)      
    
    
    x=SeqSelfAttention(attention_activation='sigmoid')(lstm)
    x = Flatten()(x)
    d1= Dense(25088,activation= act, use_bias=True)(x)
    d2= Dropout(droprate)(d1)
    d3= Dense(2048,activation= act, use_bias=True)(d2)
    d4= Dropout(droprate)(d3)
    d5= Dense(1, activation='sigmoid')(d4)

    
    model = Model(inputs=inputs, outputs=d5)
    
    model.compile(loss=keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(),tfa.metrics.F1Score(num_classes=1, average='macro',threshold=0.5)])
    return model

def create_3CNN2LSTM_Attn( img_rows, img_cols, filters,num_classes = 2,noise = 1,droprate=0.25):
    
    input_shape = ( img_rows, img_cols,filters)
    visible = Input(shape=input_shape)


    act='ReLU'
    #CNN Block 1
    A= Conv2D(64, kernel_size=(3, 3), strides=(1, 1),padding='same')(visible) 
    B = BatchNormalization()(A)
    C= Activation(act)(B)
    D= MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C)


    #CNN Block2
    E =Conv2D(64, kernel_size=(3, 3),padding='same', strides=(1, 1))(D) 
    F= Conv2D(64, kernel_size=(3, 3),padding='same', strides=(1, 1))(E) 
    G =BatchNormalization()(F)
    H =Activation(act)(G)
    I= MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(H)

    #CNN Block3
    J= Conv2D(128, kernel_size=(3, 3),padding='same', strides=(1, 1))(I) 
    K = Conv2D(128, kernel_size=(3, 3),padding='same', strides=(1, 1))(J)
    L = BatchNormalization()(K)
    M = Activation(act)(K)
    
    N = MaxPooling2D(pool_size=(4, 4), strides=(2, 2))(M)

    #LSTM Layers 

    O = Reshape((N.shape[1], N.shape[2]* N.shape[3]))(N)

    P= LSTM(64,dropout=droprate,return_sequences=True)(O)   
  
    lstm = LSTM(64,dropout=droprate,return_sequences=True)(P)
    
    #https://alvinntnu.github.io/python-notes/nlp/seq-to-seq-m21-sentiment-attention.html
    #https://stackoverflow.com/questions/42918446/how-to-add-an-attention-mechanism-in-keras
    # compute importance for each step
    sent_representation=SeqSelfAttention(attention_activation='sigmoid')(lstm)
    f = Flatten()(sent_representation) 
    
    d1= Dense(25088,activation= act, use_bias=True)(f)
    d2= Dropout(droprate)(d1)
    d3= Dense(2048,activation= act, use_bias=True)(d2)
    d4= Dropout(droprate)(d3)
    d5= Dense(1, activation='sigmoid')(d4)

    model = Model(inputs=visible, outputs=d5)
    model.compile(loss=keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(),tfa.metrics.F1Score(num_classes=1, average='macro',threshold=0.5)])
    return model

def create_shallow_model( img_rows, img_cols, filters,num_classes = 2,noise = 1,droprate=0.25):
    #size of parameters  
    activ ='ReLU'
    input_shape = ( img_rows, img_cols,filters)
    
    #print(input_shape)
    #Start Neural Network
    model = Sequential()
    model.add(Conv2D(1, kernel_size=(1, 1), activation=activ, padding="same",
              input_shape=input_shape))
    #convolution 1st layer
    model.add(Conv2D(16, kernel_size=(3, 3), 
                  activation=activ, padding="same"  )) 
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(droprate)) 

    #convolution 2nd layer
    model.add(Conv2D(16, kernel_size=(3, 3),  padding="same",
                  activation=activ)) 
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(droprate)) 
    #convolution 3rd layer
    model.add(Conv2D(16, kernel_size=(3, 3),  padding="same",
                  activation=activ)) 
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(droprate)) 
    

    

    #Fully connected 1st layer

    model.add(Flatten())

    model.add(Dense(25088,use_bias=True)) 
    model.add(Activation(activ)) 
    model.add(Dropout(droprate)) 


    model.add(Dense(2048, use_bias=True)) 
    model.add(Activation(activ))      
    model.add(Dropout(droprate))
    
    #Fully connected final layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(),tfa.metrics.F1Score(num_classes=1, average='macro',threshold=0.5)])
    return model


def call_model( use_existing_model,model,therapist, emotion, train_x, y_train, val_x, y_val, batch_size, model_path, name, epochs ):
    filters= train_x.shape[3]
    img_rows=train_x.shape[1]
    img_cols=train_x.shape[2] 
    #size of parameters 
    num_classes = 2  
    noise = 1
    droprate=0.25
    #modname ='saved_model/'+therapist+'_'+emotion+'_'+name #+'.h5'
    
    if (model==None):
        #print("Let us try")
        print("model does not existing. Creating a new one")
        if name=='shallow':
            model=create_shallow_model(img_rows,img_cols,filters)
        elif name=='3CNN2LSTM_Attn':
            model=create_3CNN2LSTM_Attn( img_rows, img_cols, filters)
        elif name=='conv1dlstmAttn':
            #print('conv1dlstmAttn')
            model=create_conv1dlstmAttn( img_rows, img_cols, filters)

        elif name=='LSTM_Attn':
            model=create_LSTM_Attn( img_rows, img_cols, filters)
        else:
            pass
            print("gdgdggd")            

       
    
    
    # prepare callbacks
    callbacks = [
        EarlyStopping(
            monitor= 'loss', #'val_f1_score', #'val_loss', #'val_categorical_accuracy', 
            patience=10,
            mode='min',
            verbose=0)
           #,ModelCheckpoint(model_path,
           # monitor='f1_score', #'val_loss', #'val_f1_score' , #'val_binary_accuracy', #val_categorical_accuracy', #val_recall', 
            #custom_objects=SeqSelfAttention.get_custom_objects(),
           # save_best_only=True, 
            #save_weights_only=False,
            #mode='max',
            #verbose=0) 
    ]

    history = model.fit(train_x, y_train, batch_size=batch_size, epochs = epochs, verbose=0, validation_data = (val_x, y_val),shuffle=True,callbacks=callbacks)
    #print(history.history['val_acc'])  

    get_memory() 
    

    return model, history


# In[7]:


if __name__ == '__main__':
    pass

'''
    ModelCheckpoint(model_path,
        monitor='val_categorical_accuracy', 
        save_best_only=True, 
        save_weights_only=False,
        mode='min',
        verbose=1) 
'''
'''
ModelCheckpoint(model_path,
    monitor='val_binary_accuracy', #val_categorical_accuracy', #val_recall', 
    #custom_objects=SeqSelfAttention.get_custom_objects(),
    save_best_only=True, 
    save_weights_only=True,
    mode='max',
    verbose=1) 
'''
'''
,
        ModelCheckpoint(model_path,
            monitor='val_binary_accuracy', #val_categorical_accuracy', #val_recall', 
            #custom_objects=SeqSelfAttention.get_custom_objects(),
            save_best_only=True, 
            save_weights_only=False,
            mode='max',
            verbose=1) 
            
,

'''
# In[ ]:




