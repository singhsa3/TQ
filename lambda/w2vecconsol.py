#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install torchaudio==0.10.1 attention pyyaml h5py nvidia_smi keras==2.9.0rc1 tensorflow librosa matplotlib pickle5 Pillow pandas numpy opencv-python numba #gdown')




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
import models_custom as mcs
from keras.models import load_model
import glob

import os

import pandas as pd


# In[1]:



#therapist='Yared Alemu'
#emotion='fear'
#name='shallow'
#use_existing_model= 'no'

# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--therapist', type=str, required=True)
parser.add_argument('--emotion', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--use_existing_model', type=str, required=False, default='no')
# Parse the argument
args = parser.parse_args()

therapist= args.therapist
emotion=args.emotion
name=args.name
use_existing_model= args.use_existing_model

pathG='../data' 
DATASET_PATH0 = pathG+"/w2v2/w2v2L0"
DATASET_PATH4 = pathG+"/w2v2/w2v2L4"
DATASET_PATH8 = pathG+"/w2v2/w2v2L8"
DATASET_PATHS = [DATASET_PATH0,DATASET_PATH4,DATASET_PATH8]
l95=5000
chunk_size = 128

modname ='saved_model/'+therapist+'_'+emotion+'_'+name
if use_existing_model=='no':
    model=None
else:
    model = load_model(modname)

epochs=150
#https://stackoverflow.com/questions/10443295/combine-3-separate-numpy-arrays-to-an-rgb-image-in-python


# In[6]:


# Get data list

df=pd.read_csv(pathG+"/labels/"+therapist+"_"+emotion+".csv")
df = df.reset_index()
df['name2'] =df['name'].apply (lambda x: x.split(".")[0]+".pickle")
 



# This is to delete the rows greater than max size
DATASET_PATH = DATASET_PATHS[0]
df2=df.copy(deep=True)
for i,row in df.iterrows():    
    fl = DATASET_PATH+"/"+row['name2']
    try:
        with open(fl,"rb") as f:
            x=pickle.load(f)     
        l1 = x[0].shape[0]
        w1 = x[0].shape[1] 
        if l1> l95:
            try:
            #print(i)
                df2=df2.drop(df.iloc[i].name)
            except:
                print(i,l1)
                print("encountered and error")
    except:
        pass

df= df2.reset_index()
x=None


# In[9]:


# Normalizing data and creating an array
# Normalizing data and creating an array
def create_arr(fl,l95, height,width,img):
    if img==1:
        pxl=255
    else:
        pxl=1  
    with open(fl,"rb") as f:
        x=pickle.load(f) 
    l1 = x[0].shape[0]
    try:
        #print(l95-l1+1)
        x=np.pad(x.cpu().detach().numpy(), ((0,0), (10,l95-l1+1), (0, 0)), 'constant')
        c= x[0]
        #print(c.shape)
        if img==1:      
            b = np.max(c)
            a = np.min(c)
            c =pxl*(c-a)/(b-a)
            c=c.astype(np.uint8)
            data=im.fromarray(c)
            data = data.resize((height,width) )
            arr=np.array(data)
            arr = arr.reshape(height,width,1)
            x=None
        else:        
            arr=x[0]  
            arr = arr.reshape(arr.shape[0],arr.shape[1],1)
            #print(arr.shape)
    except Exception as e: # work on python 2.x
        #print("oops")
        print(str(e))
        arr= None     
    return arr


# In[11]:


from os.path import exists
# Converting to image
def imgconvert(df, l95,DATASET_PATH,height=500, width=64, img=1):
    df=df.drop(columns=['level_0'])
    df=df.reset_index()
    arrlist0=[]
    for DATASET_PATH in DATASET_PATHS:
        fl= DATASET_PATH +"/"+df.iloc[0]['name2']           
        if exists(fl)==True:
            arrlist0.append(create_arr(fl,l95, height,width,img))
        else:
            print(fl+" Does not exists")
    arr= np.concatenate(tuple(arrlist0), axis=2)
    #arr = arr.reshape(1, height,width,len(DATASET_PATHS))
    arr= np.expand_dims(arr, axis=0)
    #print(arr.shape)
    #fl= DATASET_PATH +"/"+df.iloc[0]['name2']  
    #arr=create_arr(fl,l95, height,width,img)
    
    idx=[]
    for i,row in df.iterrows():
        #fl = DATASET_PATH+"/"+row['name2']
        arrlist2=[]
        
        for DATASET_PATH in DATASET_PATHS:            
            fl = DATASET_PATH+"/"+row['name2'] 
            if exists(fl)==True:
                arrlist2.append(create_arr(fl,l95, height,width,img)) 
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


def split_dataframe(df, chunk_size): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


# In[13]:


from sklearn.model_selection import train_test_split
X= df
y= df['emotion']
X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test1, y_test1, test_size=0.5, random_state=42)


# In[14]:


val_x, val_y =imgconvert(X_val,l95,DATASET_PATHS,img=1)
test_x, test_y =imgconvert(X_val,l95,DATASET_PATHS,img=1)
val_y=val_y.astype(int) 
test_y=test_y.astype(int) 
y_val = to_categorical(val_y)
y_test= to_categorical(test_y)


# In[16]:



chunks=split_dataframe(X_train,chunk_size)





for chunk in chunks[0:len(chunks)]:
    chunk=chunk.drop(columns=['level_0'])
    chunk=chunk.reset_index()
    try:
        train_x, train_y =imgconvert(chunk,l95,DATASET_PATHS,img=1)       
        # Train and Test
        train_y=train_y.astype(int)       
        # one hot encode outputs
        y_train = to_categorical(train_y)        
        print(train_x.shape, y_train.shape)
        if name=='shallow':
            model= mcs.call_model( model,therapist.replace(" ",""), emotion, train_x, y_train, val_x, y_val, train_x.shape[0],  modname, name, epochs  )      
      
        
    except Exception as e:
        print(str(e))


# In[17]:


#from keras.models import load_model
#modname ='saved_model/'+therapist+'_'+emotion+'_'+name #+'.h5'
#model = load_model(modname)
score = model.evaluate(test_x, y_test, verbose=1)

#print loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# using time module
import time
  
# ts stores the time in seconds
ts = time.time()

f = open("results.txt", "a")
f.write(str(ts)+'|'+therapist+'|'+emotion+'|'+name+'|'+str(score[1])+'\n')
f.close()

from numba import cuda 
device = cuda.get_current_device()
device.reset()
        
    
    


# In[ ]:




