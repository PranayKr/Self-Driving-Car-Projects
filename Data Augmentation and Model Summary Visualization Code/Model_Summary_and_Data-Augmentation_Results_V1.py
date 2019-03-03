#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential, Model 
from keras.layers import Lambda, Cropping2D, Convolution2D, ELU, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from random import shuffle
import os
import cv2
import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


INPUT_SHAPE = (160, 320, 3)
LEARNING_PARAMETER = .0001 #1e-4


# In[4]:


model = Sequential()
model.add(Lambda(lambda x: x/255.-0.5,input_shape=INPUT_SHAPE))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation="elu"))
model.add(Dense(50, activation="elu"))
model.add(Dense(10, activation="elu"))
model.add(Dense(1))

adam = Adam(lr=LEARNING_PARAMETER)
model.compile(optimizer=adam,loss='mse')


# In[5]:


model.summary()


# In[6]:


PATH_TO_IMG = 'C:/Users/Dell/Desktop/Behavior-Cloning/sim_data/sim_data/data/IMG/'
PATH_TO_CSV = 'C:/Users/Dell/Desktop/Behavior-Cloning/sim_data/sim_data/data/driving_log.csv'
CORRECTION = 0.25


# In[7]:


def get_csv():
    df = pd.read_csv(PATH_TO_CSV, index_col=False)
    df.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']
    df = df.sample(n=len(df))

    return df


# In[8]:


# Randomly selecting the let, right, and center images
def random_select_image(data, i , default=0):
     
    random = np.random.randint(3)
    
    path = " "
    
    if random == 0:
        path = PATH_TO_IMG+data['left'][i].split('/')[-1]
        
        difference = CORRECTION
    elif random == 1:
        path = PATH_TO_IMG+data['center'][i].split('/')[-1]
        
        difference = 0 
    elif random == 2:
        path = PATH_TO_IMG+data['right'][i].split('/')[-1]
       
        difference = -CORRECTION
        
    image = cv2.imread(path)
    
    #image = cv2.resize(image, (160, 320), cv2.INTER_AREA)
    
    image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    angle = float(data['steer'][i])+difference
    
    f, (ax1,ax2) = plt.subplots(1, 2, figsize=(11,11))
    
    if(difference==0):
        ax1.set_title('Center Image in BGR Color Space')
    elif(difference>0):
        ax1.set_title('Left Image in BGR Color Space')
    elif(difference<0):
        ax1.set_title('Right Image in BGR Color Space')
    ax1.imshow(image)
    ax1.set_xlabel("Steering Angle:"+ str(float(data['steer'][i])))
    if(difference==0):
        ax2.set_title('Center Image in RGB Color Space')
    elif(difference>0):
        ax2.set_title('Left Image in RGB Color Space')
    elif(difference<0):
        ax2.set_title('Right Image in RGB Color Space')
    ax2.imshow(image_converted)
    if(difference==0):
        ax2.set_xlabel("Steering Angle (No Correction):"+ str(angle))
    elif(difference>0 or difference<0):
        ax2.set_xlabel("Original Steering Angle:"+str(float(data['steer'][i]))+"\n Steering Angle with Correction:"+ str(angle))
    
    
  
    return image_converted, angle , difference


# In[9]:


def flip_img_angle(image, angle , difference):
    image_flipped = cv2.flip(image, 1)
    flppedangle = -1.0 * angle
    
    f, (ax1,ax2) = plt.subplots(1, 2, figsize=(11,11))
    
    if(difference==0):
        ax1.set_title('Unflipped Center Image in RGB Color Space')
    elif(difference>0):
        ax1.set_title('Unflipped Left Image in RGB Color Space')
    elif(difference<0):
        ax1.set_title('Unflipped Right Image in RGB Color Space')
    ax1.imshow(image)
    ax1.set_xlabel("Steering Angle:"+ str(angle))
    if(difference==0):
        ax2.set_title('Flipped Center Image in RGB Color Space')
    elif(difference>0):
        ax2.set_title('Flipped Left Image in RGB Color Space')
    elif(difference<0):
        ax2.set_title('Flipped Right Image in RGB Color Space')
    ax2.imshow(image_flipped)
    ax2.set_xlabel("Flipped Steering Angle:"+ str(flppedangle))
    

    return image_flipped, angle


# In[10]:


def brightnessed_img(origimage,difference):
    image = cv2.cvtColor(origimage, cv2.COLOR_RGB2HSV)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    random_bright = .25 + np.random.uniform()
    image[:,:,2] = image[:,:,2] * random_bright
    image_brightalter = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    #image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    f, (ax1,ax2) = plt.subplots(1, 2, figsize=(11,11))
    
    if(difference==0):
        ax1.set_title('Original Center Image in RGB Color Space')
    elif(difference>0):
        ax1.set_title('Original Left Image in RGB Color Space')
    elif(difference<0):
        ax1.set_title('Original Right Image in RGB Color Space')
    ax1.imshow(origimage)
    #ax1.set_xlabel("Steering Angle:"+ str(angle))
    if(difference==0):
        ax2.set_title('Brightness-Altered Center Image in RGB Color Space')
    elif(difference>0):
        ax2.set_title('Brightness-Altered Left Image in RGB Color Space')
    elif(difference<0):
        ax2.set_title('Brightness-Altered Right Image in RGB Color Space')
    ax2.imshow(image_brightalter)
    
    #ax2.set_xlabel("Flipped Steering Angle:"+ str(flppedangle))"""
    
    return image_brightalter


# In[11]:


def trans_image(image, steer , difference):
    trans_range = 100
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 0
    M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, M, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    
    """f, (ax1,ax2) = plt.subplots(1, 2, figsize=(11,11))
    
    if(difference==0):
        ax1.set_title('Original Center Image in RGB Color Space')
    elif(difference>0):
        ax1.set_title('Original Left Image in RGB Color Space')
    elif(difference<0):
        ax1.set_title('Original Right Image in RGB Color Space')
    ax1.imshow(image)
    ax1.set_xlabel("Steering Angle:"+ str(steer))
    if(difference==0):
        ax2.set_title('Translated Center Image in RGB Color Space')
    elif(difference>0):
        ax2.set_title('Translated Left Image in RGB Color Space')
    elif(difference<0):
        ax2.set_title('Translated Right Image in RGB Color Space')
    ax2.imshow(image_tr)
    ax2.set_xlabel("Steering Angle after translation:"+ str(steer_ang))"""
    
    return image_tr, steer_ang


# In[12]:


# Getting fetatures and lables from training and validation data
def get_data(data):
    images = []
    angles = []
    for i in data.index:
        image, angle , difference = random_select_image(data, i , 0)

        # Data augumentation
        if np.random.uniform() < 0.5:
            image, angle = flip_img_angle(image, angle , difference)
        image = brightnessed_img(image,difference)
        image, angle = trans_image(image, angle,difference)
        images.append(image)
        angles.append(angle)

    # Creating as numpy array
    X = np.array(images)
    y = np.array(angles)

    return X, y


# In[ ]:


samples = get_csv()

# Training and Validation data
training_count = int(0.8 * len(samples))
training_data = samples[:training_count].reset_index()
validation_data = samples[training_count:].reset_index()

# Getting features and labels for training and validation.
X_train, y_train = get_data(training_data)
X_valid, y_valid = get_data(validation_data)


# In[ ]:




