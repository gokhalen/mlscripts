# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:00:42 2020

@author: Nachiket Gokhale

Program to classify images of stiffness into two categories 
        - has inclusion 
        - does not have inclusion

"""



import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


# TODO: Add more than 1 inclusion
#       Confusion matrix
#       Different values of stiffness

def plot_batch(data,fname):
    # data  - one batch (ndarray,labels)
    # fname - descriptor to plot
    batch_size = data[0].shape[0]
    nnodex     = data[0].shape[1]
    nnodey     = data[0].shape[2]
    label      = data[1]
    
    ycoord,xcoord = np.indices((nnodex,nnodey))

    for ii in range(batch_size):
        title = fname+str(ii) +'label='+str(label[ii])
        plt.figure(title)
        # breakpoint()
        plt.pcolormesh(xcoord,ycoord,data[0][ii,:,:,0])
        plt.title(title)
    

def make_stiffness(nnodex,nnodey,batch_size,nbatch):
    stfmin  = 1.0
    stfmax  = 5.0
    
    # arrays to choose centers from
    xcenlist = np.arange(3,nnodex-3)
    ycenlist = np.arange(3,nnodey-3)
    
    stiffness_field = []
    
    # generate training data - make inclusion stiffness variable later
    for ibatch in range(nbatch):
        # background stiffness is stfmin
        stiffness_data = stfmin*np.ones((batch_size,nnodex,nnodey,1),dtype='float64')  
        stiffness_label = np.zeros((batch_size,),dtype='int64')

        for isize in range(batch_size):
            
            kk = stfmin*np.ones((nnodex,nnodey))
            
            # probability of getting inclusion
            pp = random.uniform(0.0,1.0)
            # add an inclusion if pp >=0.3
            if ( pp >= 0.3):
                stiffness_label[isize] = 1
                
                ycoord,xcoord = np.indices((nnodex,nnodey))
                rad       = int(min(nnodex,nnodey)*random.uniform(0.05,0.1))
                # radlist.append(rad)
                if rad == 0:
                    sys.exit('Radius equals zero')

                # create center coordinates
                xcen  = random.choice(xcenlist)
                ycen  = random.choice(ycenlist)
                
                xcen  = xcen*np.ones((nnodex,nnodey))
                ycen  = ycen*np.ones((nnodex,nnodey))
                
                xdist = (xcen - xcoord)**2
                ydist = (ycen - ycoord)**2
                
                dist  = xdist + ydist
                rad2  = rad**2
                
                mask  = dist <= rad2
                
                # create masks
                kk[mask] = stfmax
                
                
            kk = kk.reshape(nnodex,nnodey,1)
            stiffness_data[isize,...] = kk 
        stiffness_field.append((stiffness_data,stiffness_label))

    return stiffness_field

def generate_data(nnodex,nnodey,batch_size,nbatch):
     '''
     Creates 3 lists containing training_data,validation_data,test_data
     Each list contains tuples
           - first element of the tuple is a numpy array of shape
             (batch_size,nodex,nodey,1)
           - second element of tuple is a numpy array of shape
             (batch_size,)
             
     Size of training_data   is nbatch
     Size of validation_data is np.ceil(nbatch/3)
     Size of test_data       is np.ceil(nbatch/3)
             
     '''
     # don't let foll values go to zero. add 1
     nbatch_val  = int(nbatch/3) + 1
     nbatch_test = int(nbatch/3) + 1
     
     training_data   = make_stiffness(nnodex,nnodey,batch_size,nbatch)
     validation_data = make_stiffness(nnodex,nnodey,batch_size,nbatch_val)
     test_data       = make_stiffness(nnodex,nnodey,batch_size,nbatch_test)
  
     return (training_data,validation_data,test_data)
 

nnodex,nnodey,batch_size,nbatch=64,64,32,32

train_data,valid_data,test_data = generate_data(nnodex=nnodex,nnodey=nnodey,
                                                batch_size=batch_size,nbatch=nbatch)
# Initialising the CNN
cnn = tf.keras.models.Sequential()


# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[nnodex, nnodey, 1]))
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# plot
tf.keras.utils.plot_model(
        cnn, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history=cnn.fit(x = train_data, validation_data = test_data, epochs = 25)
