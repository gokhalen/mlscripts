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
    

def make_stiffness(nnodex,nnodey,nsamples):
    
    # returns a tuple (stiffness_data,stiffness_label)
    # stiffness_data is a ndarray of shape (nsamples,nnodex,nnodey,1)
    #                it contains nsample stiffness images    
    #                the last 1 is preparation for handling displacement fields
    #                with more than 1 component
    # stiffness_label is a ndarray of shape (nsamples,)
    #                 it contains truth value of the stiffness image
    #                 0 if it is homogeneous
    #                 1 if it has an inclusion     
    
    stfmin  = 1.0
    stfmax  = 5.0
    
    # arrays to choose centers from
    # we leave some space on the sides
    xcenlist = np.arange(3,nnodex-3)
    ycenlist = np.arange(3,nnodey-3)
    
    # generate training data - make inclusion stiffness a variable later
    # background stiffness is stfmin
    stiffness_data = stfmin*np.ones((nsamples,nnodex,nnodey,1),dtype='float64')  
    stiffness_label = np.zeros((nsamples,),dtype='int64')

    for isample in range(nsamples):
        
        kk = stfmin*np.ones((nnodex,nnodey))
        
        # probability of getting inclusion
        pp = random.uniform(0.0,1.0)
        # add an inclusion if pp >=0.3
        if ( pp >= 0.3):
            stiffness_label[isample] = 1
            
            # note the inversion of x and y
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
        stiffness_data[isample,...] = kk 

    return (stiffness_data,stiffness_label)

def generate_data(nnodex,nnodey,ntrain):
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
     nval  = int(ntrain/3) + 1
     ntest = int(ntrain/3) + 1
     
     training_data   = make_stiffness(nnodex,nnodey,ntrain)
     validation_data = make_stiffness(nnodex,nnodey,nval)
     test_data       = make_stiffness(nnodex,nnodey,ntest)
  
     return (training_data,validation_data,test_data)
 

nnodex,nnodey,ntrain=64,64,32
nepochs = 32

train_data,valid_data,test_data = generate_data(nnodex=nnodex,nnodey=nnodey,
                                                ntrain=ntrain)


# Initialising the CNN
cnn = tf.keras.models.Sequential()
# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                               input_shape=[nnodex, nnodey, 1]))
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

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Summary
cnn.summary()

# Training the CNN on the Training set and evaluating it on the Test set
# nhg - how does cnn.fit know that data has finished?
#     - if I do 'for i in training_set' it keeps yielding forever
history=cnn.fit(x = train_data[0], y = train_data[1],
                validation_data = valid_data, epochs = nepochs)


# plot 
for ikey in history.history.keys():
    plt.figure(ikey)
    data   = history.history[ikey]
    epochs = range(1,len(data)+1)
    plt.plot(epochs,data)
    plt.title(ikey)
    plt.xlabel('epochs')
    plt.ylabel(ikey)

'''
plt.figure('Training Accuracy')
train_accuracy = history.history['accuracy']
epochs         = range(1, len(train_accuracy)+1)
plt.plot(epochs, train_accuracy)
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
'''