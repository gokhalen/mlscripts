# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:56:35 2020

@author: aa
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_batch(data,fname,pred_label=None):
    # plots one batch of training data and labels
    # data is a tuple containing (training_data,labels)
    # training_data has dims (batch_size,nnodex,nnodey,1)
    # labels has shape (batch_size,)
    # pred_label has shape (batch_size,)
    # data  - one batch (ndarray,labels)
    # fname - description of data to plot. 
    #         fname is used to create the title
    
    batch_size = data[0].shape[0]
    nnodex     = data[0].shape[1]
    nnodey     = data[0].shape[2]
    label      = data[1]
    
    ycoord,xcoord = np.indices((nnodex,nnodey))

    for ii in range(batch_size):
        title = fname+str(ii) +' label='+str(label[ii])
        if pred_label is not None:
            title = title + ' pred='+str(pred_label[ii])
        plt.figure(title)
        # breakpoint()
        plt.pcolormesh(xcoord,ycoord,data[0][ii,:,:,0])
        plt.title(title)


def make_stiffness(nnodex,nnodey,nsamples,homogeneous=True):
    # nnodex   = nodes/pixels in the x-direction
    # nnodey   = nodes/pixels in the y-direction
    # nsamples = number of examples to generate
    # homogeneous -> if true,  then homogeneous examples are generated
    #             -> if false, then homogeneous examples are not generated                

    # makes stiffness data
    # returns a tuple (stiffness_data,stiffness_label,stiffness_center,stiffness_value)
    # stiffness_data is a ndarray of shape (nsamples,nnodex,nnodey,1)
    #                it contains nsample stiffness images    
    #                the last 1 is preparation for handling displacement fields
    #                with more than 1 component
    # stiffness_label is a ndarray of shape (nsamples,)
    #                 it contains truth value of the stiffness image
    #                 0 if it is homogeneous
    #                 1 if it has an inclusion     
    # stiffness_center a 2d ndarray (xcen,ycen) center of the inclusion 
    #                  (-1,-1) if there is no inclusion
    # stiffness_value  a scalar containing the value of the stiffness 
    #                  between stfmin and stfmax
    #                  -1 if there is no inclusion                  
    
    stfback = 1.0
    stfmin  = 2.0
    stfmax  = 5.0
    
    # arrays to choose centers from
    # we leave some space on the sides
    xcenlist = np.arange(3,nnodex-3)
    ycenlist = np.arange(3,nnodey-3)
    
    # generate training data - make inclusion stiffness a variable later
    # background stiffness is stfback
    stiffness_data   = stfback*np.ones((nsamples,nnodex,nnodey,1),dtype='float64')  
    stiffness_label  = np.zeros((nsamples,),dtype='int64')
    stiffness_center = -1*np.ones((nsamples,2),dtype='int64')
    stiffness_value  = -1*np.ones((nsamples,))

    for isample in range(nsamples):
        
        # set background stiffness
        kk = stfback*np.ones((nnodex,nnodey))
        
        # probability of getting inclusion
        pp = np.random.uniform(0.0,1.0)
        # add an inclusion if pp >=0.3 of if homogeneous = False
        if (( pp >= 0.3) or (homogeneous == False)):
            stiffness_label[isample] = 1
            
            # note the inversion of x and y
            ycoord,xcoord = np.indices((nnodex,nnodey))
            rad       = int(min(nnodex,nnodey)*np.random.uniform(0.05,0.1))
            # radlist.append(rad)
            if rad == 0:
                sys.exit('Radius equals zero')

            # create center coordinates
            xcen1 = np.random.choice(xcenlist)
            ycen1 = np.random.choice(ycenlist)
            
            xcen  = xcen1*np.ones((nnodex,nnodey))
            ycen  = ycen1*np.ones((nnodex,nnodey))
            
            xdist = (xcen - xcoord)**2
            ydist = (ycen - ycoord)**2
            
            dist  = xdist + ydist
            rad2  = rad**2
            
            # create masks
            mask  = dist <= rad2
            # set stiffness,center and value
            stfval = np.random.uniform(stfmin,stfmax)
            kk[mask] = stfval
            stiffness_center[isample] = xcen1,ycen1
            stiffness_value[isample]  = stfval
             
            
            
        kk = kk.reshape(nnodex,nnodey,1)
        stiffness_data[isample,...] = kk 

    return (stiffness_data,stiffness_label,stiffness_center,stiffness_value)


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
 
def plotall(history):
    # plot everything in history
    # history is returned by model.fit
    for ikey in history.history.keys():
        plt.figure(ikey)
        data   = history.history[ikey]
        epochs = range(1,len(data)+1)
        plt.plot(epochs,data)
        plt.title(ikey)
        plt.xlabel('epochs')
        plt.ylabel(ikey)
