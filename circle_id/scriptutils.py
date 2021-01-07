# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:56:35 2020

@author: aa
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def feature_scaling_forward(data,sc=None):
    # data - tuple generated from (make_stiffness)
    # sc   - tuple of scalers to be used for each field in data
    #      - if sc is none then a new scaler is computed
    # returns - scaled data and the scalers used to scale
    
    # data[1] are labels which are 0 or 1. They are not scaled.
    
    # if the x and y coordinates of the image are drastically different
    # then they need to be feature scaled separately
    # in other words, if the picture is badly scaled. 
    
    
    oldshape = data[0].shape

    if ( sc == None ):
       scimg    = StandardScaler()
       sclabel  = None
       sccenter = StandardScaler()
       scvalue  = StandardScaler()
       scradius = StandardScaler()
       
       
       # fitting requires a 2D array
       scimg.fit(data[0].reshape(-1,1))
       sccenter.fit(data[2].reshape(-1,1))
       scvalue.fit(data[3].reshape(-1,1))
       scradius.fit(data[4].reshape(-1,1))
       
    else:
       scimg    = sc[0]
       sclabel  = sc[1]
       sccenter = sc[2]
       scvalue  = sc[3]
       scradius = sc[4]
       
    newdata0 = scimg.transform(data[0].reshape(-1,1)).reshape(oldshape)
    # data 2 is a 2D array
    newdata2 = sccenter.transform(data[2].reshape(-1,1)).reshape((-1,2))
    newdata3 = scvalue.transform(data[3].reshape(-1,1)).reshape((-1,))
    newdata4 = scradius.transform(data[4].reshape(-1,1)).reshape((-1,))
    
    newdata = (newdata0,data[1],newdata2,newdata3,newdata4)
    
    return (newdata,(scimg,sclabel,sccenter,scvalue,scradius))

def feature_scaling_inverse(data,sc):
    # inverse transforms features according to sc
    oldshape = data[0].shape
    newdata0 = sc[0].inverse_transform(data[0].reshape(-1,1)).reshape(oldshape)
    # data 2 is a 2D array
    newdata2 = sc[2].inverse_transform(data[2].reshape(-1,1)).reshape((-1,2))
    newdata3 = sc[3].inverse_transform(data[3].reshape(-1,1)).reshape((-1,))
    newdata4 = sc[4].inverse_transform(data[4].reshape(-1,1)).reshape((-1,))
    
    newdata = (newdata0,data[1],newdata2,newdata3,newdata4)
    return newdata
    

def plot_batch(data,fname,pred_label=None,plot_type='all'):
    # plots one batch of training data and labels
    # data is a tuple containing (training_data,labels)
    # training_data has dims (batch_size,nnodex,nnodey,1)
    # labels has shape (batch_size,)
    # pred_label has shape (batch_size,)
    # data  - one batch (ndarray,labels)
    # fname - description of data to plot. 
    #         fname is used to create the title
    # pred_label - predicted label by NN
    # plot_type  - 'correct','incorrect','all'
    #            - 'correct' plots all examples
    #            - 'incorrect' plots only incorrectly labelled examples
    #            - 'all' plots all examples
    
    # check if plot_type is in allowed types
    assert (plot_type in ['all','correct','incorrect']),'plot_type should \
    be one of "all" "correct" or "incorrect" '
         
    
    batch_size = data[0].shape[0]
    nnodex     = data[0].shape[1]
    nnodey     = data[0].shape[2]
    label      = data[1]
    
    # note the inversion of ycoord and xcoord
    ycoord,xcoord = np.indices((nnodex,nnodey))
    
    # create index lists of only the examples to plot
    if ( plot_type == 'all'):
        # get all indices using self comparison
        idxlist = np.arange(batch_size)
        
    if ( plot_type == 'correct'):
        idxlist = np.where(pred_label == label)[0]
        
    if ( plot_type == 'incorrect'):
        idxlist = np.where(pred_label != label)[0]
        
       
    for ii in idxlist:
        title = fname+str(ii) +' label='+str(label[ii])
        if pred_label is not None:
            title = title + ' pred='+str(pred_label[ii])
        plt.figure(title)
        # breakpoint()
        plt.pcolormesh(xcoord,ycoord,data[0][ii,:,:,0],shading='auto')
        plt.title(title)


def make_stiffness(nnodex,nnodey,nsamples,create_homo=True):
    # nnodex   = nodes/pixels in the x-direction
    # nnodey   = nodes/pixels in the y-direction
    # nsamples = number of examples to generate
    # create_homo -> if true,  then homogeneous examples are generated
    #             -> if false, then homogeneous examples are not generated                

    # makes stiffness data
    # returns a tuple (stiffness_data,stiffness_label,stiffness_center,
    #                  stiffness_value,stiffness_radius)
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
    # stiffness_radius radius of the inclusion                
    
    stfback = 1.0
    stfmin  = 2.0
    stfmax  = 5.0
    
    # arrays to choose centers from
    # we leave some space on the sides
    xcenlist = np.arange(3,nnodex-3)
    ycenlist = np.arange(3,nnodey-3)
    
    # generate training data - make inclusion stiffness a variable later
    # background stiffness is stfback
    stiffness_data   = stfback*np.ones((nsamples,nnodey,nnodex,1),dtype='float64')  
    stiffness_label  = np.zeros((nsamples,),dtype='int64')
    stiffness_center = -1*np.ones((nsamples,2),dtype='int64')
    stiffness_value  = -1*np.ones((nsamples,))
    stiffness_radius = -1*np.ones((nsamples,))

    for isample in range(nsamples):
        
        # set background stiffness
        kk = stfback*np.ones((nnodey,nnodex))
        
        # probability of getting inclusion
        pp = np.random.uniform(0.0,1.0)
        # add an inclusion if pp >=0.3 of if homogeneous = False
        if (( pp >= 0.3) or ( create_homo == False)):
            stiffness_label[isample] = 1
            
            ycoord,xcoord = np.indices((nnodey,nnodex))
            rad       = int(min(nnodex,nnodey)*np.random.uniform(0.05,0.10))
            
            # radlist.append(rad)
            if rad == 0:
                sys.exit('Radius equals zero')

            # create center coordinates
            xcen1 = np.random.choice(xcenlist)
            ycen1 = np.random.choice(ycenlist)
            
            xcen  = xcen1*np.ones((nnodey,nnodex))
            ycen  = ycen1*np.ones((nnodey,nnodex))
            
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
            stiffness_radius[isample] = rad
            
            
        kk = kk.reshape(nnodey,nnodex,1)
        stiffness_data[isample,...] = kk 

    return (stiffness_data,stiffness_label,stiffness_center,stiffness_value,
            stiffness_radius)


def generate_data(nnodex,nnodey,ntrain,nval=None,ntest=None,create_homo=True):
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
     if ( nval == None):
         nval  = int(ntrain/3) + 1
     if ( ntest == None):
         ntest = int(ntrain/3) + 1
     
     training_data   = make_stiffness(nnodex,nnodey,ntrain,create_homo)
     validation_data = make_stiffness(nnodex,nnodey,nval,create_homo)
     test_data       = make_stiffness(nnodex,nnodey,ntest,create_homo)
  
     return (training_data,validation_data,test_data)
 
def plotall(history,outputdir):
    # plot everything in history
    # history is returned by model.fit
    for ikey in history.history.keys():
        plt.figure(ikey)
        data   = history.history[ikey]
        epochs = range(1,len(data)+1)
        yscale = 'linear'
        if 'loss' in ikey:
            yscale = 'log'
        plt.plot(epochs,data)
        plt.yscale(yscale)
        plt.title(ikey)
        plt.xlabel('epochs')
        plt.ylabel(ikey)
        plt.grid(True,which='both')
        plt.savefig(outputdir+'plot'+ikey+'.png')
