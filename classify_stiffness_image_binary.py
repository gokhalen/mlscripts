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


def make_circle():
    pass

def generate_data(nnodex,nnodey,batch_size,nbatch):
     '''
     Creates 3 lists containing training_data,validation_data,test_data
     Each list contains tuples
           - first element of the tuple is a numpy array of shape
             (batch_size,nodex,nodey,1)
           - second element of tuple is a numpy array of shape
             (batch_size,)
             
     Size of training_data   is nbatch
     Size of validation_data is nbatch/3
     Size of test_data       is nbatch/3
             
     '''
     
     training_data = []; validation_data = []; test_data = []
     radlist = []
     
     
     # arrays to choose centers from
     xcenlist = np.arange(3,nnodex-3)
     ycenlist = np.arange(3,nnodey-3)
     
     # generate training data - make inclusion stiffness variable later
     for ibatch in range(nbatch):
         # background stiffness is 1
         data1 = np.ones((batch_size,nnodex,nnodey,1))  
         for isize in range(batch_size):
                # probability of getting inclusion
                pp = random.uniform(0.0,1.0)
                
                # add an inclusion if pp >=0.3
                if ( pp >= 0.3):
                    kk        = np.ones((nnodex,nnodey))
                    rows,cols = np.indices((nnodex,nnodey))
                    rad       = int(min(nnodex,nnodey)*random.uniform(0.05,0.1))
                    radlist.append(rad)
                    if rad == 0:
                        sys.exit('Radius equals zero')

                    # create center coordinates
                    xcen      = random.choice(xcenlist)
                    ycen      = random.choice(ycenlist)
                    
                    xcen      = xcen*np.ones((nnodex,nnodey))
                    ycen      = ycen*np.ones((nnodex,nnodey))
                    
                    # create masks

                        
                    
                    
            
         training_data.append(data1)
         validation_data.append(data1)
         test_data.append(data1)
     
  
     return (training_data,validation_data,test_data,radlist)
 

training_data,validation_data,test_data,radlist = generate_data(nnodex=64,nnodey=64,batch_size=32,nbatch=100)