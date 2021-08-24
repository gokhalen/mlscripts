# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:54:56 2021

@author: User
"""

# combines inserts npy data of cross into the test data of CNN1

import numpy as np

file_prefixes = ['binary',
               'center',
               'field',
               'images',
               'radius',
               'strain',
               'value'
               ]

ntrain = 6000
nvalid = 2000
nsmallsize = 14
startcross = ntrain+nvalid
endcross   = startcross+nsmallsize

for fprefix in file_prefixes:
    
    fsmall=np.load(fprefix+'_small.npy')
    flarge=np.load(fprefix+'_large.npy')
    flarge[startcross:endcross] = fsmall
    
    np.save(fprefix+'.npy',flarge)
    
