# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:03:08 2021

@author: aa
"""

import numpy as np

correct   = np.load('correct_strainyy.npy')
correct   = correct[...,1]
predicted = np.load('prediction_imagesy.npy')

nexamples = predicted.shape[0]
back_norm = np.zeros((nexamples,),dtype='float64')
inc_norm  = np.zeros((nexamples,),dtype='float64')

for iexample in range(nexamples):
    # get indices
    back_idx = np.where(correct[iexample,:,:]==1.0)
    inc_idx  = np.where(correct[iexample,:,:]!=1.0)
    
    mu_back_correct = correct[iexample,:,:][back_idx]
    mu_inc_correct  = correct[iexample,:,:][inc_idx]
    
    mu_back_pred    = predicted[iexample,:,:][back_idx]
    mu_inc_pred     = predicted[iexample,:,:][inc_idx]
    
    # compute norm for each example
    back_norm[iexample] = np.linalg.norm(mu_back_pred)/np.linalg.norm(mu_back_correct)
    inc_norm[iexample]  = np.linalg.norm(mu_inc_pred)/np.linalg.norm(mu_inc_correct)
    

back_ave = np.sum(back_norm)/nexamples
inc_ave  = np.sum(inc_norm)/nexamples

print(f'{back_ave=}')
print(f'{inc_ave=}')