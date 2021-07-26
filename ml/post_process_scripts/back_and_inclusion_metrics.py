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
twotothree_norm  = np.zeros((nexamples,),dtype='float64')
nexamples2t3 = 0

for iexample in range(nexamples):
    # get indices
    back_idx = np.where(correct[iexample,:,:]==1.0)
    inc_idx  = np.where(correct[iexample,:,:]!=1.0)

    # mask for getting numbers between 2.0 and 3.0  
    mask1 = correct[iexample,:,:]>=2.0
    mask2 = correct[iexample,:,:]<=3.0
    mask2t3 = np.logical_and(mask1,mask2)
    
    nback   = back_idx[0].shape[0]
    ninc    = inc_idx[0].shape[0]
    
    assert (nback+ninc==6305),'inconsistent lengths'
    
    mu_back_correct = correct[iexample,:,:][back_idx]
    mu_inc_correct  = correct[iexample,:,:][inc_idx]
    mu_2t3_correct  = correct[iexample,:,:][mask2t3]
    
    n2t3    = mu_2t3_correct.shape[0]

    
    mu_back_pred    = predicted[iexample,:,:][back_idx]
    mu_inc_pred     = predicted[iexample,:,:][inc_idx]
    mu_2t3_pred     = predicted[iexample,:,:][mask2t3]
    
    # compute norm for each example
    #back_norm[iexample] = np.linalg.norm(mu_back_pred-mu_back_correct)/np.linalg.norm(mu_back_correct)
    #inc_norm[iexample]  = np.linalg.norm(mu_inc_pred-mu_inc_correct)/np.linalg.norm(mu_inc_correct)
    
    back_norm[iexample]  = np.sum((mu_back_pred-mu_back_correct)/mu_back_correct)/nback
    inc_norm[iexample]   = np.sum((mu_inc_pred-mu_inc_correct)/mu_inc_correct)/ninc
    if n2t3 !=0:
        nexamples2t3 +=1
        twotothree_norm[iexample] = np.sum((mu_2t3_pred-mu_2t3_correct)/mu_2t3_correct)/n2t3

back_ave = np.sum(back_norm)/nexamples
inc_ave  = np.sum(inc_norm)/nexamples
twotothree_ave = np.sum(twotothree_norm)/nexamples2t3
print(f'{back_ave=}')
print(f'{inc_ave=}')
print(f'{twotothree_ave=},{nexamples2t3=}')