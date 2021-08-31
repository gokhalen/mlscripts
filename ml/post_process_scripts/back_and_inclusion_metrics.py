# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:03:08 2021

@author: aa
"""

import numpy as np

cnnidx = 1

correct   = np.load('correct.npy')
correct   = correct[...,1]
predicted = np.load(f'predictioncnn{cnnidx}.npy')

nexamples = predicted.shape[0]
back_norm = np.zeros((nexamples,),dtype='float64')
inc_norm  = np.zeros((nexamples,),dtype='float64')

twotothree_norm  = np.zeros((nexamples,),dtype='float64')
threetofour_norm = np.zeros((nexamples,),dtype='float64')
fourtofive_norm  = np.zeros((nexamples,),dtype='float64')


nexamples2t3 = 0
nexamples3t4 = 0
nexamples4t5 = 0

for iexample in range(nexamples):
    # get indices
    back_idx = np.where(correct[iexample,:,:]==1.0)
    inc_idx  = np.where(correct[iexample,:,:]!=1.0)

    # mask for getting numbers between 2.0 and 3.0  
    mask1_2t3 = correct[iexample,:,:] >= 2.0
    mask2_2t3 = correct[iexample,:,:] <  3.0
    mask2t3   = np.logical_and(mask1_2t3,mask2_2t3)
    
    # mask for getting numbers between 3.0 and 4.0
    mask1_3t4 = correct[iexample,:,:] >= 3.0
    mask2_3t4 = correct[iexample,:,:] <  4.0
    mask3t4   = np.logical_and(mask1_3t4,mask2_3t4)
    
    # mask for getting numbers between 4.0 and 5.0
    mask1_4t5 = correct[iexample,:,:] >= 4.0
    mask2_4t5 = correct[iexample,:,:] <  5.0
    mask4t5   = np.logical_and(mask1_4t5,mask2_4t5)
    
    nback   = back_idx[0].shape[0]
    ninc    = inc_idx[0].shape[0]
    
    assert (nback+ninc==6305),'inconsistent lengths'
    
    mu_back_correct = correct[iexample,:,:][back_idx]
    mu_inc_correct  = correct[iexample,:,:][inc_idx]
    mu_2t3_correct  = correct[iexample,:,:][mask2t3]
    mu_3t4_correct  = correct[iexample,:,:][mask3t4]
    mu_4t5_correct  = correct[iexample,:,:][mask4t5]
    
    n2t3    = mu_2t3_correct.shape[0]
    n3t4    = mu_3t4_correct.shape[0]
    n4t5    = mu_4t5_correct.shape[0]

    
    mu_back_pred  = predicted[iexample,:,:][back_idx]
    mu_inc_pred   = predicted[iexample,:,:][inc_idx]
    mu_2t3_pred   = predicted[iexample,:,:][mask2t3]
    mu_3t4_pred   = predicted[iexample,:,:][mask3t4]
    mu_4t5_pred   = predicted[iexample,:,:][mask4t5]
    
    #compute norm for each example
    #back_norm[iexample] = np.linalg.norm(mu_back_pred-mu_back_correct)/np.linalg.norm(mu_back_correct)
    #inc_norm[iexample]  = np.linalg.norm(mu_inc_pred-mu_inc_correct)/np.linalg.norm(mu_inc_correct)
    
    back_norm[iexample]  = np.sum((mu_back_pred-mu_back_correct)/mu_back_correct)/nback
    inc_norm[iexample]   = np.sum((mu_inc_pred-mu_inc_correct)/mu_inc_correct)/ninc
    
    if n2t3 !=0:
        nexamples2t3 +=1
        twotothree_norm[iexample] = np.sum((mu_2t3_pred-mu_2t3_correct)/mu_2t3_correct)/n2t3
        
    if n3t4 !=0:
        nexamples3t4 +=1
        threetofour_norm[iexample] = np.sum((mu_3t4_pred-mu_3t4_correct)/mu_3t4_correct)/n3t4
        
    if n4t5 !=0:
        nexamples4t5 +=1
        fourtofive_norm[iexample] = np.sum((mu_4t5_pred-mu_4t5_correct)/mu_4t5_correct)/n4t5

back_ave = np.sum(back_norm)/nexamples
inc_ave  = np.sum(inc_norm)/nexamples
twotothree_ave  = np.sum(twotothree_norm)/nexamples2t3
threetofour_ave = np.sum(threetofour_norm)/nexamples3t4
fourtofive_ave  = np.sum(fourtofive_norm)/nexamples4t5

print('-'*80)
print(f'{cnnidx=}')
print(f'{back_ave=}')
print(f'{inc_ave=}')
print(f'{twotothree_ave=},{nexamples2t3=}')
print(f'{threetofour_ave=},{nexamples3t4=}')
print(f'{fourtofive_ave=},{nexamples4t5=}')
print('-'*80)
