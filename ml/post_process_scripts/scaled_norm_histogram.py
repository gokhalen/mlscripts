# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 17:11:29 2021

@author: aa
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

binwidth = 0.025
fontsize = 14

with open('logcnn1.txt','rt') as fin:
    dd=json.load(fin)
    
scaled_error_array = np.asarray(dd['scaled_error_list'])
nexamples = scaled_error_array.shape[0]

# https://stackoverflow.com/questions/57026223/how-to-re-scale-the-counts-in-a-matplotlib-histogram
# See ImportanceOfBeingEarnest's comment
weights   = (1.0/nexamples)*np.ones_like(scaled_error_array)
average   = np.average(scaled_error_array)
stddev    = np.std(scaled_error_array)
min_val   = np.min(scaled_error_array)
max_val   = np.max(scaled_error_array)

# all values in being plotted are greater than 0
leftn     =  int(min_val/binwidth)
rightn    =  int(max_val/binwidth)+1

assert (leftn>=0),'leftn should be >=0'
assert (rightn>=0),'rightn should be >=0'

leftval  = leftn*binwidth
rightval = rightn*binwidth
newbins  = np.linspace(leftval,rightval,rightn-leftn+1,endpoint=True)


for ictr in range(2):
    fig = plt.figure(figsize=(5,4))
    ax  = fig.gca()
    
    if ictr==0: 
        cumbool=False
        plt_title='histogram'
        plt_save='histogram'
    if ictr==1: 
        cumbool=True
        plt_title='cumulative histogram'
        plt_save='cumulative'
    
    values2,bin2,artists2=plt.hist(scaled_error_array,cumulative=cumbool,
                                   bins=newbins,weights=weights)
    plt.grid(True,which='both')
    plt.xlabel('scaled error',fontsize=fontsize)
    plt.ylabel('fraction of examples',fontsize=fontsize)
    plt.title(plt_title,fontsize=fontsize)
    plt.tight_layout()
    
    xaxis_ticks = newbins[0::]
    major_xtick_label = [str(num)[0:5].rstrip('0') for num in xaxis_ticks]
    major_replace     = ['']*len(major_xtick_label)
    # replace every alternate label with ''
    major_xtick_label[1::2]=major_replace[1::2]
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(xaxis_ticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(major_xtick_label))
    
    plt.tight_layout()
    
    plt.savefig(plt_save,bbox_inches='tight') 

