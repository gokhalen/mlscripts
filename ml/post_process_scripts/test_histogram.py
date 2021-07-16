# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:50:29 2021

@author: aa
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

nn=1024*1024
noise_std_dev = 0.1
noise = np.random.normal(0.0,noise_std_dev,size=(nn,))
max_val =  np.max(noise)
min_val =  np.min(noise)
bins    =  np.linspace(min_val,max_val,20)
# https://stackoverflow.com/questions/57026223/how-to-re-scale-the-counts-in-a-matplotlib-histogram
# See ImportanceOfBeingEarnest's comment
values,bins,artists=plt.hist(noise,bins=bins,weights=(1.0/nn)*np.ones_like(noise))
plt.title('original') 


binwidthper = 5 #binwidth in per
binwidth    = binwidthper/100
leftn       = -(int(min_val/binwidth)-1)
rightn      = int(max_val/binwidth)+1
leftval     = -leftn*0.05
rightval    =  rightn*0.05
newbins     = np.linspace(leftval,rightval,leftn+rightn+1)

fig = plt.figure(figsize=(8,4))
ax  = fig.gca()

values2,bin2,artists2=plt.hist(noise,bins=newbins,weights=(1.0/nn)*np.ones_like(noise))
plt.grid(True,which='both')

# major_xtick_label = [str(int(round(round(_f,2)*100,2))) for _f in newbins]
major_xtick_label = list(range(-leftn*binwidthper,rightn*binwidthper,binwidthper))
ax.xaxis.set_major_locator(ticker.FixedLocator(newbins))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(major_xtick_label))

