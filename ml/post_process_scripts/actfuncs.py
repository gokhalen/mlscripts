# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:48:41 2021

@author: aa
"""

import tensorflow as tf
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

xmin  = -10.0
xmax  =  10.0
npoints = 1024

xvals = np.linspace(xmin,xmax,npoints)


def plotarr(xx,yy,leg=None,fname=None):
    fontsize=18
    minorticksize=18
    legendfont=16
    fig = plt.figure(figsize=(6,4))
    ax  = fig.gca()
    
    plt.plot(xx,yy,linewidth='4')
    plt.grid(True,which='both',linewidth='2')
    
    plt.locator_params(axis="x", nbins=6)
    plt.locator_params(axis="y", nbins=6)    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tick_params(axis='both', which='minor', labelsize=minorticksize)
    
    
    plt.xlabel('input',fontsize=fontsize)
    plt.ylabel('output',fontsize=fontsize)
    plt.legend([leg],fontsize=legendfont,loc='upper left')
    
    
    plt.tight_layout()    
    plt.savefig(fname,bbox_inches='tight')
    


# plot tanh
yvals = tf.keras.backend.tanh(xvals)
plotarr(xvals,yvals,leg='tanh',fname='tanh.png')
# plot logistic
yvals = tf.keras.backend.sigmoid(xvals)
plotarr(xvals,yvals,leg='logistic',fname='logistic.png')
# plot softplus
yvals = tf.keras.backend.softplus(xvals)
plotarr(xvals,yvals,leg='softplus',fname='softplus.png')
# twisted tanh
yvals = tf.keras.backend.tanh(xvals) + 0.01*xvals
plotarr(xvals,yvals,leg='twisted tanh',fname='twistedtanh.png')
# max softplus
yvals = np.minimum(tf.keras.backend.softplus(xvals),1.0)
plotarr(xvals,yvals,leg='min(softplus,1)',fname='minsoftplus.png')
