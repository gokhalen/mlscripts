# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:59:52 2021

@author: User
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


loss = np.load('loss_cnn1.npy')
val_loss = np.load('val_loss_cnn1.npy')

def plotlossfunc(data,title,ylabel,yaxis_ticks=None,yaxis_labels=None):
    
    fontsize=22
    linewidth=4
    labelsize=22
    
    fig=plt.figure(figsize=(7,4))
    ax=fig.gca()
    epochs=range(1,len(data)+1)
    
    plt.plot(epochs,data,linewidth=linewidth)
    plt.yscale('log')
    
    plt.title(title,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.xlabel('epochs',fontsize=fontsize)
    
    plt.tick_params(axis='both',which='both',labelsize=labelsize)
    plt.grid(True,which='both',linewidth=2)
   
    if yaxis_ticks is not None and yaxis_labels is not None:
        ax.yaxis.set_major_locator(ticker.FixedLocator(yaxis_ticks))
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(yaxis_labels))
        
        
    plt.tight_layout()
    plt.savefig(title+'.png')

yaxis_ticks=None
yaxis_labels=None    
yaxis_ticks=[0.1,0.06,0.04,0.03,0.02,0.01,0.008,0.005]
yaxis_labels=[r'$10^{-1}$',
              r'$6\times10^{-2}$',
              r'$4\times10^{-2}$',
              r'$3\times10^{-2}$',
              r'$2\times10^{-2}$',
              r'$10^{-2}$',
              r'$8\times10^{-3}$',              
              r'$5\times10^{-3}$',              
              ]    

plotlossfunc(loss,'training loss for CNN1',
            ylabel='training loss',
            yaxis_ticks=yaxis_ticks,
            yaxis_labels=yaxis_labels,
            )

    
    