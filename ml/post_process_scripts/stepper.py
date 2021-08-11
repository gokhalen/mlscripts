# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 09:24:29 2021

@author: User
"""

import json
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def subplotfields(xx,yy,fields,titles,cmin,cmax,fname,outputdir):
    # fields and titles are iterables
    # fields - fields to be plotted
    # titles - titles for the subplots
    
    # philn's answer
    # https://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    
    
    nf = len(fields)
    nt = len(titles)
    assert (nf == nt),f'Number of fields {nf} should be equal to number of titles {nt}'

    
    fig,axes = plt.subplots(nrows=1,ncols=nt)
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.7,
                        wspace=0.12, hspace=0.05)

    cmap=cm.get_cmap('viridis')
    normalizer=Normalize(cmin,cmax)
    im=cm.ScalarMappable(norm=normalizer,cmap=cmap)
    
    for it,ax in enumerate(axes.flat):
        ax.pcolormesh(xx,yy,fields[it],cmap=cmap,norm=normalizer)
        ax.set_aspect('equal')
        ax.tick_params(axis='both',which='both',
                       bottom=False,top=False,left=False,right=False,
                       labelbottom=False,
                       labelleft=False
                       )
        ax.set_title(titles[it])               
    
    yticks        = np.linspace(cmin,cmax,5) 
    ytick_labels  = np.round(yticks,decimals=1)
    cbar=fig.colorbar(im, ax=axes.ravel().tolist(),shrink=0.59,pad=0.02)
    cbar.set_ticks(yticks)
    cbar.set_ticklabels(ytick_labels)    
    plt.show()
    plt.savefig(outputdir+'/'+fname,bbox_inches='tight')
    plt.close()


with open('logcnn3.txt','rt') as fin:
    dd1=json.load(fin)
    
scaled_error_list = dd1['scaled_error_list']
sorted_idx_list   = dd1['sorted_idx_list']

# parameters
nnodex    = 65
nnodey    = 97

coord  = np.load('coord.npy')    
xx     = coord[:,0].reshape(nnodex,nnodey).T
yy     = coord[:,1].reshape(nnodex,nnodey).T

true       = np.load('correct.npy')
true       = true[...,1]
prediction = np.load('predictioncnn3.npy')

for ictr,ifield in enumerate(sorted_idx_list):
    # combine arrays to find min and max for each ifield
    _true   = true[ifield,:,:]
    _pred   = prediction[ifield,:,:]
    #_error  = _pred - _true
    
    mulist  = [_true,_pred]
    combarr = np.hstack(mulist)

    cmin = np.min(combarr)
    cmax = np.max(combarr)
    outfilename = f'mu_{ictr}_{ifield}.png'
    
    subplotfields(xx,yy,mulist,['true','prediction'],cmin,cmax,outfilename,'sortedpics')
    print(f'{ifield=},scaled_error={scaled_error_list[ifield]}')
