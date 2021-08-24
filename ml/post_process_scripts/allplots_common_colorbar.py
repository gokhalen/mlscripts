# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:42:53 2021

@author: aa
"""



import numpy as np
import matplotlib as mpl
# https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x-server-loc See nanounanue's answer
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
from matplotlib.colors import Normalize

import os


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
        ax.set_title(titles[it],fontsize=16)               
    
    yticks        = np.linspace(cmin,cmax,5) 
    ytick_labels  = np.round(yticks,decimals=1)
    cbar=fig.colorbar(im, ax=axes.ravel().tolist(),shrink=0.59,pad=0.02)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(yticks)
    cbar.set_ticklabels(ytick_labels)    
    plt.show()
    plt.savefig(outputdir+'/'+fname,bbox_inches='tight')
    #plt.close()

# parameters
nnodex    = 65
nnodey    = 97
# for single inclusion problem
# best,worst,and another two

# ALL MU NUMBERS ARE DIRECT ENTRIES INTO THE PREDICTION ARRAY
# FIRST 4 NUMBERS ARE MAIN ENTRIES NEXT 8 ARE APPENDIX ENTRIES
# OF WHICH THE LAST TWO ARE USED
# CNN1
# munumbers = [1517,1935,521,1087,1381,625,1366,852,719,1813,72,1912] 
# CNN1 errors [0.0705,0.476,0.0298,0.247,0.0838,0.137,0.168,0.185,0.205,0.222
#                       0.243,0.264]

# CNN2
# munumbers = [1517,1935,521,1087,43,1495,120,1552,1204,1828,721,180]
# CNN2 errors [0.157,0.03976,0.327,0.346,0.156,0.205,0.227,0.247,
#                       0.265,0.283,0.312,0.359]  
#                        

# CNN3 
# munumbers = [1517,1935,521,1087,967,615,1008,14,147,1657,702,1126]
# Appendix CNN3 Errors = [0.0754,0.131,0.161,0.181,0.199,0.223,0.249,0.305]
# sorted numbers - 9,253,500,757,1001,1257,1507,1848


# load data
# this index controls which data is loaded CNN1/CNN2/CNN3

cnnidx = 3
munumbers = [1517,1935,521,1087,967,615,1008,14,147,1657,702,1126]
numexamp  = len(munumbers)

coord  = np.load('coord.npy')    
xx     = coord[:,0].reshape(nnodex,nnodey).T
yy     = coord[:,1].reshape(nnodex,nnodey).T

true       = np.load('correct.npy')
true       = true[...,1]
prediction = np.load(f'predictioncnn{cnnidx}.npy')
mufields   = [true,prediction]

with open(f'logcnn{cnnidx}.txt') as fin:
    logdict = json.load(fin)

scaled_error_list = logdict['scaled_error_list']
sorted_idx_list   = logdict['sorted_idx_list']

for ictr,ifield in enumerate(munumbers):
    sc_error = scaled_error_list[ifield]
    print(f'Processing example {ictr+1} of {numexamp}...scaled_error is {sc_error}')
    dirname = f'app{cnnidx}/ex{ictr+1}'
    # make directory if it does not exist
    if (not os.path.exists(dirname)):
        os.makedirs(dirname)

    # combine arrays to find min and max for each ifield
    _true   = true[ifield,:,:]
    _pred   = prediction[ifield,:,:]
    #_error  = _pred - _true
    
    mulist  = [_true,_pred]
    combarr = np.hstack(mulist)

    cmin = np.min(combarr)
    cmax = np.max(combarr)
    
    # print(f'{cmin=},{cmax=}')

    outfilename = 'mu'
    
    subplotfields(xx,yy,mulist,['true','prediction'],cmin,cmax,outfilename,dirname)
