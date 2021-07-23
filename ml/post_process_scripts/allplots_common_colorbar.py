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
        ax.set_title(titles[it])               
    
    yticks        = np.linspace(cmin,cmax,5) 
    ytick_labels  = np.round(yticks,decimals=1)
    cbar=fig.colorbar(im, ax=axes.ravel().tolist(),shrink=0.59,pad=0.02)
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
# munumbers = [17,724,122,269]  # for CNN1
# munumbers = [389,320,122,269] # for CNN2
munumbers = [389,320,122,269]

numexamp  = len(munumbers)

# load data
coord  = np.load('coord.npy')    
xx     = coord[:,0].reshape(nnodex,nnodey).T
yy     = coord[:,1].reshape(nnodex,nnodey).T

true       = np.load('correct_strainyy.npy')
true       = true[...,1]
prediction = np.load('prediction_imagesy.npy')
mufields   = [true,prediction]


for ictr,ifield in enumerate(munumbers):
    print(f'Processing example {ictr+1} of {numexamp}')
    dirname = f'ex{ictr+1}'
    # make directory if it does not exist
    if (not os.path.exists(dirname)):
        os.mkdir(dirname)

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
