# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:42:53 2021

@author: aa
"""

import numpy as np
import matplotlib as mpl
# https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x-server-loc See nanounanue's answer
mpl.use('Agg')
import matplotlib.pyplot as plt
import os


def subplotfields(xx,yy,fields,titles,cmin,cmax,fname,outputdir):
    # fields and titles are iterables
    # fields - fields to be plotted
    # titles - titles for the subplots
    
    nf = len(fields)
    nt = len(titles)
    assert (nf == nt),f'Number of fields {nf} should be equal to number of titles {nt}'

    # compute maximum and minimum for current example
    # cmax = np.max(np.asarray(fields))
    # cmin = np.min(np.asarray(fields))
    
    plt.figure()
    
    for it in range(nt):
        plt.subplot(1,nt,it+1)
        plt.pcolormesh(xx,yy,fields[it])
        plt.clim([cmin,cmax])
        yticks = np.linspace(cmin,cmax,7)
        yticks = np.round(yticks,decimals=2)
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        # See Skytaker's answer
        cbar = plt.colorbar(fraction=0.07,pad=0.04)
        # cbar.ax.set_yticklabels(yticks)
        plt.title(titles[it])
        ax = plt.gca()
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    plt.savefig(outputdir+'/'+fname,bbox_inches='tight')
    plt.close()

# parameters
nnodex    = 65
nnodey    = 97
# for single inclusion problem
munumbers = [785]
numexamp  = len(munumbers)
munames   = ['true','softplus','tanhshift0','tanhshiftp25','tanhshiftp50']


# load data
coord  = np.load('coord.npy')    
xx     = coord[:,0].reshape(nnodex,nnodey).T
yy     = coord[:,1].reshape(nnodex,nnodey).T

true         = np.load('correct.npy')
true         = true[...,1]
softplus     = np.load('softplus.npy')
tanhshift0   = np.load('tanhshift0.npy')
tanhshiftp25 = np.load('tanhshiftp25.npy')
tanhshiftp50 = np.load('tanhshiftp50.npy')

mufields    = [true,softplus,tanhshift0,tanhshiftp25,tanhshiftp50]




for ictr,ifield in enumerate(munumbers):
    print(f'Processing example {ictr+1} of {numexamp}')
    dirname = f'ex{ictr+1}'
    # make directory if it does not exist
    if (not os.path.exists(dirname)):
        os.mkdir(dirname)

    # combine arrays to find min and max for each ifield
    combarr = true[ifield,:,:]
    for iarr in mufields[1:]:
        combarr = np.hstack((combarr,iarr[ifield,:,:]))

    cmin = np.min(combarr)
    cmax = np.max(combarr)

    for imuname,imufield in zip(munames,mufields):
        outfilename = 'mu'+imuname
        subplotfields(xx,yy,[imufield[ifield,:,:]],[''],cmin,cmax,outfilename,dirname)
