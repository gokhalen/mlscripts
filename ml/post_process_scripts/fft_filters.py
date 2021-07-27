# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 10:10:55 2021

@author: aa
"""

import numpy as np
import scipy.fft
import matplotlib.pyplot as plt

def subplotfields(fields,titles,cmin=None,cmax=None,fname='out',outputdir='.'):
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
        plt.pcolor(fields[it])
        # plt.clim([cmin,cmax])
        # yticks = np.linspace(cmin,cmax,7)
        # yticks = np.round(yticks,decimals=2)
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
    # plt.close()


filters=np.load('filtercnn1.npy')
nfilters=filters.shape[-1]
fft_abs=np.zeros((filters.shape[0],filters.shape[1],nfilters),dtype='float64')

xder=np.zeros((3,3),dtype='float64')
xder[1][0] = -1
xder[1][2] = +1
xder_fft_abs = np.abs(scipy.fft.fft2(xder))

print(f'{xder_fft_abs=}')

yder=np.zeros((3,3),dtype='float64')
yder[0][1] = -1
yder[2][1] = +1
yder_fft_abs = np.abs(scipy.fft.fft2(yder))

print(f'{yder_fft_abs=}')



for ifilter in range(0,16):
    print(f'{ifilter=}')
    current_filter   = filters[:,:,0,ifilter]
    current_fft      = scipy.fft.fft2(current_filter)
    current_fft_abs  = np.abs(current_fft)
    fft_abs[:,:,ifilter] = current_fft_abs 
    plt_titles       = ['filter'+str(ifilter),'absfft'+str(ifilter)]
    subplotfields([current_filter,current_fft_abs],plt_titles,fname='out',outputdir='filters')
    
    