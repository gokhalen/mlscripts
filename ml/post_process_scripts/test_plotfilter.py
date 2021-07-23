# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:29:38 2021

@author: aa
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plotfilter(xx,yy,field,title,cmin=None,cmax=None,shading='flat',fname='out.png',outputdir='.'):
    
# https://stackoverflow.com/questions/17158382/centering-x-tick-labels-between-tick-marks-in-matplotlib
# this plots the 'element number' on the x and y axis
# this has been specifically designed to work with matplotlib 3.1.2
# for this to work xx,yy should contain one more row and column than field
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.pcolormesh.html    

    if (cmin==None):   cmin = np.min(field)
    if (cmax==None):   cmax = np.max(field)
    
    plt.figure(title)
    plt.pcolormesh(xx,yy,field,shading=shading)
    plt.title(title)
    plt.colorbar()
    plt.clim([cmin,cmax])
    ax = plt.gca()

    nrows = xx.shape[0]
    ncols = xx.shape[1]

    # set tick locations and then labels for those locations
    xticksloc = [ 0.5*(xx[0,_i]+xx[0,_i+1]) for _i in range(ncols-1)]
    yticksloc = [ 0.5*(yy[_i,0]+yy[_i+1,0]) for _i in range(nrows-1)]
    xticks    = [ int(_f+0.5) for _f in xticksloc]
    yticks    = [ int(_f+0.5) for _f in yticksloc]
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(xticksloc))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(xticks))

    ax.yaxis.set_major_locator(ticker.FixedLocator(yticksloc))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(yticks))

    plt.xlabel('XX')
    plt.ylabel('YY')
  
    ax.set_aspect('equal')

    plt.savefig(outputdir+'/'+fname,bbox_inches='tight')
    plt.close()


rows=4
cols=2


xvec = np.linspace(0,cols,cols+1)
yvec = np.linspace(0,rows,rows+1)

xx,yy = np.meshgrid(xvec,yvec)
data = np.arange(rows*cols).reshape(rows,cols)


plotfilter(xx,yy,data,cmin=None,cmax=None,shading='flat',title='nach',fname='nach') 
 

