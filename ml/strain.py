# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 15:50:58 2021

@author: aa
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json


with open('traindata0/input0.json.in','r') as fin:
    dd = json.load(fin)
    
coord  = np.asarray(dd['coord'])
nnodex = dd['nnodex'] 
nnodey = dd['nnodey']
nelemx = nnodex-1
nelemy = nnodey-1

xmin = np.min(coord[:,0])
xmax = np.max(coord[:,0])

ymin = np.min(coord[:,1])
ymax = np.max(coord[:,1])

dx = (xmax-xmin)/nelemx
dy = (ymax-ymin)/nelemy



img  = np.load('images.npy')
nimg = len(img)

strain=np.empty((nimg,nnodey,nnodex,3),dtype='float64')


def plotfield(coord,field,nnodex,nnodey,fieldname,fmin=None,fmax=None,suffix=''):
    # do not try .reshape(self.nnodey,self.nnodex)
    # if you want to switch dimensions, then take the transpose
    xx    = np.asarray(coord)[:,0].reshape(nnodex,nnodey)
    yy    = np.asarray(coord)[:,1].reshape(nnodex,nnodey)
    field = field.reshape(nnodex,nnodey) 

    plt.figure(fieldname)
    plt.pcolormesh(xx,yy,field,vmin=fmin,vmax=fmax)
    plt.title(fieldname)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect('equal')

print('Changed range to 1')

for ii in range(nimg):
    ux=img[ii,:,:,0].T
    uy=img[ii,:,:,1].T
    exx=np.diff(ux,axis=0)/dx
    eyy=np.diff(uy,axis=1)/dy
    ux_y=np.diff(ux,axis=1)/dy
    uy_x=np.diff(uy,axis=0)/dx
    
    # zero padding
    px=np.zeros((1,nnodey),dtype='float64')
    py=np.zeros((nnodex,1),dtype='float64')
        
    exx  = np.vstack((exx,px))
    eyy  = np.hstack((eyy,py))
    ux_y = np.hstack((ux_y,py))
    uy_x = np.vstack((uy_x,px))
    exy  = 0.5*(ux_y + uy_x)
    
    exx  = exx / np.max(np.abs(exx))
    eyy  = eyy / np.max(np.abs(eyy))
    exy  = exy / np.max(np.abs(exy))
      
    strain[ii,:,:,0]=exx.T
    strain[ii,:,:,1]=eyy.T
    strain[ii,:,:,2]=exy.T
    print(ii)


plotfield(coord,ux,nnodex,nnodey,'ux')
plotfield(coord,uy,nnodex,nnodey,'uy')
plotfield(coord,exx,nnodex,nnodey,'exx')
plotfield(coord,eyy,nnodex,nnodey,'eyy')
plotfield(coord,exy,nnodex,nnodey,'exy')
np.save(file='strain',arr=strain)

