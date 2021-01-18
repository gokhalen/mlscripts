import matplotlib as mpl
# https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x-server-loc See nanounanue's answer
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from .config import outputdir

def plotall_and_save(mltype,iptype,history,outputdir=outputdir):
    # plot everything in history
    # history is returned by model.fit
    title_key  = mltype+'_'+iptype
    title_dict = {'binary_images':'for binary classification from displacement data',
                  'center_images':'for prediction of inclusion center location (x,y) from displacement data',
                  'radius_images':'for prediction of inclusion radius from displacement data',
                  'value_images':'for prediction of inclusion shear modulus value from displacement data',
                  'binary_strain':'for binary classification from strain data',
                  'center_strain':'for prediction of inclusion center location (x,y) from strain data',
                  'radius_strain':'for prediction of inclusion radius from strain data',
                  'value_strain':'for prediction of inclusion shear modulus value from strain data',
                  }

    if (not os.path.exists(outputdir)):
        os.mkdir(outputdir)
         
    for ikey in history.history.keys():
        plt.figure(ikey)
        data   = history.history[ikey]
        epochs = range(1,len(data)+1)
        yscale = 'linear'
        # plot losses on log scale
        if 'loss' in ikey:
            yscale = 'log'
        plt.plot(epochs,data)
        plt.yscale(yscale)
        plt.title(ikey+' '+ title_dict[title_key])
        plt.xlabel('epochs')
        plt.ylabel(ikey)
        plt.grid(True,which='both')
        plt.savefig(f'{outputdir}/{title_key}'+'_plot_'+ikey+'.png')
        np.save(arr=data,file=f'{outputdir}/{title_key}'+'_plot_'+ikey)


def plotcurves(xdata,ydata,xlabel,ylabel,title,legend=None,fname=None,lw=1,outputdir=outputdir):

    if ( not os.path.exists(outputdir)):
        os.mkdir(outputdir)

    plt.figure(title)
    for yy in ydata:
        plt.plot(xdata,yy,linewidth=lw)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (legend is not None):
        plt.legend(legend)
    if (fname is not None):
        plt.savefig(outputdir+'/'+fname)  

def plotfield(xx,yy,field,title,fname,outputdir=outputdir):

    if ( not os.path.exists(outputdir)):
        os.mkdir(outputdir)
    
    plt.figure(title)
    plt.pcolormesh(xx,yy,field)
    plt.title(title)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(outputdir+'/'+fname)
    plt.close()

