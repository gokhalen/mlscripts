import matplotlib as mpl
# https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x-server-loc See nanounanue's answer
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def plotall_and_save(mltype,iptype,history,outputdir):
    # plot everything in history
    # history is a dictionary which maps keys to lists of numbers
    plot_title  = mltype+'_'+iptype
         
    for ikey in history.keys():
        plt.figure(ikey)
        data   = history[ikey]
        epochs = range(1,len(data)+1)
        yscale = 'linear'
        # plot losses on log scale
        if 'loss' in ikey:
            yscale = 'log'
        plt.plot(epochs,data)
        plt.yscale(yscale)
        plt.title(plot_title)
        plt.xlabel('epochs')
        plt.ylabel(ikey)
        plt.grid(True,which='both')
        plt.tight_layout()
        plt.savefig(f'{outputdir}/{plot_title}'+'_plot_'+ikey+'.png')
        np.save(arr=data,file=f'{outputdir}/{plot_title}'+'_plot_'+ikey)

def plotcurves(xdata,ydata,xlabel,ylabel,title,outputdir,legend=None,fname=None,lw=1):

    plt.figure(title)
    for yy in ydata:
        plt.plot(xdata,yy,linewidth=lw)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if (legend is not None):
        plt.legend(legend,bbox_to_anchor=[1.05,1])

    plt.tight_layout()
    if (fname is not None):
        plt.savefig(outputdir+'/'+fname)  

def plotfield(xx,yy,field,title,fname,outputdir):
    
    plt.figure(title)
    plt.pcolormesh(xx,yy,field)
    plt.title(title)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(outputdir+'/'+fname)
    plt.close()


def subplotfields(xx,yy,fields,titles,fname,outputdir):
    # fields and titles are iterables
    # fields - fields to be plotted
    # titles - titles for the subplots
    
    nf = len(fields)
    nt = len(titles)
    assert (nf == nt),f'Number of fields {nf} should be equal to number of titles {nt}'

    # compute maximum and minimum over all input fields
    cmax = np.max(np.asarray(fields))
    cmin = np.min(np.asarray(fields))
    
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
    plt.savefig(outputdir+'/'+fname)
    plt.close()

