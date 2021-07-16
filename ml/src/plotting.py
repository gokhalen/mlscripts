import matplotlib as mpl
# https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x-server-loc See nanounanue's answer
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

def plotall_and_save(mltype,iptype,activation,history,outputdir):
    # plot everything in history
    # history is a dictionary which maps keys to lists of numbers
    plot_title    = mltype+'_'+iptype
    file_title    = mltype+'_'+iptype
    fontsize      = 18
    minorticksize = 14

    # change plot title to latex for the cases in the paper

    if ( plot_title == 'field_strain'):
        plot_title    = f'Training CNN {activation} ' +r'$\epsilon_{xx}$ ' + '& ' +r'$\epsilon_{xy}$ ' +'& ' + r'$\epsilon_{yy}$'

    if ( plot_title == 'field_strainxxyy' ):
        plot_title    = f'Training CNN {activation} ' +r'$\epsilon_{xx}$ ' + '& ' + r'$\epsilon_{yy}$'
        
    if ( plot_title  == 'field_strainyy'):
        plot_title    = f'Training CNN {activation} ' +r'$\epsilon_{yy}$'

    if ( plot_title  == 'field_images'):
        plot_title    = f'Training CNN {activation} ' +r'$u_x$ ' + '& ' + r'$u_y$'

    if ( plot_title  == 'field_imagesy'):
        plot_title    = f'Training CNN {activation} ' +r'$u_y$'

         
    for ikey in history.keys():
        plt.figure(ikey)
        data   = history[ikey]
        epochs = range(1,len(data)+1)
        yscale = 'linear'
        # plot losses on log scale
        if 'loss' in ikey:
            yscale = 'log'
        plt.plot(epochs,data,linewidth='4')
        plt.yscale(yscale)
        plt.title(plot_title,fontsize=fontsize)
        plt.xlabel('epochs',fontsize=fontsize)
        
        if 'val' in ikey:
            plt.ylabel('validation loss',fontsize=fontsize)
        else:
            plt.ylabel('training loss',fontsize=fontsize)
            
        plt.grid(True,which='both',linewidth='2')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tick_params(axis='both', which='minor', labelsize=minorticksize)
        plt.tight_layout()
        plt.savefig(f'{outputdir}/{file_title}'+'_plot_'+ikey+'.png',bbox_inches='tight')
        np.save(arr=data,file=f'{outputdir}/{file_title}'+'_plot_'+ikey)

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
        plt.savefig(outputdir+'/'+fname,bbox_inches='tight')  

def plotfield(xx,yy,field,title,xticks=None,yticks=None,cmin=None,cmax=None,shading='auto',fname='out.png',outputdir='.'):

    if (cmin==None):   cmin = np.min(field)
    if (cmax==None):   cmax = np.max(field)
    
    plt.figure(title)
    plt.pcolormesh(xx,yy,field,shading=shading)
    plt.title(title)
    plt.colorbar()
    plt.clim([cmin,cmax])
    ax = plt.gca()
    
    if xticks != None:  plt.xticks(xticks)
    if yticks != None:  plt.yticks(yticks)
    
    ax.set_aspect('equal')
    plt.savefig(outputdir+'/'+fname,bbox_inches='tight')
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
    plt.savefig(outputdir+'/'+fname,bbox_inches='tight')
    plt.close()

def plotfilter(xx,yy,field,title,cmin=None,cmax=None,xlabel=None,ylabel=None,shading='flat',fname='out.png',outputdir='.'):
    
# https://stackoverflow.com/questions/17158382/centering-x-tick-labels-between-tick-marks-in-matplotlib
# this plots the 'element number' on the x and y axis
# this has been specifically designed to work with matplotlib 3.1.2
# for this to work xx,yy should contain one more row and column than field
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.pcolormesh.html    

    fontsize=18
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

    if ( xlabel != None ): plt.xlabel(xlabel,fontsize=fontsize)
    if ( ylabel != None ): plt.ylabel(ylabel,fontsize=fontsize)
  
    ax.set_aspect('equal')

    plt.savefig(outputdir+'/'+fname,bbox_inches='tight')
    plt.close()
