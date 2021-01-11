import matplotlib as mpl
# https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x-server-loc See nanounanue's answer
mpl.use('Agg')
import matplotlib.pyplot as plt

def plotall(mltype,history):
    # plot everything in history
    # history is returned by model.fit
    title_dict = {'binary':'for binary classification',
                  'center':'for prediction of center'
                  }
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
        plt.title(ikey+' '+ title_dict[mltype])
        plt.xlabel('epochs')
        plt.ylabel(ikey)
        plt.grid(True,which='both')
        plt.savefig('plot_'+ikey+f'_{mltype}'+'.png')


def plotfield(xx,yy,field,title,fname):
    plt.figure(title)
    plt.pcolormesh(xx,yy,field)
    plt.title(title)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(fname)
    plt.close()
