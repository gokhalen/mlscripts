import numpy as np
import json
import glob
import matplotlib.pyplot as plt

from .datastrc import *
from sklearn.preprocessing import StandardScaler


def get_data(params):
    # reads training,validation and test data and returns it
    # prefix is the prefix of the directories which contain training validation and test data
    # ntrain,nvalid,ntest are integers and should sum up to less than the number of files
    # returned by glob
    # train_data is a namedtuple with four components image,label,center,value,radius
    
    train_data,valid_data,test_data = None,None,None
    # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory?rq=1
    # Udit Bansal's answer
    gg = glob.glob(f'{params["prefix"]}*/')  # get directories starting with prefix

    ntotal = len(gg);
    ntrain = params['ntrain'];
    nvalid = params['nvalid'];
    ntest  = params['ntest'];
    prefix = params['prefix'];
    nnodex = params['nelemx']+1;
    nnodey = params['nelemy']+1;
    
    assert (ntotal >= (ntrain+nvalid+ntest)),'Number of examples not sufficient'

    train_data = read_data(0,ntrain,prefix,nnodex,nnodey);
    valid_data = read_data(ntrain,ntrain+nvalid,prefix,nnodex,nnodey);
    test_data  = read_data(ntrain+nvalid,ntotal,prefix,nnodex,nnodey);
               
    return train_data,valid_data,test_data

def read_data(start,stop,prefix,nnodex,nnodey):
    # get ndime - better be consistent across all training examples.
    inputname  = prefix+'0/input0.json.in';
    with open(inputname,'r') as fin:
        dd    = json.load(fin)
        ndime = dd['ndime']
        coord = np.asarray(dd['coord'])
        # do not .reshape(nnodey,nnodex)
        xx    = coord[:,0].reshape(nnodex,nnodey).T
        yy    = coord[:,1].reshape(nnodex,nnodey).T
    
    nexamples    = stop - start
    images       = np.empty((nexamples,nnodey,nnodex,ndime),dtype='float64')
    binary_label = np.empty((nexamples,),dtype='int64')
    center_label = np.empty((nexamples,2),dtype='float64')
    radius_label = np.empty((nexamples,),dtype='float64')
    mu_label     = np.empty((nexamples,),dtype='float64')
    
    # iloc is training example, ii is file suffix
    for iloc,ii in enumerate(range(start,stop)):

        mlinfoname = prefix+str(ii)+'/mlinfo'+str(ii)+'.json.in';
        outputname = prefix+str(ii)+'/output'+str(ii)+'.json.out';
        outsolx    = prefix+str(ii)+'/uxml'+str(ii)+'.png';
        outsoly    = prefix+str(ii)+'/uyml'+str(ii)+'.png';
        
        with open(mlinfoname,'r') as fin:
            dd = json.load(fin)
            binary_label[iloc] = dd['label'][0]
            center_label[iloc] = np.asarray(dd['centers'])  # assume only one inclusion
            radius_label[iloc] = dd['radii'][0]
            mu_label[iloc]     = dd['mu']

        # get solution (displacement data)
        with open(outputname,'r') as fin:
            dd   = json.load(fin)
            sol  = np.asarray(dd['solution'])
            # do not reshape.(nnodey,nnodex)
            solx = sol[:,0].reshape(nnodex,nnodey).T
            soly = sol[:,1].reshape(nnodex,nnodey).T
            images[iloc,:,:,0] = solx
            images[iloc,:,:,1] = soly

        # plotfield(xx,yy,images[iloc,:,:,0],'ux',outsolx)
        # plotfield(xx,yy,images[iloc,:,:,1],'uy',outsoly)
    ll = Labels(binary=binary_label,center=center_label,radius=radius_label,value=mu_label,field=None)
    out = CNNData(images=images,labels=ll)
    return out


def create_scalers(train_data):
    pass
    
def forward_scale_data():
    pass

def inverse_scale_data():
    pass

def plotfield(xx,yy,field,title,fname):
    plt.figure(title)
    plt.pcolormesh(xx,yy,field)
    plt.title(title)
    plt.colorbar()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(fname)
    plt.close()

