import numpy as np
import json
import glob
import os
import sys

from .datastrc import *
from .plotting import *
from .config   import mltypelist

from sklearn.preprocessing import StandardScaler


def split_cnndata(cnndata,start,stop):
    assert(stop > start),'stop should be > start in split_cnndata'
    labels = Labels(binary = cnndata.labels.binary[start:stop,...],
                    center = cnndata.labels.center[start:stop,...],
                    radius = cnndata.labels.radius[start:stop,...],
                    value  = cnndata.labels.value[start:stop,...],
                    field  = cnndata.labels.field[start:stop,...]
                    )
    out    = CNNData(images=cnndata.images[start:stop,...],
                  labels=labels)
    return out

def get_data(params):
    # reads training,validation and test data and returns it
    # prefix is the prefix of the directories which contain training validation and test data
    # ntrain,nvalid,ntest are integers and should sum up to less than the number of files
    # returned by glob
    # train_data is a namedtuple with four components image,label,center,value,radius
    
    train_data,valid_data,test_data = None,None,None

    ntrain = params['ntrain'];
    nvalid = params['nvalid'];
    ntest  = params['ntest'];
    prefix = params['prefix'];
    nnodex = params['nelemx']+1;
    nnodey = params['nelemy']+1;
    nsum   = ntrain+nvalid+ntest

    bool_files_exist = (os.path.exists('images.npy') and
                        os.path.exists('binary.npy') and
                        os.path.exists('center.npy') and
                        os.path.exists('radius.npy') and
                        os.path.exists('value.npy')  and
                        os.path.exists('field.npy')  
                        )
    
    if bool_files_exist :
        print('-'*80,f'\n Loading previously created .npy files directly\n','-'*80,sep='')
        images = np.load('images.npy')
        binary = np.load('binary.npy')
        center = np.load('center.npy')
        radius = np.load('radius.npy')
        value  = np.load('value.npy')
        field  = np.load('field.npy')

        assert ( images.shape[0] ==
                 binary.shape[0] ==
                 center.shape[0] ==
                 radius.shape[0] ==
                 value.shape[0]  ==
                 field.shape[0]
               ),'Arrays should have same first dimension'

        assert ( images.shape[0] >= nsum ), 'Not enough training examples in saved files'
        

        labels = Labels(binary=binary,
                        center=center,
                        radius=radius,
                        value=value,
                        field=field
                       )
        
        full_data = CNNData(images=images,labels=labels)
    else:
        print('-'*80,f'\n Previously created files not found, processing data and creating .npy files','-'*80,sep='')
        # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory?rq=1
        # Udit Bansal's answer
        gg = glob.glob(f'{params["prefix"]}*/')  # get directories starting with prefix
        ntotal = len(gg);
        assert (ntotal >= nsum),'Number of examples not sufficient'
        full_data  = read_data(0,nsum,prefix,nnodex,nnodey,'full_data')
        np.save('images',full_data.images)
        np.save('binary',full_data.labels.binary)
        np.save('center',full_data.labels.center)
        np.save('radius',full_data.labels.radius)
        np.save('value', full_data.labels.value)
        np.save('field', full_data.labels.field)

        
    train_data = split_cnndata(full_data,0,ntrain)
    valid_data = split_cnndata(full_data,ntrain,ntrain+nvalid)
    test_data  = split_cnndata(full_data,ntrain+nvalid,nsum)

    return train_data,valid_data,test_data


def read_data(start,stop,prefix,nnodex,nnodey,strtype):
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
    # if we get to doing full field ML we'll put in the right dimensions
    field_label  = np.ones((nexamples,),dtype='float64')               
    
    # iloc is training example, ii is file suffix
    for iloc,ii in enumerate(range(start,stop)):
        print(f'Reading example {iloc+1} of {nexamples} for {strtype}')
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
            if dd['stftype'] == 'homogeneous':
                print('WARNING: Homogeneous mu_label set to mu_back')
                mu_label[iloc] = dd['muback']


        # get solution (displacement data)
        with open(outputname,'r') as fin:
            dd   = json.load(fin)
            sol  = np.asarray(dd['solution'])
            # do not reshape.(nnodey,nnodex)
            solx = sol[:,0].reshape(nnodex,nnodey).T
            soly = sol[:,1].reshape(nnodex,nnodey).T
            images[iloc,:,:,0] = solx
            images[iloc,:,:,1] = soly

        plotfield(xx,yy,images[iloc,:,:,0],'ux',outsolx)
        plotfield(xx,yy,images[iloc,:,:,1],'uy',outsoly)
        
    ll  = Labels(binary=binary_label,center=center_label,radius=radius_label,value=mu_label,field=field_label)
    out = CNNData(images=images,labels=ll)

    return out

# assuming well scaled data - not creating scalers at this time
def create_scalers(train_data):
    pass
    
def forward_scale_data():
    pass

def inverse_scale_data():
    pass


