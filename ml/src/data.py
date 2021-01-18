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
                     strain=cnndata.strain[start:stop,...],
                     labels=labels)
    return out

def get_data(ntrain,nvalid,ntest,nnodex,nnodey,prefix):
    # reads training,validation and test data and returns it
    # prefix is the prefix of the directories which contain training validation and test data
    # ntrain,nvalid,ntest are integers and should sum up to less than the number of files
    # returned by glob
    # train_data is a namedtuple with four components image,label,center,value,radius
    
    train_data,valid_data,test_data = None,None,None

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

        strain    = np.load('strain.npy')
        full_data = CNNData(images=images,strain=strain,labels=labels)
    else:
        print('-'*80,f'\nPreviously created files not found, processing data and creating .npy files\n','-'*80,sep='')
        # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory?rq=1
        # Udit Bansal's answer
        gg = glob.glob(f'{prefix}*/')  # get directories starting with prefix
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
    field_label  = np.empty((nexamples,nnodey,nnodex,2),dtype='float64')               
    
    # iloc is training example, ii is file suffix
    for iloc,ii in enumerate(range(start,stop)):
        print(f'Reading example {iloc+1} of {nexamples} for {strtype}')
        mlinfoname = prefix+str(ii)+'/mlinfo'+str(ii)+'.json.in';
        outputname = prefix+str(ii)+'/output'+str(ii)+'.json.out';
        inputname  = prefix+str(ii)+'/input'+str(ii)+'.json.in';
        
        outsolx = prefix+str(ii)+'/uxml'+str(ii)+'.png';
        outsoly = prefix+str(ii)+'/uyml'+str(ii)+'.png';
        outlam  = prefix+str(ii)+'/lamml'+str(ii)+'.png';
        outmu   = prefix+str(ii)+'/muml'+str(ii)+'.png';
        
        with open(mlinfoname,'r') as fin:
            dd = json.load(fin)
            binary_label[iloc] = dd['label'][0]
            center_label[iloc] = np.asarray(dd['centers'])  # assume only one inclusion
            radius_label[iloc] = dd['radii'][0]
            mu_label[iloc]     = dd['mu']
            if dd['stftype'] == 'homogeneous':
                print('WARNING: Homogeneous mu_label set to mu_back')
                mu_label[iloc]     = dd['muback']
                print('WARNING: Homogeneous center_label set to -1.0,-1.0')
                center_label[iloc] = -1.0,-1.0
                print('WARNING: Homogeneous radius_label set to -1.0')
                radius_label[iloc] = -1.0
        # get solution (displacement data)
        with open(outputname,'r') as fin:
            dd   = json.load(fin)
            sol  = np.asarray(dd['solution'])
            # do not reshape.(nnodey,nnodex)
            solx = sol[:,0].reshape(nnodex,nnodey).T
            soly = sol[:,1].reshape(nnodex,nnodey).T
            images[iloc,:,:,0] = solx
            images[iloc,:,:,1] = soly

        # get field
        with open(inputname,'r') as fin:
            dd   = json.load(fin)
            prop = np.asarray(dd['prop'])
            lam  = prop[:,0].reshape(nnodex,nnodey).T
            mu   = prop[:,1].reshape(nnodex,nnodey).T
            field_label[iloc,:,:,0] = lam
            field_label[iloc,:,:,1] = mu
            

        plotfield(xx,yy,images[iloc,:,:,0],'ux',outsolx)
        plotfield(xx,yy,images[iloc,:,:,1],'uy',outsoly)
        
        plotfield(xx,yy,field_label[iloc,:,:,0],'lam',outlam)
        plotfield(xx,yy,field_label[iloc,:,:,1],'mu',outmu)

    listbin = list(binary_label)
    _h      = listbin.count(0)
    _nh     = listbin.count(1)
    
    print(f'Found {_nh} non-homogeneous and {_h} homogeneous examples out of a total {_nh+_h}')
    strain = np.load('strain.npy')
    ll     = Labels(binary=binary_label,center=center_label,radius=radius_label,value=mu_label,field=field_label)
    out    = CNNData(images=images,strain=strain,labels=ll)

    return out

def forward_scale_all(datatuple,length,breadth,valmin,valmax,valave):
    # datatuple - tuple containing CNNData to scale
    ll = []
    for d in datatuple:
        ss = forward_scale_data(data=d,
                                length=length,
                                breadth=breadth,
                                valmin=valmin,
                                valmax=valmax,
                                valave=valave
                               )
        ll.append(ss)

    return tuple(ll)

def inverse_scale_all(datatuple,length,breadth,valmin,valmax,valave):
    # datatuple - tuple containing CNNData to scale
    ll = []
    for d in datatuple:
        ss = inverse_scale_data(data=d,
                                length=length,
                                breadth=breadth,
                                valmin=valmin,
                                valmax=valmax,
                                valave=valave
                               )
        ll.append(ss)

    return tuple(ll)
        

def forward_scale_data(data,length,breadth,valmin,valmax,valave):
    scaled_center = forward_scale_center(data.labels.center,length,breadth)
    scaled_value  = forward_scale_value(data.labels.value,valmin,valmax,valave)
    
    labels = Labels(binary = data.labels.binary,
                    center = scaled_center,
                    radius = data.labels.radius,
                    value  = scaled_value,
                    field  = data.labels.field
                   )
    return CNNData(images=data.images,
                   strain=data.strain,
                   labels=labels
                   )

def inverse_scale_data(data,length,breadth,valmin,valmax,valave):
    scaled_center = inverse_scale_center(data.labels.center,length,breadth)
    scaled_value  = inverse_scale_value(data.labels.value,valmin,valmax,valave)
    labels = Labels(binary = data.labels.binary,
                    center = scaled_center,
                    radius = data.labels.radius,
                    value  = scaled_value,
                    field  = data.labels.field
                   )
    return CNNData(images=data.images,
                   strain=data.strain,
                   labels=labels
                   )

def forward_scale_center(centers,length,breadth):
    scaler_center = np.asarray([length,breadth])
    scaled_center = centers / scaler_center
    return scaled_center

def inverse_scale_center(centers,length,breadth):
    scaler_center = np.asarray([length,breadth])
    scaled_center = centers * scaler_center
    return scaled_center

def forward_scale_value(value,valmin,valmax,valave):
    assert (valmax != valmin),f'Valmax should not be equal to valmin in __file__'
    scaled_value = (value - valave)/(valmax - valmin)
    return scaled_value

def inverse_scale_value(value,valmin,valmax,valave):
    unscaled_value = (value*(valmax-valmin) + valave)
    return unscaled_value

def inverse_scale_prediction(mltype,prediction,length,breadth,valmin,valmax,valave):

    if (mltype == 'binary'):
        return prediction
    
    if (mltype == 'center'):
        return inverse_scale_center(prediction,length,breadth)

    if (mltype == 'radius'):
        return prediction

    if (mltype == 'value'):
        return inverse_scale_value(prediction,valmin=valmin,valmax=valmax,valave=valave)

