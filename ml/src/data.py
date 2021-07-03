import numpy as np
import json
import glob
import os
import sys
# import copy

from .datastrc import *
from .plotting import *
from .config   import mltypelist

# from sklearn.preprocessing import StandardScaler

def get_max_abs_scaled_array(arr,ntrain):
    # typical input arr is a 4D array
    # scales each arr[...,i] by np.max(np.abs[0:ntrain,...,i])
    nc    = arr.shape[-1]
    alist = []
    for ic in range(nc):
        _max = np.max(np.abs(arr[0:ntrain,...,ic]))
        _arr = arr[...,ic]/_max
        alist.append(_arr)
    return np.stack(alist,axis=-1)

def select_input_comps(data,iptype):
    # data - namedtuple of type cnndata
    # depending on iptype we modify data.images and data.strain
    # there is a case for using default dict here
    # the default returned object would be data.images and data.strain

    # we are doing 0:1 instead of just 0 so that the 4D shape is retained    
    images_dict = {'images':data.images,
                  'imagesx':data.images[...,0:1],    
                  'imagesy':data.images[...,1:2],
                  'strain':data.images,
                  'strainxx':data.images,
                  'strainyy':data.images,
                  'strainxxyy':data.images
                 }

    strain_dict = {'images':data.strain,
                  'imagesx':data.strain,
                  'imagesy':data.strain,
                  'strain':data.strain,
                  'strainxx':data.strain[...,0:1],
                  'strainyy':data.strain[...,1:2],
                  'strainxxyy':data.strain[...,0:2]
                 }
    
    return CNNData(images=images_dict[iptype],
                   strain=strain_dict[iptype],
                   labels=data.labels
                   )


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

def normalize_input_cnndata(data,ntrain,nvalid,ntest,inputscale):
    # should be called after select_input_comps
    # this does not modify data. maybe it should. it would save memory
    # note: normalization parameters are computed over training images only

    if (inputscale == 'global'):
        # find the maximum of data.images and data.strain and scale by them
        umax = np.max(np.abs(data.images[0:ntrain,...]))
        emax = np.max(np.abs(data.strain[0:ntrain,...]))
        new_images = data.images/umax
        new_strain = data.strain/emax

    if (inputscale == 'individual'):
        new_images = get_max_abs_scaled_array(data.images,ntrain=ntrain)
        new_strain = get_max_abs_scaled_array(data.strain,ntrain=ntrain) 


    cnndata_norm_input = CNNData(images=new_images,
                                 strain=new_strain,
                                 labels=data.labels
                                )
    
    return cnndata_norm_input


def get_data(ntrain,nvalid,ntest,nnodex,nnodey,noise,inputscale,prefix,outputdir,iptype):
    # reads training,validation and test data and returns it
    # select_input_comps and normalize_cnndata are called
    # strain and image are normalized
    
    
    # prefix is the prefix of the directories which contain training validation and test data
    # ntrain,nvalid,ntest are integers and should sum up to less than the number of files
    # returned by glob
    # train_data is a namedtuple with four components image,label,center,value,radius

    # reads the complete data, both displacements, all three strains
    # if we need fewer components, for training, CNN definition and prediction it is handled in
    # define_cnn,train_cnn,predict_cnn
    
    train_data,valid_data,test_data = None,None,None

    nsum   = ntrain+nvalid+ntest

    bool_files_exist = (os.path.exists('images.npy') and
                        os.path.exists('strain.npy') and 
                        os.path.exists('binary.npy') and
                        os.path.exists('center.npy') and
                        os.path.exists('radius.npy') and
                        os.path.exists('value.npy')  and
                        os.path.exists('field.npy')  and
                        os.path.exists('coord.npy')
                        )
    
    if bool_files_exist :
        print('-'*80,f'\n Loading previously created .npy files directly\n','-'*80,sep='')
        images = np.load('images.npy')
        strain = np.load('strain.npy')
        binary = np.load('binary.npy')
        center = np.load('center.npy')
        radius = np.load('radius.npy')
        value  = np.load('value.npy')
        field  = np.load('field.npy')

        assert ( images.shape[0] ==
                 strain.shape[0] ==
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

        full_data = CNNData(images=images,strain=strain,labels=labels)
    else:
        print('-'*80,f'\nPreviously created files not found, processing data and creating .npy files\n','-'*80,sep='')
        # https://stackoverflow.com/questions/973473/getting-a-list-of-all-subdirectories-in-the-current-directory?rq=1
        # Udit Bansal's answer
        gg = glob.glob(f'{prefix}*/')  # get directories starting with prefix
        ntotal = len(gg);
        assert (ntotal >= nsum),'Number of examples not sufficient'
        full_data  = read_data(0,nsum,prefix,nnodex,nnodey,'full_data',outputdir=outputdir)

        #  read coord
        with open(f'{prefix}0/input0.json.in') as fin:
            dd = json.load(fin)
            coord = np.asarray(dd['coord'])
            
        
        np.save('images',full_data.images)
        np.save('strain',full_data.strain)
        np.save('binary',full_data.labels.binary)
        np.save('center',full_data.labels.center)
        np.save('radius',full_data.labels.radius)
        np.save('value', full_data.labels.value)
        np.save('field', full_data.labels.field)
        np.save('coord',coord)


    # first select, then add noise to test examples, then normalize
    full_data  = select_input_comps(data=full_data,iptype=iptype)
    
    # full_data_copy = copy.deepcopy(full_data)
    full_data  = addnoise(data=full_data,noise=noise,ntrain=ntrain,nvalid=nvalid,ntest=ntest,nnodex=nnodex,nnodey=nnodey)
    #print('no noise images= ',np.linalg.norm(full_data.images[0:ntrain+nvalid,...]-full_data_copy.images[0:ntrain+nvalid,...]))
    #print('noise images= ',np.linalg.norm(full_data.images[ntrain+nvalid:,...]-full_data_copy.images[ntrain+nvalid:,...]))
    #print('no noise strain= ',np.linalg.norm(full_data.strain[0:ntrain+nvalid,...]-full_data_copy.strain[0:ntrain+nvalid,...]))
    #print('noise strain= ',np.linalg.norm(full_data.strain[ntrain+nvalid:,...]-full_data_copy.strain[ntrain+nvalid:,...]))
    
    full_data  = normalize_input_cnndata(data=full_data,ntrain=ntrain,nvalid=nvalid,ntest=ntest,inputscale=inputscale)


    
    train_data = split_cnndata(full_data,0,ntrain)
    valid_data = split_cnndata(full_data,ntrain,ntrain+nvalid)
    test_data  = split_cnndata(full_data,ntrain+nvalid,nsum)

    return train_data,valid_data,test_data


def read_data(start,stop,prefix,nnodex,nnodey,strtype,outputdir):
    # get ndime - better be consistent across all training examples.
    inputname  = prefix+'0/input0.json.in';
    with open(inputname,'r') as fin:
        dd    = json.load(fin)
        ndime = dd['ndime']
        coord = np.asarray(dd['coord'])
        # do not .reshape(nnodey,nnodex)
        xx    = coord[:,0].reshape(nnodex,nnodey).T
        yy    = coord[:,1].reshape(nnodex,nnodey).T
        xmin  = np.min(coord[:,0])
        xmax  = np.max(coord[:,0])
        ymin  = np.min(coord[:,1])
        ymax  = np.max(coord[:,1])

    nelemx = nnodex - 1
    nelemy = nnodey - 1
    dx = (xmax-xmin)/nelemx
    dy = (ymax-ymin)/nelemy
    
    nexamples    = stop - start
    images       = np.empty((nexamples,nnodey,nnodex,ndime),dtype='float64')
    strain       = np.empty((nexamples,nnodey,nnodex,ndime+1),dtype='float64')
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
        
        outsolx = '/uxml'+str(ii)+'.png';
        outsoly = '/uyml'+str(ii)+'.png';
        outlam  = '/lamml'+str(ii)+'.png';
        outmu   = '/muml'+str(ii)+'.png';
        outexx  = '/exxml'+str(ii)+'.png';
        outeyy  = '/eyyml'+str(ii)+'.png';
        outexy  = '/exyml'+str(ii)+'.png';
        
        with open(mlinfoname,'r') as fin:
            dd = json.load(fin)
            binary_label[iloc] = dd['label'][0]
            # center_label, radius_label, mu_label are used only for parameter prediction
            # they are not used for field prediction
            center_label[iloc] = np.asarray(dd['centers'])[-1] 
            radius_label[iloc] = dd['radii'][-1]
            mu_label[iloc]     = dd['mu'][-1]
            if dd['stftype'] == 'homogeneous':
                print('WARNING: Homogeneous mu_label set to mu_back')
                mu_label[iloc]     = dd['muback']
                print('WARNING: Homogeneous center_label set to -1.0,-1.0')
                center_label[iloc] = -1.0,-1.0
                print('WARNING: Homogeneous radius_label set to -1.0')
                radius_label[iloc] = -1.0
                
        # get displacement data and strain data
        with open(outputname,'r') as fin:
            dd   = json.load(fin)
            sol  = np.asarray(dd['solution'])
            exx  = np.asarray(dd['exx'])
            eyy  = np.asarray(dd['eyy'])
            exy  = np.asarray(dd['exy'])
            
            # do not reshape.(nnodey,nnodex)
            solx = sol[:,0].reshape(nnodex,nnodey).T
            soly = sol[:,1].reshape(nnodex,nnodey).T
            images[iloc,:,:,0] = solx
            images[iloc,:,:,1] = soly

            # put into strain array
            strain[iloc,:,:,0]=exx.reshape(nnodex,nnodey).T
            strain[iloc,:,:,1]=eyy.reshape(nnodex,nnodey).T
            strain[iloc,:,:,2]=exy.reshape(nnodex,nnodey).T

        # get field
        with open(inputname,'r') as fin:
            dd   = json.load(fin)
            prop = np.asarray(dd['prop'])
            lam  = prop[:,0].reshape(nnodex,nnodey).T
            mu   = prop[:,1].reshape(nnodex,nnodey).T
            field_label[iloc,:,:,0] = lam
            field_label[iloc,:,:,1] = mu

        plotfield(xx,yy,images[iloc,:,:,0],'ux',outsolx,outputdir=prefix+str(ii))
        plotfield(xx,yy,images[iloc,:,:,1],'uy',outsoly,outputdir=prefix+str(ii))
        
        plotfield(xx,yy,field_label[iloc,:,:,0],'lam',outlam,outputdir=prefix+str(ii))
        plotfield(xx,yy,field_label[iloc,:,:,1],'mu',outmu,outputdir=prefix+str(ii))

        # plot strains
        plotfield(xx,yy,strain[iloc,:,:,0],'exx',outexx,outputdir=prefix+str(ii))
        plotfield(xx,yy,strain[iloc,:,:,1],'eyy',outeyy,outputdir=prefix+str(ii))
        plotfield(xx,yy,strain[iloc,:,:,2],'exy',outexy,outputdir=prefix+str(ii))

    listbin = list(binary_label)
    _h      = listbin.count(0)
    _nh     = listbin.count(1)
    
    print(f'Found {_nh} non-homogeneous and {_h} homogeneous examples out of a total {_nh+_h}')
    # strain = np.load('strain.npy')
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


def addnoise(data,noise,ntrain,nvalid,ntest,nnodex,nnodey):
    # be careful here: addnoise modifies np arrays in data and returns data
    # addnoise can be applied on test data before or after normalization
    # this is because the normalization factor is computed on the training data
    # if the normalization factor was being computed on the test_data, then
    # things would be different.
    
    # data : CNNData Tuple
    # noise: noise factor
    # plotbool: decide whether or not to plot data
    
    print('-'*80,f'\n Adding {noise} noise\n','-'*80,sep='')

    nstrain = data.strain.shape[-1]    # number of strain components: see select_input_comps
    nimages = data.images.shape[-1]    # number of displacement components: see select_input_comps
    
    nfactor = np.random.uniform(1-noise,1+noise)

    # add noise only to test data
    # breakpoint()
    for itest in range(ntrain+nvalid,ntrain+nvalid+ntest):

        for istrain in range(nstrain):
            noisemaker = np.random.uniform(1.0-noise,1.0+noise,size=(nnodey,nnodex))
            data.strain[itest,:,:,istrain] *= noisemaker

            if (itest == (ntrain+nvalid+ntest-1)):
                # print snr for last strain/disp image
                # noisemaker is not 0 anywhere because it should be in range (1-noise,1+noise) and (0<noise<1)
                # so we can divide by noisemaker without fear
                clean  = data.strain[-1,:,:,istrain] / noisemaker  
                noisy  = data.strain[-1,:,:,istrain] 
                noisepercen = np.linalg.norm(noisy - clean) / np.linalg.norm(noisy)
                if ( noisepercen != 0.0):
                    snrdb  = 20*np.log10(1.0/noisepercen)
                    print('snr in dB in strain ==',snrdb)
                else:
                    print('no noise in strain')

        for idisp in range(nimages):
            noisemaker = np.random.uniform(1.0-noise,1.0+noise,size=(nnodey,nnodex))
            data.images[itest,:,:,idisp] *= noisemaker
            if ( itest == (ntrain+nvalid+ntest-1)):
                # print snr for last strain/disp image                
                clean = data.images[-1,:,:,idisp] / noisemaker
                noisy = data.images[-1,:,:,idisp]
                noisepercen = np.linalg.norm(noisy - clean) / np.linalg.norm(noisy)

                print('norm noisy= ',np.linalg.norm(noisy))
                print('norm clean= ',np.linalg.norm(clean))
                
                if ( noisepercen != 0.0):
                    snrdb  = 20*np.log10(1.0/noisepercen)
                    print('snr in dB in disp ==',snrdb)

                else:
                    print('no noise in disp')


    newdata = data

    return newdata

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

    if (mltype == 'field'):
        return prediction


def forscale_linear(xmin,xmax,data):
    # data is np.array of shape (?,nnodey,nnodex)
    # data is modified in place
    data[:,:,:] -= xmin
    data[:,:,:] /= (xmax-xmin)

def invscale_linear(xmin,xmax,data):
    # data is of np.array of shape (?,nnodey,nnodex)
    data[:,:,:] *= (xmax-xmin)
    data[:,:,:] += xmin

def forscale_p1m1(xmin,xmax,data):
    data -= xmin
    data /= (xmax-xmin)
    data -= 0.5
    data *= 2.0
    
def invscale_p1m1(xmin,xmax,data):
    data /= 2.0
    data += 0.5
    data *=(xmax-xmin)
    data += xmin


'''
def normalize_input_cnndata_old(data,ntrain,nvalid,ntest):
    # should be called before select_input_comps
    # note: normalization parameters are computed over training images only
    
    uxmax  = np.max(np.abs(data.images[0:ntrain,:,:,0]))
    uxnorm = data.images[:,:,:,0]/uxmax
    
    uymax  = np.max(np.abs(data.images[0:ntrain,:,:,1]))
    uynorm = data.images[:,:,:,1]/uymax

    new_images = np.stack((uxnorm,uynorm),axis=-1)

    exxmax = np.max(np.abs(data.strain[0:ntrain,:,:,0]))
    eyymax = np.max(np.abs(data.strain[0:ntrain,:,:,1]))
    exymax = np.max(np.abs(data.strain[0:ntrain,:,:,2]))

    exxnorm = data.strain[:,:,:,0]/exxmax
    eyynorm = data.strain[:,:,:,1]/eyymax
    exynorm = data.strain[:,:,:,2]/exymax

    new_strain = np.stack((exxnorm,eyynorm,exynorm),axis=-1)

    cnndata_norm_input = CNNData(images=new_images,
                                 strain=new_strain,
                                 labels=data.labels
                                )
    
    return cnndata_norm_input

'''

'''
def normalize_input_cnndata_single(data,ntrain,nvalid,ntest):
    # single denotes that a single factor is used to normalize data
    # unlike normalize_input_data where each component of images and strain
    # is scaled differently
    # for incompressible elasticity exx+eyy ~ 0
    # and ux and uy have about the same magnitude
    # so normalize_input_cnndata effectively reduces to normalize_input_cnndata_single

    umax = np.max(np.abs(data.images[0:ntrain,:,:,:]))
    emax = np.max(np.abs(data.strain[0:ntrain,:,:,:]))
    
    # not implemented

'''

'''
def addnoise_old(data,noise,nnodex,nnodey):
    # be careful here: addnoise modifies np arrays in data and returns data
    # addnoise can be applied on test data before or after normalization
    # this is because the normalization factor is computed on the training data
    # if the normalization factor was being computed on the test_data, then
    # things would be different.
    
    # data : CNNData Tuple
    # noise: noise factor
    # plotbool: decide whether or not to plot data
    
    print('-'*80,f'\n Adding {noise} noise\n','-'*80,sep='')

    ntest   = data.strain.shape[0]
    nstrain = data.strain.shape[-1]    # number of strain components: see select_input_comps
    nimages = data.images.shape[-1]    # number of displacement components: see select_input_comps
    
    nfactor = np.random.uniform(1-noise,1+noise)
    # nmaker  = np.empty(cstrain.shape,dtype='float64')

    for itest in range(ntest):

        for istrain in range(nstrain):
            noisemaker = np.random.uniform(1.0-noise,1.0+noise,size=(nnodey,nnodex))
            data.strain[itest,:,:,istrain] *= noisemaker
            # nmaker[itest,:,:,istrain]       = noisemaker
            if (itest == (ntest-1)):
                # print snr for last strain/disp image
                clean  = data.strain[-1,:,:,-1] / noisemaker
                noisy  = data.strain[-1,:,:,-1] 
                noisepercen = np.linalg.norm(noisy - clean) / np.linalg.norm(noisy)
                if ( noisepercen != 0.0):
                    snrdb  = 20*np.log10(1.0/noisepercen)
                    print('snr in dB in strain ==',snrdb)
                else:
                    print('no noise in strain')

        for idisp in range(nimages):
            noisemaker = np.random.uniform(1.0-noise,1.0+noise,size=(nnodey,nnodex))
            data.images[itest,:,:,idisp] *= noisemaker
            if ( itest == (ntest-1)):
                # print snr for last strain/disp image                
                clean = data.images[-1,:,:,-1] / noisemaker
                noisy = data.images[-1,:,:,-1]
                noisepercen = np.linalg.norm(noisy - clean) / np.linalg.norm(noisy)
                if ( noisepercen != 0.0):
                    snrdb  = 20*np.log10(1.0/noisepercen)
                    print('snr in dB in disp ==',snrdb)

                else:
                    print('no noise in disp')

    newdata = data

    return newdata
'''
