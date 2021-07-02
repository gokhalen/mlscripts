import logging
import os
import copy

# Silence tf warnings and deprecation mesages
# Must be before tf is imported
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/38645250#38645250
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from src.misc      import *
from src.datastrc  import *
from src.argparams import *
from src.data      import *
from src.cnn       import *
from src.plotting  import *

import numpy as np
import sys

if __name__ =='__main__':
    
    welcome()
    
    args      = get_args()                             # get the args namespace
    params    = get_params(args.inputfile)             # get the params dictionary
    newparams = update_params(params=params,args=args) # override parameters in params with those in args

    # should be able to put these variables into the local variable dictionary programmatically
    ntrain       = newparams['ntrain']
    nvalid       = newparams['nvalid']
    ntest        = newparams['ntest']
    nnodex       = newparams['nelemx']+1;
    nnodey       = newparams['nelemy']+1;
    mltype       = newparams['mltype']
    iptype       = newparams['iptype']
    epochs       = newparams['nepochs']
    prefix       = newparams['prefix']
    length       = newparams['length']
    breadth      = newparams['breadth']
    optimizer    = newparams['optimizer']
    activation   = newparams['activation']
    mode         = newparams['mode']
    outputdir    = newparams['outputdir']
    nimg         = newparams['nimg']
    noise        = newparams['noise']
    mubndmin     = newparams['muback']
    mubndmax     = newparams['mumax']
    featurescale = newparams['featurescale']
    inputscale   = newparams['inputscale'] 


    if ( not os.path.exists(outputdir)):
        os.mkdir(outputdir)

    # this data is not scaled
    train_data,valid_data,test_data = get_data( ntrain=ntrain,
                                                nvalid=nvalid,
                                                ntest=ntest,
                                                nnodex=nnodex,
                                                nnodey=nnodey,
                                                inputscale=inputscale,
                                                prefix=prefix,
                                                outputdir=outputdir,
                                                iptype=iptype
                                               )


    # be careful here: addnoise modifies np arrays in test_data
    test_data  = addnoise(test_data,noise,nnodex,nnodey)

    if ( featurescale == 'True'):
        # labels.field is modified in place
        # images and strains are normalized in get_data()
        forscale_p1m1(xmin=mubndmin,xmax=mubndmax,data=train_data.labels.field[:,:,:,1])
        forscale_p1m1(xmin=mubndmin,xmax=mubndmax,data=valid_data.labels.field[:,:,:,1])
        forscale_p1m1(xmin=mubndmin,xmax=mubndmax,data=test_data.labels.field[:,:,:,1])
    
    train_data_scaled = train_data
    valid_data_scaled = valid_data
    test_data_scaled  = test_data

    cnn = load_or_train_and_plot_cnn( mltype=mltype,
                                      iptype=iptype,
                                      train_data=train_data_scaled,
                                      valid_data=valid_data_scaled,
                                      nnodex=nnodex,
                                      nnodey=nnodey,
                                      mubndmin=mubndmin,
                                      mubndmax=mubndmax,
                                      epochs=epochs,
                                      activation=activation,
                                      optimizer=optimizer,
                                      mode=mode,
                                      outputdir=outputdir
                                     )

    cnn_summary(cnn=cnn,mltype=mltype,iptype=iptype,noise=noise,outputdir=outputdir)
    
    prediction_inv = predict_cnn( mltype=mltype,
                                  iptype=iptype,
                                  cnn=cnn,
                                  test_data=test_data_scaled,
                                  nnodex=nnodex,
                                  nnodey=nnodey
                                 )

            
    if ( featurescale == 'True'):
        invscale_p1m1(xmin=mubndmin,xmax=mubndmax,data=prediction_inv)
        invscale_p1m1(xmin=mubndmin,xmax=mubndmax,data=test_data_scaled.labels.field[:,:,:,1])

    save_prediction_test_data( mltype=mltype,
                               iptype=iptype,
                               noise=str(noise),
                               prediction=prediction_inv,
                               test_data=test_data,
                               outputdir=outputdir
                              )

    postproc = post_process_cnn( mltype=mltype,
                                 iptype=iptype,
                                 noise=noise,
                                 ntrain=ntrain,
                                 nvalid=nvalid,
                                 ntest=ntest,
                                 prediction=prediction_inv,
                                 test_data=test_data_scaled,
                                 outputdir=outputdir,
                                 nnodex=nnodex,
                                 nnodey=nnodey,
                                 nimg=nimg
                               )

    goodbye()






