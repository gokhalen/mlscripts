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

    ntrain    = newparams['ntrain']
    nvalid    = newparams['nvalid']
    ntest     = newparams['ntest']
    nnodex    = newparams['nelemx']+1;
    nnodey    = newparams['nelemy']+1;
    mltype    = newparams['mltype']
    iptype    = newparams['iptype']
    epochs    = newparams['nepochs']
    prefix    = newparams['prefix']
    length    = newparams['length']
    breadth   = newparams['breadth']
    optimizer = newparams['optimizer']

    # feature scaling is applied to use prior knowledge about the data
    # e.g. We know the max and min coordinates of the center
    # so we can rescale them to be in (0.0,1.0)
    
    train_data,valid_data,test_data = get_data( ntrain=ntrain,
                                                nvalid=nvalid,
                                                ntest=ntest,
                                                nnodex=nnodex,
                                                nnodey=nnodey,
                                                prefix=prefix
                                               )
    

    tt = (train_data,valid_data,test_data)

    valmin  = np.min(train_data.labels.value)
    valmax  = np.max(train_data.labels.value)
    valave  = np.mean(train_data.labels.value)

    # skip scaling for mu value 
    valmin = 0
    valmax = 1
    valave = 0

    

    # vf = forward_scale_value(train_data.labels.value,valmin=valmin,valmax=valmax,valave=valave)
    # vi = inverse_scale_value(vf,valmin=valmin,valmax=valmax,valave=valave)

    # print(np.linalg.norm(train_data.labels.value-vi))
    # sys.exit()
     
    train_data_scaled,valid_data_scaled,test_data_scaled = forward_scale_all( datatuple=tt,
                                                                              length=length,
                                                                              breadth=breadth,
                                                                              valmin=valmin,
                                                                              valmax=valmax,
                                                                              valave=valave
                                                                             )


    tt=(train_data_scaled,valid_data_scaled,test_data_scaled)


    cnn = load_or_train_and_plot_cnn( mltype=mltype,
                                      iptype=iptype,
                                      train_data=train_data_scaled,
                                      valid_data=valid_data_scaled,
                                      nnodex=nnodex,
                                      nnodey=nnodey,
                                      epochs=epochs,
                                      optimizer=optimizer
                                     )

    cnn.summary()
    
    prediction = predict_cnn( mltype=mltype,
                              iptype=iptype,
                              cnn=cnn,
                              test_data=test_data_scaled
                             )


    
    prediction_inv = inverse_scale_prediction( mltype=mltype,
                                               prediction=prediction,
                                               length=length,
                                               breadth=breadth,
                                               valmin=valmin,
                                               valmax=valmax,
                                               valave=valave
                                              )
    save_prediction(mltype=mltype,
                    iptype=iptype,
                    prediction=prediction_inv)


    
    postproc = post_process_cnn( mltype=mltype,
                                 iptype=iptype,
                                 ntrain=ntrain,
                                 nvalid=nvalid,
                                 ntest=ntest,
                                 prediction=prediction_inv,
                                 test_data=test_data
                                )

    goodbye()



    '''
    (ti,vi,tsi) = inverse_scale_all(datatuple=tt,
                                    length=length,
                                    breadth=breadth,
                                    valmin=valmin,
                                    valmax=valmax,
                                    valave=valave
                                    )

    print('train diff',np.linalg.norm(train_data.labels.binary-ti.labels.binary),
          np.linalg.norm(train_data.labels.center-ti.labels.center),
          np.linalg.norm(train_data.labels.radius-ti.labels.radius),
          np.linalg.norm(train_data.labels.value-ti.labels.value),
          )
    
    print('valid diff',np.linalg.norm(valid_data.labels.binary-vi.labels.binary),
          np.linalg.norm(valid_data.labels.center-vi.labels.center),
          np.linalg.norm(valid_data.labels.radius-vi.labels.radius),
          np.linalg.norm(valid_data.labels.value-vi.labels.value),
          )

    print('test diff',np.linalg.norm(test_data.labels.binary-tsi.labels.binary),
          np.linalg.norm(test_data.labels.center-tsi.labels.center),
          np.linalg.norm(test_data.labels.radius-tsi.labels.radius),
          np.linalg.norm(test_data.labels.value-tsi.labels.value),
          )
    '''



