from src.misc      import *
from src.datastrc  import *
from src.argparams import *
from src.data      import *
from src.cnn       import *
from src.plotting  import *


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
    epochs    = newparams['nepochs']
    prefix    = newparams['prefix']
    length    = newparams['length']
    breadth   = newparams['breadth']

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
    
    train_data_scaled,valid_data_scaled,test_data_scaled = forward_scale_all( datatuple=tt,
                                                                              length=length,
                                                                              breadth=breadth
                                                                             )

    
    cnn = load_or_train_and_plot_cnn( mltype=mltype,
                                      train_data=train_data_scaled,
                                      valid_data=valid_data_scaled,
                                      nnodex=nnodex,
                                      nnodey=nnodey,
                                      epochs=epochs
                                     )
    
    cnn.summary()
    
    prediction = predict_cnn( mltype=mltype,
                              cnn=cnn,
                              test_data=test_data_scaled
                             )
    
    prediction_inv = inverse_scale_prediction( mltype=mltype,
                                               prediction=prediction,
                                               length=length,
                                               breadth=breadth
                                              )
    save_prediction(mltype=mltype,
                    prediction=prediction_inv)
    
    postproc = post_process_cnn( mltype=mltype,
                                 ntrain=ntrain,
                                 nvalid=nvalid,
                                 ntest=ntest,
                                 prediction=prediction_inv,
                                 test_data=test_data
                                )
    goodbye()


