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

    # no feature scaling is being applied because all quantities are scaled nicely
    # for the default problem
    
    train_data,valid_data,test_data = get_data( ntrain=ntrain,
                                                nvalid=nvalid,
                                                ntest=ntest,
                                                nnodex=nnodex,
                                                nnodey=nnodey,
                                                prefix=prefix
                                               )

    cnn = load_or_train_and_plot_cnn( mltype=mltype,
                                      train_data=train_data,
                                      valid_data=valid_data,
                                      nnodex=nnodex,
                                      nnodey=nnodey,
                                      epochs=epochs
                                     )
    cnn.summary()
    prediction = predict_and_save_cnn( mltype=mltype,
                                       cnn=cnn,
                                       test_data=test_data
                                      )
    
    postproc = post_process_cnn( mltype=mltype,
                                 ntrain=ntrain,
                                 nvalid=nvalid,
                                 ntest=ntest,
                                 prediction=prediction,
                                 test_data=test_data
                                )
    goodbye()


