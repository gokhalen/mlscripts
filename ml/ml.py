import os

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
    mltype    = newparams['mltype']

    # no feature scaling is being applied because all quantities are scaled nicely
    # for the default problem
    
    train_data,valid_data,test_data = get_data(params=newparams)


    if (os.path.exists(mltype)):
        print('-'*80,f'\n Old model for mltype={mltype} exists...loading old model\n','-'*80)
        cnn=tf.keras.models.load_model(mltype)
    else:
        cnn         = define_cnn(params=newparams)
    
        cnn,history = train_cnn(mltype=mltype,
                                cnn=cnn,
                                train_data=train_data,
                                valid_data=valid_data,
                                epochs=12
                                )
        plotall(mltype,history)

        # https://github.com/tensorflow/tensorflow/issues/44178 - Deprecation
        # warnings are nothing to worry about
        tf.keras.models.save_model(cnn,'binary',overwrite=True,include_optimizer=True)

    cnn.summary()
    prediction = predict_cnn(mltype=mltype,
                             cnn=cnn,
                             test_data=test_data
                             )

    postproc  = post_process_cnn(mltype=mltype,
                                 prediction=prediction,
                                 test_data=test_data
                                 )
    goodbye()


