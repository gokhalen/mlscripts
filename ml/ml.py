from src.misc      import *
from src.datastrc  import *
from src.argparams import *
from src.data      import *
from src.cnn       import *


if __name__ =='__main__':
    welcome()
    args      = get_args()                             # get the args namespace
    params    = get_params(args.inputfile)             # get the params dictionary
    newparams = update_params(params=params,args=args) # override parameters in params with those in args
    
    train_data,valid_data,test_data = get_data(params=newparams)
    
    cnn         = define_cnn(params=newparams)
    history,cnn = train_cnn(cnn=cnn,train_data=train_data,valid_data=valid_data) 
    goodbye()


