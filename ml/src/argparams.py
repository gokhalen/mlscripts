import argparse
import json

from   .config import mltypelist

def get_args():
    parser = argparse.ArgumentParser(description='Identification of inclusion parameters using ML\
 Nachiket Gokhale gokhalen@gmail.com')
    parser.add_argument('--inputfile',help='input file generated by mlsetup',
                        required=False,default='mlargs.json.out',type=str)

    parser.add_argument('--outputdir',help='output directory',
                        required=False,type=str)    
    
    parser.add_argument('--mltype',help='type of ml to do',required=True,
                        choices=mltypelist)
    
    # images  - used both components of displacement
    # imagesx - uses  x component only
    # imagesy - uses  y component only
    # strain  - uses  all three components of strain
    # strainxx - uses e_xx only
    # strainyy - uses e_yy only
    # strainxxyy - uses e_xx and e_yy
    
    parser.add_argument('--iptype',help='input data to use',required=True,
                        choices=['images','imagesx','imagesy','strain','strainxx','strainyy','strainxxyy']
                        )
    
    parser.add_argument('--ntrain',help='number of training examples to generate',
                        required=False,type=int)
    parser.add_argument('--nvalid',help='number of validation examples to generate',
                        required=False,type=int)
    parser.add_argument('--ntest', help='number of test examples to generate',
                        required=False,type=int)

    parser.add_argument('--nepochs', help='number of epochs',
                        required=False,default=4,type=int)

    parser.add_argument('--prefix', help='prefix of data directories',
                        required=False,type=str,default='traindata')

    # optimizer
    parser.add_argument('--optimizer', help='name of optimizer',
                        required=False,type=str,default='adam',
                        choices=['sgd','rmsprop','adam','adadelta','adagrad','adamax','nadam','ftrl'])

    
    # can either predict with a checkpointed model or continue training with the saved
    parser.add_argument('--mode',help='predict with checkpointed model or continue training saved model',
                        required=True,type=str,
                        choices=['checkpoint','train']
                        )

    parser.add_argument('--nimg',help='number of images to post process',
                        required=False,type=int,
                        default=32
                        )

    

    args = parser.parse_args()
    return args

def get_params(fname):
    # loads config file created by mlsetup reads variables and returns them as a dictionary
    with open(fname,'r') as fin:
        jj=json.load(fin)
    return jj

def update_params(params,args):
    # returns an upadated parameter dictionary. parameters in params are overridden by the ones in args
    # if parameter values are specified as arguments on the command line then the
    # appropriate values in params are replaced

    # we can make a more elegant solution later.
    # based on updating dictionaries if values are not none
    newparams = params.copy()
    
    if (args.ntrain != None): newparams['ntrain'] = args.ntrain
    if (args.nvalid != None): newparams['nvalid'] = args.nvalid
    if (args.ntest  != None): newparams['ntest']  = args.ntest
    if (args.prefix != None): newparams['prefix'] = args.prefix

    if (args.ntrain != None) or (args.nvalid !=None) or (args.ntest != None):
        newparams['ntotal'] = newparams['ntrain'] + newparams['nvalid'] + newparams['ntest']

    newparams['mltype']    = args.mltype
    newparams['nepochs']   = args.nepochs
    newparams['optimizer'] = args.optimizer
    newparams['iptype']    = args.iptype
    newparams['mode']      = args.mode
    newparams['nimg']      = min(args.nimg,newparams['ntest'])
    
    if ( args.nimg > newparams['ntest']):
        print(f'{__file__}: nimg > ntest ...setting nimg to ntest')

    newparams['outputdir'] = args.mltype+'_'+args.iptype+'_output'
    if ( args.outputdir != None):
        newparams['outputdir'] = args.outputdir
        
    return newparams
