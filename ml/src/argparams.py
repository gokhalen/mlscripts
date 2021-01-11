import argparse
import json
from   .misc import message
from   .config import mltypelist

def get_args():
    parser = argparse.ArgumentParser(description=message)
    parser.add_argument('--inputfile',help='input file generated by mlsetup',
                        required=False,default='mlargs.json.out',type=str)
    parser.add_argument('--mltype',help='type of ml to do',required=False,default='binary',
                        choices=mltypelist)
    
    parser.add_argument('--ntrain',help='number of training examples to generate',
                        required=False,type=int)
    parser.add_argument('--nvalid',help='number of validation examples to generate',
                        required=False,type=int)
    parser.add_argument('--ntest', help='number of test examples to generate',
                        required=False,type=int)

    parser.add_argument('--nepochs', help='number of epochs',
                        required=False,default=12,type=int)

    parser.add_argument('--prefix', help='prefix of data directories',
                        required=False,type=str,default='traindata')

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

    newparams['mltype']  = args.mltype
    newparams['nepochs'] = args.nepochs
        
    return newparams
