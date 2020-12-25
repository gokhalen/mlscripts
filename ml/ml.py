import tensorflow as tf
import numpy as np
import sys,os,glob
import json
import argparse

message = 'Identification of inclusion parameters using ML\
 Nachiket Gokhale gokhalen@gmail.com'

def get_args():
    parser = argparse.ArgumentParser(description=message)
    parser.add_argument('--inputfile',help='input file generated by mlsetup',
                        required=False,default='mlargs.json.out',type=str)
    parser.add_argument('--mltype',help='type of ml to do',required=False,default='binary',
                        choices=['binary','location','radius','value'])
    parser.add_argument('--ntrain',help='number of training   examples to generate',
                        required=False,type=int,default=16)
    parser.add_argument('--nvalid',help='number of validation examples to generate',
                        required=False,type=int,default=16)
    parser.add_argument('--ntest', help='number of test examples to generate',
                        required=False,type=int,default=16)

    args = parser.parse_args()
    return args

def get_params(fname):
    # loads config file created by mlsetup reads variables and returns them as a tuple
    with open(fname,'r') as fin:
        jj=json.load(fin)
    return jj

def get_data(prefix,nelemx,nelemy,ntrain,nvalid,ntest):
    # reads training,validation and test data and returns it
    # prefix is the prefix of the directories which contain training validation and test data
    # ntrain,nvalid,ntest are integers and should sum up to less than the number of files
    # returned by glob
    # train_data is a namedtuple with four components image,label,center,value,radius
    
    train_data,valid_data,test_data = None,None,None
    gg = glob.glob(f'{prefix}*')  # assume that there are no files straring with prefix, only directories
    ntotal = len(gg)
    assert (ntotal >= (ntrain+nvalid+ntest)),'Number of examples not sufficient'
    for ii, dirname in enumerate(gg):
        print(dirname,ii)
    return train_data,valid_data,test_data

def forward_scale_data():
    pass

def inverse_scale_data():
    pass

def define_cnn():
    ndim = 2 # number of displacement components.
    cnn = None
    return cnn


def welcome():
    print('-'*85)
    print(message)
    print('-'*85)

def goodbye():
    print('-'*80)
    print('Exiting gracefully ... goodbye!')
    print('-'*80)

if __name__ =='__main__':
    welcome()
    args   = get_args()
    params = get_params(args.inputfile)
    train_data,valid_data,test_data = get_data(params['prefix'],params['nelemx'],params['nelemy'],
                                               args.ntrain,args.nvalid,args.ntest)
    cnn    = define_cnn()
    goodbye()

