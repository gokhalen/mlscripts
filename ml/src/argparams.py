import argparse
import json
import sys
import os

from .config import mltypelist

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
                        choices=['sgd','rmsprop','adam','adadelta','adagrad','adamax','nadam','ftrl']
                        )


    parser.add_argument('--activation',help='activation of output layer for mltype=field',
                        required=False,type=str,default='softplus',
                        choices=['softplus','sigmoid',
                                 'shift_square_both','shift_softplus_both','shift_sigmoid_both',
                                 'shift_softplus_lower','sigmoid_symmetric','tanh','twisted_tanh'
                                 ]
                        )

    
    # can either predict with a checkpointed model or continue training with the saved
    parser.add_argument('--mode',help='predict with checkpointed model or continue training saved model',
                        required=True,type=str,
                        choices=['checkpoint','train']
                        )

    parser.add_argument('--nimg',help='number of images to post process',
                        required=False,type=int,
                        default=32
                        )


    # a random number is generated between nn \in (-noise,+noise) and a
    # factor 1+nn is computed
    # the displacement is multiplied by this factor
    parser.add_argument('--noise',help='noise perturbation',
                        required=False,type=float,
                        default=0.00
                       )

    parser.add_argument('--noisetype',help='Additive or multiplicative noise',
                        required=True,type=str,
                        default='add',
                        choices=['add','mult']
                        )
                        


    parser.add_argument('--featurescale',help='scale or not scale mu field',
                        required=False,type=str,
                        default='False',
                        choices=['True','False']
                        )

    # choose whether to scale all channels in the image by a single constant
    # or scale each channel individually
    parser.add_argument('--inputscale',help='choose scaling for strain/displacement',
                        required=True,type=str,
                        choices=['global','individual']
                        )
    
    args = parser.parse_args()

    # assert ( 0.0 <= args.noise < 1.0 ), 'args.noise not in range [0,1), required for division by noisemaker to work'
    
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

    newparams['mltype']        = args.mltype     
    newparams['nepochs']       = args.nepochs    
    newparams['optimizer']     = args.optimizer  
    newparams['activation']    = args.activation 
    newparams['iptype']        = args.iptype     
    newparams['mode']          = args.mode       
    newparams['nimg']          = min(args.nimg,newparams['ntest'])
    newparams['noise']         = args.noise
    newparams['noisetype']     = args.noisetype
    newparams['featurescale']  = args.featurescale
    newparams['inputscale']    = args.inputscale

    if ( args.nimg > newparams['ntest']):
        print(f'{__file__}: nimg > ntest ...setting nimg to ntest')  # the setting is done a few lines above

    newparams['outputdir'] = args.mltype+'_'+args.iptype+f'_noise_{args.noise}_output'
    if ( args.outputdir != None):
        newparams['outputdir'] = args.outputdir

    if (args.featurescale=='True'):
        if (args.activation not in ['sigmoid','sigmoid_symmetric','tanh','twisted_tanh']):
            print(f'{__file__}:featurescale=True requires "sigmoid" or "sigmoid_symmetric" or "twisted_tanh" activation only. You specified: {args.activation}')
            sys.exit()


    if ( not os.path.exists(newparams['outputdir'])):
        os.mkdir(newparams['outputdir'])

    with open(f"{newparams['outputdir']}/config.out",'w') as fout:
        json.dump(newparams,fout)
        
    return newparams
