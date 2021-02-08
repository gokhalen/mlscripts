import argparse
import os
import glob
import shutil
import json
import random
import sys
import numpy as np

from timerit import Timer
from collections import Counter

# the object which will generate data
from fypy.src.fypy    import FyPy
from fypy.src.libmesh import FyPyMesh

'''
Nachiket Gokhale gokhalen@gmail.com

This script generates training, validation and testing examples for EL ML.
This works as follows

1. Getting arguments: `getargs` is called to get arguments (the object `args`)
2. Example generation: if `args.generate` is True, all example are generated and written to disk.
   examples are generated using `generate_training_parameters(args)`
3. Solving: If `args.solve` is True, the examples are solved

The meat of this script is in `2. Example generation`.

We will now take a closer look at it. 

6. Example Generation:
   1. `args.ntotal` random examples are generated using `generate_random(args)`.
      `generate_random` generates random circular inclusions.
      `generate_random` basically creates a dictionary which FyPyMesh uses to create meshes
       labels and categories are set manually

   2. if `args.nhomo` > 0 then it determines the number of training examples are made homogeneous by setting `stf` to `homogeneous`
      if `args.nhomo` > 0 then an equivalent percentage of validation and test examples are made homogeneous

       We need to make homogeneous examples explicitly because `generate_random` does not do that for us.

   3. We count number of positive and negative training examples
'''

def getargs():
    parser = argparse.ArgumentParser(description='driver script to generate data for Elasticity Imaging Machine Learning (EIML)')
    
    parser.add_argument('--prefix',help='prefix for training example directories',required=False,default='traindata',type=str)

    parser.add_argument('--generate',help='generate input files for training examples',required=False,default='False')
    parser.add_argument('--solve',help='solve training examples',required=False,default='False')

    parser.add_argument('--rmin',help='lower bound on inclusion radius',required=False,type=float,default=0.05)
    parser.add_argument('--rmax',help='lower bound on inclusion radius',required=False,type=float,default=0.15)
    parser.add_argument('--nelemx',help='number of elements in the x-direction',required=False,type=int,default=16)
    parser.add_argument('--nelemy',help='number of elements in the y-direction',required=False,type=int,default=16)
    parser.add_argument('--length',help='length of the domain in x direction',required=False,type=float,default=1.0)
    parser.add_argument('--breadth',help='length of the domain in y direction',required=False,type=float,default=1.0)

    parser.add_argument('--muback',help='background mu',required=False,type=float,default=1.0)
    parser.add_argument('--mumin',help='lower bound on mu in inclusion',required=False,type=float,default=2.0)
    parser.add_argument('--mumax',help='upper bound on mu in inclusion',required=False,type=float,default=5.0)
    parser.add_argument('--nu',help="Poisson's ratio",required=False,type=float,default=0.49)
    parser.add_argument('--eltype',help="Element type",required=False,type=str,default='linelas2dnumbasri',choices=['linelas2d','linelas2dnumba','linelas2dnumbasri'])

    # see below, where we check problemtype, for constraints on ntrain  
    parser.add_argument('--ntrain',help='number of training   examples to generate',required=False,type=int,default=16)
    parser.add_argument('--nvalid',help='number of validation examples to generate',required=False,type=int)
    parser.add_argument('--ntest', help='number of test examples to generate',      required=False,type=int)
    parser.add_argument('--nhomo',help='number of homogeneous examples to generate',required=False,type=int,default=4)
    parser.add_argument('--shift',help='shift for numbering',required=False,type=int,default=0)
    
    args = parser.parse_args()

    assert ( 0<= args.nhomo <= args.ntrain),'args.nhomo must lie between 0 and args.ntrain (both inclusive)'


    # for now, we're skipping creating nvalid and ntest and letting the actual ML program handle splitting into train,valid and test 
    if (args.nvalid == None): args.nvalid = int(0.2*args.ntrain); 
    if (args.ntest  == None): args.ntest  = int(0.2*args.ntrain);

    if ( args.nhomo > 0 ):
        args.nhomovalid = int((args.nhomo/args.ntrain)*(args.nvalid)) + 1
        args.nhomotest  = int((args.nhomo/args.ntrain)*(args.ntest))  + 1

    else:
        args.nhomovalid = 0
        args.nhomotest  = 0

    args.ntotal = args.ntrain + args.nvalid + args.ntest
    args.nlabel =  2

    # check that the inclusion specified, is big enough to include atleast 2 nodes in the x and y direction
    # number of nodes in rmin in x and y direction
    rmin,rmax = get_rmin_rmax(args)
    nrminx   = rmin*(args.nelemx/args.length)
    nrminy   = rmin*(args.nelemy/args.breadth)

    #  print('node capture assertion turned off')
    assert ( int(min(nrminx,nrminy)) > 1 ),'Not enough nodes captured in inclusion specified. Increase nelemx,nelemy,rmin,rmax as appropriate'

    # dump the true min and max radii - this info is not needed for fypymesh.create_mesh_2d so we can't put it into args
    radir = {}
    truermin = rmin
    truermax = rmax
    radir['truermin']=rmin
    radir['truermax']=rmax

    argsdir = vars(args)
    argsdir.update(radir)
    
    # save parameters passed to mlsetup
    with open('mlargs.json.out','w') as fout:
        json.dump(argsdir,fout,indent=4)

    return args

def get_rmin_rmax(args):
    refl  = min(args.length,args.breadth)  # reference length
    rmin  = refl*args.rmin
    rmax  = refl*args.rmax
    return (rmin,rmax)

class FyPyArgs():
    def __init__(self,inputfile,outputfile,inputdir,outputdir,partype,profile,solvertype):
        self.inputfile  = inputfile
        self.outputfile = outputfile
        self.inputdir   = inputdir
        self.outputdir  = outputdir
        self.partype    = partype
        self.profile    = profile
        self.solvertype = solvertype
        self.nprocs     = 1
        self.chunksize  = 1

# will return list of dictionaries of parameter lists
def generate_training_parameters(args):
    outlist = []                                # list of arguments to pass to mesh creator
    labels  = []                                # list of labels. This is a list of lists (artifact of when the
                                                # script used to do multiclass classification)
                                                
    counts  = [0]*args.nlabel                   # only two types of labels
    category = [None]*args.ntotal
    
    for iexample in range(0,args.ntotal):
        dd = generate_random(args)
        outlist.append(dd)
        labels.append([1])
        category[iexample] = 1

    if ( args.nhomo > 0): 
        # make homogeneous training examples
        istart = 0; iend = args.nhomo
        make_homogeneous(istart,iend,outlist,labels,category,'training')
    
        # make homogeneous validation examples
        istart = args.ntrain; iend = args.ntrain + args.nhomovalid
        make_homogeneous(istart,iend,outlist,labels,category,'validation')    

        # make homogneous test examples
        istart = args.ntrain+args.nvalid; iend = args.ntrain + args.nvalid + args.nhomotest
        make_homogeneous(istart,iend,outlist,labels,category,'test')
    
    # making and counting (how many positive and how many negative) labels
    # count only in the training examples
    llcount   = labels[:args.ntrain]
    counts[0] = llcount.count([0])
    counts[1] = llcount.count([1])

    print('-'*80)
    print(f'Positive (tumor)       training examples {counts[1]} {(counts[1]/args.ntrain)*100:0.2f}%')
    print(f'Negative (homogeneous) training examples {counts[0]} {(counts[0]/args.ntrain)*100:0.2f}%')
    print(f'Total examples {args.ntotal}, Training examples {args.ntrain}, Validation examples {args.nvalid}, Test Examples {args.ntest} ')
    print('-'*80)
    
    return outlist,labels,counts,category


def make_homogeneous(istart,iend,outlist,labels,category,datatype):
    print(f'Making {datatype} examples {istart} to {iend-1} (both inclusive) homogeneous')
    for ihomo in range(istart,iend):
        outlist[ihomo]['stftype']='homogeneous'
        labels[ihomo]   = [0]
        category[ihomo] =  0

def generate_random(args):
    # generates a random dictionary of parameters
    dd    = {}
    
    dd['length']  = args.length
    dd['breadth'] = args.breadth
    dd['nelemx']  = args.nelemx
    dd['nelemy']  = args.nelemy
    dd['stftype'] = 'inclusion'
    dd['bctype']  = 'trac'
    dd['bcmag']   = -0.06
    rmin,rmax     = get_rmin_rmax(args)
    dd['radii']   =  [np.random.uniform(rmin,rmax)]

    # make sure the center of the inclusion is mostly inside the domain
    xcen          = np.random.uniform(0.0+0.05*args.length,args.length*0.95)
    ycen          = np.random.uniform(0.0+0.05*args.breadth,args.breadth*0.95)
    dd['centers'] = [[xcen,ycen]]
    dd['mu']      = np.random.uniform(args.mumin,args.mumax)
    dd['muback']  = args.muback
    dd['nu']      = args.nu
    dd['eltype']  = args.eltype

    return dd

def make_file_names(args,idx):
    dirname     = f'{args.prefix}' + str(idx)+'/'
    inputname   = f'input'  + str(idx) + '.json.in'
    outputname  = f'output' + str(idx) + '.json.out'
    mlinfoname  = f'{dirname}'+f'mlinfo'  + str(idx) + '.json.in'
    
    return dirname,inputname,outputname,mlinfoname
 
if __name__ == '__main__':
    
    args = getargs()

    if ( args.generate == 'True'):
        outlist,labels,counts,category = generate_training_parameters(args)

    if (args.generate == 'True'):
        for iexample,argdict in enumerate(outlist):
            dirname,inputname,outputname,mlinfoname = make_file_names(args,iexample+args.shift)
            
            print(f'Creating training inputfiles for example {iexample+1} of {args.ntotal}')
            mesh2d = FyPyMesh(inputdir=dirname,outputdir=dirname)
            os.mkdir(dirname)
            mesh2d.create_mesh_2d(**argdict)
            mesh2d.json_dump(filename=inputname)
            mesh2d.preprocess(str(iexample+args.shift))
            # dump label
            mlinfodir = {}
            mlinfodir['label']    = labels[iexample]
            mlinfodir['category'] = category[iexample]
            mlinfodir.update(argdict)

            # some checks
            for xcen,ycen in argdict['centers']:
                if ( xcen >= args.length):
                    print(f'xcen out of range in {iexample=}')

                if ( ycen >= args.breadth):
                    print(f'ycen out of range in {iexample=}')

            # dump machine learning info
            with open(mlinfoname,'x') as fout:
                json.dump(mlinfodir,fout,indent=4)

    if (args.solve == 'True'):
        for iexample in range(args.ntotal):
             dirname,inputname,outputname,mlinfoname = make_file_names(args,iexample+args.shift)
             print(f'Solving training example {iexample+1} of {args.ntotal}')
             tsolve = Timer('Solve timer',verbose=0)
             with tsolve:
                 fypyargs = FyPyArgs(
                        inputfile  = inputname,
                        outputfile = outputname,
                        inputdir   = dirname,
                        outputdir  = dirname,
                        partype    = 'list',
                        profile    = 'False',
                        solvertype = 'spsolve'
                    )
                 fypy = FyPy(fypyargs)
                 fypy.doeverything(str(iexample+args.shift))
             print(f'Solved example {iexample+1} of {args.ntotal} in {tsolve.elapsed:.2f}s')
             
                                   

