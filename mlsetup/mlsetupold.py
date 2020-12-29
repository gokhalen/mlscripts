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

This script generates training, validation and testing examples for binary classification.
This works as follows

1. Getting arguments: `getargs` is called to get arguments (the object `args`)
2. Example generation: if `args.generate` is True, all examples (depending on args.problemtype) are generated and written to disk
3. Solving: If `args.solve` is True, the examples are solved

The meat of this script is in `2. Example generation`.

4. Binary Classification: examples are generated using `generate_binary_training_parameters(args)`
5. Multiclass classification: examples are generated using `generate_multiclass_training_parameters(args)`

We will now take a closer look at these. 

6. Binary Classification:
   1. `args.ntotal` random examples are generated using `generate_random(args)`.
      `generate_random` generates random circular inclusions.
      `generate_random` basically creates a dictionary which FyPyMesh uses to create meshes
       labels and categories are set manually

   2. `args.nhomo` examples are made homogeneous by setting `stf` to `homogeneous`
       We need to make homogeneous examples explicitly because `generate_random` does not do that for us.

   3. We count number of positive and negative training examples

7. Multiclass classifcation:
 
   1. We divide the domain into `args.nclassx` cells in the x direction and `args.nclassy` 
      cells in the y direction, calculate their centers. These cells are our classes. 
      We aim to classify a tumor into one of these cells (or homogeneous), depending
      on where its center is located.  We create labels for each class, stored in labellist

   2. We generate all examples.
          1. `generate_random_multiclass`: this is `generate_random` plus some code 
              to generate homogeneous examples as well

          2. we make label and categorize example using `make_label`
              
          
  3. Overwrite:
          1. Overwrite the first args.nminexamp to make sure that each class contains 
             atleast args.nminexamp examples

  4. Make labels and count and categorize


'''

def getargs():
    parser = argparse.ArgumentParser(description='driver script to generate data for Elasticity Imaging Machine Learning (EIML)')
    
    parser.add_argument('--prefix',help='prefix for training example directories',required=False,default='traindata',type=str)

    parser.add_argument('--problemtype',help='binary classification or multiclass classification',required=True,choices=['binary','multiclass'])
    parser.add_argument('--generate',help='generate input files for training examples',required=False,default='False')
    parser.add_argument('--solve',help='solve training examples',required=False,default='False')


    parser.add_argument('--ninc',help='number of inclusions',required=False,type=int,default=1)
    parser.add_argument('--rmin',help='lower bound on inclusion radius',required=False,type=float,default=0.05)
    parser.add_argument('--rmax',help='lower bound on inclusion radius',required=False,type=float,default=0.15)
    parser.add_argument('--nelemx',help='number of elements in the x-direction',required=False,type=int,default=16)
    parser.add_argument('--nelemy',help='number of elements in the y-direction',required=False,type=int,default=16)
    parser.add_argument('--length',help='length of the domain in x direction',required=False,type=float,default=1.0)
    parser.add_argument('--breadth',help='length of the domain in y direction',required=False,type=float,default=1.0)
    
    # total number of classes is nclass=nclassx*nclassy + 1 (for homogeneous)
    parser.add_argument('--nclassx',help='number of classes to classify into (in x direction)',required=False,type=int,default=1)
    parser.add_argument('--nclassy',help='number of classes to classify into (in y direction)',required=False,type=int,default=1)
    parser.add_argument('--stfmin',help='lower bound on mu',required=False,type=float,default=1.0)
    parser.add_argument('--stfmax',help='upper bound on mu',required=False,type=float,default=5.0)
    parser.add_argument('--nu',help="Poisson's ratio",required=False,type=float,default=0.25)

    # see below, where we check problemtype, for constraints on ntrain  
    parser.add_argument('--ntrain',help='number of training   examples to generate',required=False,type=int,default=16)
    parser.add_argument('--nvalid',help='number of validation examples to generate',required=False,type=int)
    parser.add_argument('--ntest', help='number of test examples to generate',      required=False,type=int)
    parser.add_argument('--nhomo',help='number of homogeneous examples to generate',required=False,type=int,default=4)
    parser.add_argument('--nclassmin',help='minimum number of examples in each class',required=False,type=int,default=2)
    parser.add_argument('--shift',help='shift for numbering',required=False,type=int,default=0)
    
    # system arguments
    parser.add_argument('--clean',help='delete previous data',required=False,type=str,default='False')
    
    args = parser.parse_args()

    # for now, we're skipping creating nvalid and ntest and letting the actual ML program handle splitting into train,valid and test 
    if (args.nvalid == None): args.nvalid = int(0.0*args.ntrain); 
    if (args.ntest  == None): args.ntest  = int(0.0*args.ntrain); 

    args.ntotal  = args.ntrain + args.nvalid + args.ntest

    if ( args.problemtype == 'binary'):
        assert(args.ntrain > args.nhomo),f'Number of training examples {args.ntrain} must exceed number of homogeneous examples {args.nhomo}'
        args.nlabel =  2
    if (args.problemtype =='multiclass'):
        args.nlabel    = args.nclassx*args.nclassy + 1
        args.nminexamp = (args.nlabel)*args.nclassmin 
        assert( args.ntrain > args.nminexamp ),f'Number of training examples {args.ntrain} must exceed {args.nminexamp}, computed on basis of minimum number of examples per class'


    # check that the inclusion specified, is big enough to include atleast 2 nodes in the x and y direction
    # number of nodes in rmin in x and y direction
    rmin,rmax = get_rmin_rmax(args)
    nrminx   = rmin*(args.nelemx/args.length)
    nrminy   = rmin*(args.nelemy/args.breadth)

    assert ( int(min(nrminx,nrminy)) > 1 ),'Not enough nodes captured in inclusion specified. Increase nelemx,nelemy,rmin,rmax as appropriate'


    # save parameters passed to mlsetup
    with open('mlargs.json.out','w') as fout:
        json.dump(vars(args),fout,indent=4)

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
def generate_binary_training_parameters(args):
    outlist = []                                # list of arguments to pass to mesh creator
    labels  = []                                # list of labels. This is a list of lists to
                                                # keep it consistent with multiclass classification
    counts  = [0]*args.nlabel                   # only two types of labels
    category = [None]*args.ntotal
    
    for iexample in range(0,args.ntotal):
        dd = generate_random(args)
        outlist.append(dd)
        labels.append([1])
        category[iexample] = 1
        
    # make homogeneous examples
    for ihomo in range(0,args.nhomo):
        outlist[ihomo]['stftype']='homogeneous'
        labels[ihomo] = [0]
        category[ihomo] = 0
        
    # making and counting (how many positive and how many negative) labels
    # count only in the training examples
    llcount   = labels[:args.ntrain]
    counts[0] = llcount.count([0])
    counts[1] = llcount.count([1])

    print('-'*80)
    print(f'Binary classificaton: Positive (tumor)       examples {counts[1]} {(counts[1]/args.ntrain)*100:0.2f}%')
    print(f'Binary classificaton: Negative (homogeneous) examples {counts[0]} {(counts[0]/args.ntrain)*100:0.2f}%')
    print(f'Binary classificaton: Total examples {args.ntotal}, Training examples {args.ntrain}, Validation examples {args.nvalid}, Test Examples {args.ntest} ')
    print('-'*80)
    
    return outlist,labels,counts,category

# will return list of dictionaries of parameter lists
def generate_multiclass_training_parameters(args):
    # nhomo homogeneous examples + nclass classification examples + ntrain random examples

    print(f'Generating {args.ntotal} training examples,{args.nminexamp}',
          f'compulsary/minimum training examples {args.ntrain-args.nminexamp}',
          f'random training examples',
          f'{args.nvalid} validation examples',
          f'{args.ntest} testing examples')
    
    outlist  = []                   # list of arguments to pass to mesh creator
    labels   = [None]*args.ntotal   # list of labels (list of lists of size nlabel)
    centers  = []                   # list of centers used to determine labels.
    category = [None]*args.ntotal   # numerical value of the cateory for each label
    counts   = [0]*args.nlabel      # counts for each label

    labellist = [None]*args.nlabel  # store each label - used for counting

    # A typical label is [ 0 1 0 0 ...0] of length nlabel
    # having 1 in the class where the stiffness belongs to and zeros elsewhere

    dx  = args.length  / args.nclassx
    dy  = args.breadth / args.nclassy

    # create the label for the homogeneous case
    ll = [0]*args.nlabel;  ll[0] = 1
    labellist[0] = ll
    
    # initialize centers and labels
    for iclassx in range(args.nclassx):
        for iclassy in range(args.nclassy):
            idx  = iclassx*args.nclassy + iclassy
            xcen = (dx/2.0) + dx*iclassx
            ycen = (dy/2.0) + dy*iclassy
            centers.append([xcen,ycen])

            # create label for this class
            ll = [0]*args.nlabel ; ll[idx+1]=1
            labellist[idx+1] = ll

    # generate arguments for the mesher and label 'all' examples
    for iexample in range(0,args.ntotal):
        dd = generate_random_multiclass(args)
        outlist.append(dd)
        # create dummy labels for now
        # labels.append([0]*args.ntrain)
        ll,cat = make_label(args,centers,dd)
        labels[iexample]=ll
        category[iexample]=cat

    # overwrite the first args.nminexamp with homogeneous and labelled examples
    # this is done so that we have minimum number of examples from each class

    # generate the homogeneous examples
    # by changing the arguments ('inclusion'->'homogeneous') for the mesher
    for ihomo in range(0,args.nclassmin):
        outlist[ihomo]['stftype']='homogeneous'
        ll,cat   = make_label(args,centers,outlist[ihomo])
        labels[ihomo] = ll
        category[ihomo]=cat
        
    # make sure that we have training examples for each class
    # each cell in the classification grid must have an inclusion
    # generate minimum training examples 'nclassmin' for each class
    for iclass in range(args.nclassmin):
        for iclassx in range(args.nclassx):
            for iclassy in range(args.nclassy):
                idx   = iclassx*args.nclassy + iclassy + args.nclassmin + iclass*(args.nclassx*args.nclassy)
                idcen = iclassx*args.nclassy + iclassy
                # what happends if the tumor is in more than 1 training cell
                # one thing we can do, is change the label vector to be 1 in all the cells which have the tumor
                # but let's leave that for now
                xcen,ycen = centers[idcen]
                if ( iclass == 0):
                    # first example at the center of each cell of the training grid
                    rad  = min(dx/2.0,dy/2.0)*0.6
                else:
                    # second example at a random point in each cell of the training grid
                    xmin = dx*iclassx; xmax = (dx*iclassx) + dx
                    ymin = dy*iclassy; ymax = (dy*iclassy) + dy
                    xcen = np.random.uniform(xmin,xmax)
                    ycen = np.random.uniform(ymin,ymax)
                    rad  = min(dx/2.0,dy/2.0)*0.6*np.random.uniform(0.6,1.0)

                if ( xcen >= args.length):
                    print(f'xcen out of range in {idx=}')
                    breakpoint()

                if ( ycen >= args.breadth):
                    print(f'ycen out of range in {idx=}')
                    breakpoint()

                outlist[idx]['radius'] = rad
                outlist[idx]['xcen']   = xcen
                outlist[idx]['ycen']   = ycen
                outlist[idx]['stftype']    = 'inclusion'
                ll,cat = make_label(args,centers,outlist[idx])
                labels[idx] = ll
                category[idx] = cat

    # make sure there is atleast 1 example for each class
    # making and counting labels of each type

    llcount = labels[:args.ntrain]
    for ilabel in range(args.nlabel):
        counts[ilabel] = llcount.count(labellist[ilabel])
        print(f'Multiclass classification: Class {ilabel+1} has {counts[ilabel]} training examples {(counts[ilabel]/args.ntrain)*100:0.2f}%')

    return outlist,labels,counts,category

def make_label(args,centers,dd):
    if (dd['stftype'] == 'homogeneous'):
        # homogneous case
        label = [0]*args.nlabel
        label[0] = 1
        category = 0
    else:
        # inclusion case
        xx = dd['xcen']
        yy = dd['ycen']
        maxlength = (args.length**2.0 + args.breadth**2.0)**0.5
        closest   = -10
        for icen,(cenx,ceny) in enumerate(centers):
            dist = ((xx-cenx)**2.0 + (yy-ceny)**2.0)**0.5
            if (dist <= maxlength):
                closest   = icen
                maxlength = dist

        label = [0]*args.nlabel
        # if 'closest' is 0 , the 1st place in 'label' is 1
        # if 'closest' is 1 , the 2nd place in 'label' is 1
        # and so on
        label[closest+1] = 1
        category         = closest+1
        
    # return label vector and label id
    return label,category

def generate_random(args):
    # generates a random dictionary of parameters
    dd    = {} 
    dd['length']  = args.length
    dd['breadth'] = args.breadth
    dd['nelemx']  = args.nelemx
    dd['nelemy']  = args.nelemy
    dd['stftype'] = 'inclusion'
    dd['bctype']  = 'trac'
    rmin,rmax     = get_rmin_rmax(args)
    dd['radii']   =  [np.random.uniform(rmin,rmax)]

    # make sure the center of the inclusion is mostly inside the domain
    xcen          = np.random.uniform(0.0+0.05*args.length,args.length*0.95)
    ycen          = np.random.uniform(0.0+0.05*args.breadth,args.breadth*0.95)
    dd['centers'] = [[xcen,ycen]]
    dd['mumin']   = args.stfmin
    dd['mumax']   = args.stfmax
    dd['nu']      = args.nu
    dd['nclassx'] = args.nclassx
    dd['nclassy'] = args.nclassy

    return dd

def generate_random_multiclass(args):
    # if random choice is 0 then return homogeneous stiffness
    dd      = generate_random(args)
    choices = list(range((args.nclassx*args.nclassy)+1))
    ichoice = random.choice(choices)
    
    if (ichoice == 0):
        dd['stftype']='homogeneous'

    return dd

def make_file_names(args,idx):
    '''
    if ( idx < args.ntrain + args.shift):
        dirname     = f'{args.prefix}' + '_train'  + str(idx)+'/'

    if ( args.ntrain <= idx < args.ntrain + args.nvalid + args.shift):
        dirname     = f'{args.prefix}' + '_valid'  + str(idx)+'/'

    if ( (args.ntrain + args.nvalid) <= idx < args.ntotal):
        dirname     = f'{args.prefix}' + '_test'  + str(idx)+'/'
    '''
    dirname     = f'{args.prefix}' + str(idx)+'/'
    inputname   = f'input'  + str(idx) + '.json.in'
    outputname  = f'output' + str(idx) + '.json.out'
    mlinfoname  = f'{dirname}'+f'mlinfo'  + str(idx) + '.json.in'
    
    return dirname,inputname,outputname,mlinfoname
 
if __name__ == '__main__':
    
    args = getargs()

    # clean directory if asked to. safer than manual rm -rf *
    if (args.clean == 'True'):
        gg = glob.glob(f'{args.prefix}*')
        for g in gg:
            shutil.rmtree(g)

    if ( args.generate == 'True') and (args.problemtype =='binary'):
        outlist,labels,counts,category = generate_binary_training_parameters(args)

    if ( args.generate == 'True') and (args.problemtype =='multiclass'):
        outlist,labels,counts,category = generate_multiclass_training_parameters(args)

    if (args.generate == 'True'):
        for iexample,argdict in enumerate(outlist):
            dirname,inputname,outputname,mlinfoname = make_file_names(args,iexample+args.shift)
            
            print(f'Creating training inputfiles for example {iexample+1} of {args.ntotal} {args.problemtype} classification')
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

            if (args.problemtype=='multiclass'):
                if ( ( iexample >=2 ) and (iexample < args.nminexamp) ) and ( argdict['stftype'] == 'homogeneous'):
                    print(f'Invalid homogeneous entry at {iexample=}')

            # dump machine learning info
            with open(mlinfoname,'x') as fout:
                json.dump(mlinfodir,fout,indent=4)

    if (args.solve == 'True'):
        for iexample in range(args.ntotal):
             dirname,inputname,outputname,mlinfoname = make_file_names(args,iexample+args.shift)
             print(f'Solving training example {iexample+1} of {args.ntotal} {args.problemtype} classification')
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
             
                                   

