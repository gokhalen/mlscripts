import argparse
import os
import glob
import shutil
import json
import random

from timerit import Timer
from collections import Counter

# the object which will generate data
from fypy.src.fypy    import FyPy
from fypy.src.libmesh import FyPyMesh

def getargs():
    parser = argparse.ArgumentParser(description='driver script to generate data for Elasticity Imaging Machine Learning (EIML)')
    
    parser.add_argument('--prefix',help='prefix for training example directories',required=False,default='traindata',type=str)

    parser.add_argument('--problemtype',help='binary classification or multiclass classification',required=True,choices=['binary','multiclass'])
    parser.add_argument('--generate',help='generate input files for training examples',required=False,default='False')
    parser.add_argument('--solve',help='solve training examples',required=False,default='False')


    parser.add_argument('--ninc',help='number of inclusions',required=False,type=int,default=1)
    parser.add_argument('--rmin',help='lower bound on inclusion radius',required=False,type=float,default=0.7)
    parser.add_argument('--rmax',help='lower bound on inclusion radius',required=False,type=float,default=1.0)
    parser.add_argument('--nelemx',help='number of elements in the x-direction',required=False,type=int,default=16)
    parser.add_argument('--nelemy',help='number of elements in the y-direction',required=False,type=int,default=16)
    parser.add_argument('--length',help='length of the domain in x direction',required=False,type=int,default=2.0)
    parser.add_argument('--breadth',help='length of the domain in y direction',required=False,type=int,default=5.0)
    # total number of classes is nclass=nclassx*nclassy + 1 (for homogeneous)
    parser.add_argument('--nclassx',help='number of classes to classify into (in x direction)',required=False,type=int,default=1)
    parser.add_argument('--nclassy',help='number of classes to classify into (in y direction)',required=False,type=int,default=1)
    parser.add_argument('--stfmin',help='lower bound on mu',required=False,type=float,default=1.0)
    parser.add_argument('--stfmax',help='upper bound on mu',required=False,type=float,default=5.0)
    parser.add_argument('--nu',help="Poisson's ratio",required=False,type=float,default=0.25)

    # ntrain should be more than nclass + 1
    parser.add_argument('--ntrain',help='number of training examples to generate',required=False,type=int,default=16)
    parser.add_argument('--nhomo',help='number of homogeneous examples to generate',required=False,type=int,default=4)
    parser.add_argument('--nclassmin',help='minimum number of examples in each class',required=False,type=int,default=2)

    
    # system arguments
    parser.add_argument('--clean',help='delete previous data',required=False,type=str,default='False')
    
    args = parser.parse_args()

    if ( args.problemtype == 'binary'):
        assert(args.ntrain > args.nhomo),f'Number of training examples {args.ntrain} must exceed number of homogeneous examples {args.nhomo}'
        args.nlabel =  1
    if (args.problemtype =='multiclass'):
        # extra 1 is for the homogeneous case
        args.nlabel    = args.nclassx*args.nclassy + 1
        args.nminexamp = (args.nlabel)*args.nclassmin
        assert( args.ntrain > args.nminexamp ),f'Number of training examples {args.ntrain} must exceed {args.nminexamp}, computed on basis of minimum number of examples per class'
        
    return args

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
    # nhomo homogeneous examples + ntrain random examples
    outlist = []                                # list of arguments to pass to mesh creator
    labels  = []                                # list of labels
    for itrain in range(0,args.ntrain):
        dd = generate_random(args)
        outlist.append(dd)
        labels.append([1])
        
    # make homogeneous examples
    for ihomo in range(0,args.nhomo):
        outlist[ihomo]['stf']='homogeneous'
        labels[ihomo] = [0]
        
    # making and counting (how many positive and how many negative) labels
    
    return outlist,labels

# will return list of dictionaries of parameter lists
def generate_multiclass_training_parameters(args):
    # nhomo homogeneous examples + nclass classification examples + ntrain random examples

    print(f'Generating {args.ntrain} training examples,{args.nminexamp}',
          f'compulsary/minimum training examples {args.ntrain-args.nminexamp}',
          f'random training examples')
    
    outlist  = []               # list of arguments to pass to mesh creator
    labels   = []*args.ntrain   # list of labels (list of lists of size nlabel)
    centers  = []               # list of centers used to determine labels.
    category = []*args.ntrain   # numerical value of the cateory for each label

    # A typical label is [ 0 1 0 0 ...0] of length nlabel
    # having 1 in the class where the stiffness belongs to and zeros elsewhere


    dx  = args.length  / args.nclassx
    dy  = args.breadth / args.nclassy
    
    # initialize centers
    for iclassx in range(args.nclassx):
        for iclassy in range(args.nclassy):
            idx  = iclassx*args.nclassy + iclassy
            xcen = (dx/2.0) + dx*iclassx
            ycen = (dy/2.0) + dy*iclassy
            centers.append([xcen,ycen])
            
    # generate arguments for the mesher and label 'all' examples
    for itrain in range(0,args.ntrain):
        dd = generate_random(args)
        outlist.append(dd)
        # create dummy labels for now
        # labels.append([0]*args.ntrain)
        ll = get_closest(args,centers,dd['xcen'],dd['ycen'])
        labels.append(ll)

    # generate the homogeneous examples
    # by changing the arguments ('inclusion'->'homogeneous') for the mesher
    for ihomo in range(0,args.nclassmin):
        outlist[ihomo]['stf']='homogeneous'
        ll    = [0]*args.nlabel
        ll[0] = 1                     # homogeneous example gets 1 in the first entry, zeros everywhere else
        labels[ihomo] = ll

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
                    xmin = dx*iclassx; xmax = (dx+1)*iclassx
                    ymin = dy*iclassy; ymax = (dy+1)*iclassy
                    xcen = random.uniform(xmin,xmax)
                    ycen = random.uniform(ymin,ymax)
                    rad  = min(dx/2.0,dy/2.0)*0.6*random.uniform(0.6,1.0)

                outlist[idx]['radius'] = rad
                outlist[idx]['xcen']   = xcen
                outlist[idx]['ycen']   = ycen
                ll = get_closest(args,centers,xcen,ycen)
                labels[idx] = ll
                # create dummy labels for now
                # labels.append([0]*args.ntrain)
                # print(rad,xcen,ycen)

    # make sure there is atleast 1 example for each class
    # making and counting labels of each type

    return outlist,labels

def get_closest(args,centers,xx,yy):
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
    # return label vector and label id
    return label


def generate_random(args):
    # generates a random dictionary of parameters
    dd = {}
    
    dx   = args.length  / args.nclassx
    dy   = args.breadth / args.nclassy
    rad  = min(dx/2.0,dy/2.0)*0.6
    
    dd['length']  = args.length
    dd['breadth'] = args.breadth
    dd['nelemx']  = args.nelemx
    dd['nelemy']  = args.nelemy
    dd['stf']     = 'inclusion'
    dd['bctype']  = 'trac'
    dd['ninc']    = args.ninc
    dd['rmin']    = args.rmin
    dd['rmax']    = args.rmax
    dd['radius']  = random.uniform(rad*args.rmin,rad*args.rmax)
    dd['xcen']    = random.uniform(0.0,args.length)
    dd['ycen']    = random.uniform(0.0,args.breadth)
    dd['stfmin']  = args.stfmin
    dd['nu']      = args.nu
    dd['nclassx'] = args.nclassx
    dd['nclassy'] = args.nclassy

    return dd 
 
if __name__ == '__main__':
    
    args = getargs()
    dirprefix= args.prefix

    # clean directory if asked to. safer than manual rm -rf *
    if (args.clean == 'True'):
        gg = glob.glob(f'{dirprefix}*')
        for g in gg:
            shutil.rmtree(g)

    if ( args.generate == 'True') and (args.problemtype =='binary'):
        outlist,labels = generate_binary_training_parameters(args)

    if ( args.generate == 'True') and (args.problemtype =='multiclass'):
        outlist,labels = generate_multiclass_training_parameters(args)

    if (args.generate == 'True'):
        for itrain,argdict in enumerate(outlist):
             dirname     = f'{dirprefix}'         + str(itrain)+'/'
             outputname  = f'input'  + str(itrain) + '.json.in'
             labelname   = f'{dirname}'+f'mlinfo'  + str(itrain) + '.json.in'
             print(f'Creating training inputfiles for example {itrain+1} of {args.ntrain} {args.problemtype} classification')
             mesh2d = FyPyMesh(inputdir=dirname,outputdir=dirname)
             os.mkdir(dirname)
             mesh2d.create_mesh_2d(**argdict)
             mesh2d.json_dump(filename=outputname)
             mesh2d.preprocess(str(itrain))
             # dump label
             labeldir = {}
             labeldir['label']=labels[itrain]
             with open(labelname,'x') as fout:
                 json.dump(labeldir,fout,indent=4)

    if (args.solve == 'True'):
        for itrain,dd in enumerate(outlist):
             dirname    = f'{dirprefix}'         + str(itrain)+'/'
             inputname  = f'input'  + str(itrain) + '.json.in'
             outputname = f'output' + str(itrain) + '.json.out'
             print(f'Solving training example {itrain+1} of {args.ntrain} {args.problemtype} classification')
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
                 fypy.doeverything(str(itrain))
             print(f'Solved example {itrain+1} of {args.ntrain} in {tsolve.elapsed:.2f}s')
             
                                   

