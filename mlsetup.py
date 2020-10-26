import argparse
import os
import glob
import shutil
import json
import random
from timerit import Timer

# the object which will generate data
from fypy.src.fypy    import FyPy
from fypy.src.libmesh import FyPyMesh

def getargs():
    parser = argparse.ArgumentParser(description='driver script to generate data for Elasticity Imaging ml')
    parser.add_argument('--prefix',help='prefix for training example directories',required=False,default='traindata',type=str)

    parser.add_argument('--problemtype',help='binary classification or multiclass classification',required=True,choices=['binary','multiclass'])
    parser.add_argument('--generate',help='generate input files for training examples',required=False,default='False')
    parser.add_argument('--solve',help='solve training examples',required=False,default='False')


    parser.add_argument('--ninc',help='number of inclusions',required=False,type=int,default=1)
    parser.add_argument('--rmin',help='lower bound on inclusion radius',required=False,type=float,default=0.1)
    parser.add_argument('--rmax',help='lower bound on inclusion radius',required=False,type=float,default=0.25)
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
    if (args.problemtype =='multiclass'):
        # extra 1 is for the homogeneous case
        args.nlabel = args.nclassx*args.nclassy + 1
        nclass = (args.nlabel)*args.nclassmin
        assert( args.ntrain > nclass ),f'Number of training examples {args.ntrain} exceed {nclass}, computed on basis of minimum number of examples per class'
        
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
    outlist = []
    labels  = []
    for itrain in range(0,args.ntrain):
        dd = generate_random(args)
        outlist.append(dd)
        labels.append(1)
        
    # make homogeneous examples
    for ihomo in range(0,args.nhomo):
        outlist[ihomo]['stf']='homogeneous'
        labels[ihomo] = 0
        
    # making and counting (how many positive and how many negative) labels
    
    return outlist,labels

# will return list of dictionaries of parameter lists
def generate_multiclass_training_parameters(args):
    # nhomo homogeneous examples + nclass classification examples + ntrain random examples
    outlist = []
    labels  = []
    for itrain in range(0,args.ntrain):
        dd = generate_random(args)
        outlist.append(dd)
        labels.append(1)

    make_label(args,outlist)

    # generate the homogeneous examples
    # for ihomo in range(0,args.nhomo):
    #    outlist[ihome]['stf']='homogeneous'

    # make sure there is atleast 1 example for each class
    # making and counting labels of each type
    return outlist,labels

def make_label(args,outlist):
    label = [0]*(args.nlabel)
    print(args.nlabel)
    pass

def generate_random(args):
    # generates a random dictionary of parameters
    dd = {}

    dd['length']  = args.length
    dd['breadth'] = args.breadth
    dd['nelemx']  = args.nelemx
    dd['nelemy']  = args.nelemy
    dd['stf']     = 'inclusion'
    dd['bctype']  = 'trac'
    dd['ninc']    = args.ninc
    dd['rmin']    = args.rmin
    dd['rmax']    = args.rmax
    dd['radius']  = random.uniform(args.rmin,args.rmax)
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
             labelname   = f'{dirname}'+f'label'  + str(itrain) + '.json.in'
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
             
                                   
    # if ( args.generate == 'True') or ( args.solve == 'True'):
        
    #     for itrain in range(args.ntrain):
    #         dirname    = f'{dirprefix}'         + str(itrain)+'/'
    #         inputname  = f'input'  + str(itrain) + '.json.in'
    #         outputname = f'output' + str(itrain) + '.json.out'

    #         if (args.generate =='True'):
    #             print(f'Creating training inputfiles for example {itrain+1} of {args.ntrain}')
    #             mesh2d = FyPyMesh(inputdir=dirname,outputdir=dirname)
    #             os.mkdir(dirname)
    #             mesh2d.create_mesh_2d(
    #                                   length  = args.length,
    #                                   breadth = args.breadth,
    #                                   nelemx  = args.nelemx,
    #                                   nelemy  = args.nelemy,
    #                                   stf     = 'inclusion',
    #                                   bctype  = 'trac',
    #                                   ninc    = args.ninc,
    #                                   rmin    = args.rmin,
    #                                   rmax    = args.rmax,
    #                                   radius  = 1.0,
    #                                   xcen    = 2.0,
    #                                   ycen    = 2.0,
    #                                   stfmin  = args.stfmin,
    #                                   stfmax  = args.stfmax,
    #                                   nu      = args.nu,
    #                                   nclassx = args.nclassx,
    #                                   nclassy = args.nclassy,
    #                                  )

    #             mesh2d.json_dump(filename=inputname)
    #             mesh2d.preprocess(str(itrain))

    #         if (args.solve =='True'):
    #             tsolve = Timer('Solve timer',verbose=0)
    #             with tsolve:
    #                 fypyargs = FyPyArgs(
    #                     inputfile  = inputname,
    #                     outputfile = outputname,
    #                     inputdir   = dirname,
    #                     outputdir  = dirname,
    #                     partype    = 'list',
    #                     profile    = 'False',
    #                     solvertype = 'spsolve'
    #                 )
    #                 fypy = FyPy(fypyargs)
    #                 fypy.doeverything(str(itrain))
    #             print(f'Solved example {itrain+1} of {args.ntrain} in {tsolve.elapsed:.2f}s')

