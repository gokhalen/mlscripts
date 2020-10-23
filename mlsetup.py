import argparse
import os
import glob
import shutil
from timerit import Timer

# the object which will generate data
from fypy.src.fypy    import FyPy
from fypy.src.libmesh import FyPyMesh

def getargs():
    parser = argparse.ArgumentParser(description='driver script to generate data for ml')
    parser.add_argument('--ntrain',help='number of training examples to generate',required=False,type=int,default=16)
    parser.add_argument('--ninc',help='number of inclusions',required=False,type=int,default=1)
    parser.add_argument('--rmin',help='lower bound on inclusion radius',required=False,type=float,default=0.1)
    parser.add_argument('--rmax',help='lower bound on inclusion radius',required=False,type=float,default=0.25)

    parser.add_argument('--generate',help='generate input files for training examples',required=False,default='False')
    parser.add_argument('--solve',help='solve training examples',required=False,default='False')

    parser.add_argument('--nelemx',help='number of elements in the x-direction',required=False,type=int,default=16)
    parser.add_argument('--nelemy',help='number of elements in the y-direction',required=False,type=int,default=16)
    parser.add_argument('--length',help='length of the domain in x direction',required=False,type=int,default=2.0)
    parser.add_argument('--breadth',help='length of the domain in y direction',required=False,type=int,default=5.0)

    parser.add_argument('--clean',help='delete previous data',required=False,type=str,default='False')
    
    args = parser.parse_args()
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
        
                      
 
if __name__ == '__main__':
    
    args = getargs()
    dirprefix='traindata'

    if (args.clean == 'True'):
        gg = glob.glob(f'{dirprefix}*')
        for g in gg:
            shutil.rmtree(g)

    if ( args.generate == 'True') or ( args.solve == 'True'):
        for itrain in range(args.ntrain):
            dirname    = f'{dirprefix}'         + str(itrain)+'/'
            inputname  = f'input'  + str(itrain) + '.json.in'
            outputname = f'output' + str(itrain) + '.json.out'

            if (args.generate =='True'):
                print(f'Creating training inputfiles for example {itrain+1} of {args.ntrain}')
                mesh2d = FyPyMesh(inputdir=dirname,outputdir=dirname)
                os.mkdir(dirname)
                mesh2d.create_mesh_2d(
                                      length  = args.length,
                                      breadth = args.breadth,
                                      nelemx  = args.nelemx,
                                      nelemy  = args.nelemy,
                                      stf     = 'random',
                                      bctype  = 'trac',
                                      ninc    = args.ninc,
                                      rmin    = args.rmin,
                                      rmax    = args.rmax
                                     )

                mesh2d.json_dump(filename=inputname)

            if (args.solve =='True'):
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

