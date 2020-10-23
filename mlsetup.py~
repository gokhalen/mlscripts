import argparse
import timerit
import os

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
    parser.add_argument('--length',help='length of the domain in y direction',required=False,type=int,default=2.0)
    parser.add_argument('--breadth',help='length of the domain in x direction',required=False,type=int,default=5.0)
    
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

    if ( args.generate == 'True') or ( args.solve == 'True'):
        for itrain in range(args.ntrain):
            dirname    = 'traindata'         + str(itrain)+'/'
            inputname  = f'input'  + str(itrain) + '.json.in'
            outputname = f'output' + str(itrain) + '.json.out'

            if (args.generate =='True'):
                print(f'Creating training inputfiles for example {itrain+1} of {args.ntrain}')
                mesh2d = FyPyMesh(inputdir=dirname,outputdir=dirname)
                os.mkdir(dirname)
                mesh2d.create_mesh_2d(length=2,breadth=5,nelemx=8,nelemy=8,stf='homogeneous',bctype='trac')
                mesh2d.json_dump(filename=inputname)

            if (args.solve =='True'):

                print(f'Solving example {itrain+1} of {args.ntrain}')
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
