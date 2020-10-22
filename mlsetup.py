import argparse
import timerit
import os

# the object which will generate data
from fypy.src.fypy    import FyPy
from fypy.src.libmesh import FyPyMesh

def getargs():
    parser = argparse.ArgumentParser(description='driver script to generate data for ml')
    # number of training examples to generate
    parser.add_argument('--ntrain',help='number of training examples to generate',required=False,type=int,default=16)

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

    dirlist    = []
    inputlist  = []
    outputlist = []
    

    
    for itrain in range(args.ntrain):
        dirname    = 'traindata'         + str(itrain)+'/'
        inputname  = f'input'  + str(itrain) + '.json.in'
        outputname = f'output' + str(itrain) + '.json.out'
        
        dirlist.append(dirname)
        inputlist.append(inputname)
        outputlist.append(outputname)

        print(f'Creating training inputfiles for example {itrain+1} of {args.ntrain}')
        mesh2d = FyPyMesh(inputdir=dirname,outputdir=dirname)
        os.mkdir(dirname)
        mesh2d.create_mesh_2d(length=2,breadth=5,nelemx=8,nelemy=8,stf='homogeneous',bctype='trac')
        mesh2d.json_dump(filename=inputname)


    # run solver
    # arguments for fypy
    for itr,(dirname,ipf,opf) in enumerate(zip(dirlist,inputlist,outputlist)):
        print(f'Solving example {itr+1} of {args.ntrain}')
        fypyargs = FyPyArgs(
                            inputfile  = ipf,
                            outputfile = opf,
                            inputdir   = dirname,
                            outputdir  = dirname,
                            partype    = 'list',
                            profile    = 'False',
                            solvertype = 'spsolve'
                           )
        fypy = FyPy(fypyargs)
        fypy.doeverything(str(itr))
        
        # fypy.assembly()
        # fypy.solve()
        # fypy.output()
        # fypy.postprocess()

