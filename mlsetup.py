import multiprocessing
import argparse
import timerit

# the object which will generate data
from fypy.src.fypy    import FyPy
from fypy.src.libmesh import FyPyMesh

def getargs():
    parser = argparse.ArgumentParser(description='driver script to generate data for ml')
    # number of training examples to generate
    parser.add_argument('--ntrain',help='number of training examples to generate',required=False,type=int,default=16)

    parser.add_argument('--nelemx',help='number of elements in the x-direction',required=False,type=int,default=32)
    parser.add_argument('--nelemy',help='number of elements in the y-direction',required=False,type=int,default=32)
        
    parser.add_argument('--length',help='length of the domain in y direction',required=False,type=int,default=2.0)
    parser.add_argument('--breadth',help='length of the domain in x direction',required=False,type=int,default=5.0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = getargs()

    for itrain in range(ntrain):
        dirname  = 'traindata' + str(itrain)
        filename = 'input' + str(itrain)
    
