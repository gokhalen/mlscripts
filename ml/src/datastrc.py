from collections import namedtuple
from .config import mltypelist
# ':' means size of training examples
# binary = [:] 1/0 (inclusion/no inclusion)
# center = [:,2] (x and y co-ordinates of the center)
# radius = [:] list of inclusion radius
# value  = [:] value of the stiffness / shear modulus

Labels         = namedtuple('Labels', mltypelist)
CNNData        = namedtuple('CNNData',['images','strain','labels'])
PostData       = namedtuple('PostData',mltypelist)
BinaryPostData = namedtuple('BinaryPostData',['accu_score','conf_matrix']) 
