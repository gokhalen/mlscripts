from collections import namedtuple

# ':' means size of training examples
# binary = [:] 1/0 (inclusion/no inclusion)
# center = [:,2] (x and y co-ordinates of the center)
# radius = [:] list of inclusion radius
# value  = [:] value of the stiffness / shear modulus

Labels=namedtuple('Labels',['binary','center','radius','value','field'])
CNNData=namedtuple('CNNData',['images','labels'])
