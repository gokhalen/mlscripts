# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:30:51 2021

@author: User
"""

import numpy as np
import json


# the softplus example for which minimum mu is seen
munumbers = [967,615,1008,14,147,1657,702,1126]
ntest   = 2000

with open('logcnn3.txt','rt') as fin:
    dd = json.load(fin)
    error_list = dd['scaled_error_list']

for _nmu in munumbers:    
    print(error_list[_nmu])
    
