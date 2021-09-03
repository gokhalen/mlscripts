# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:30:51 2021

@author: User
"""

import numpy as np
import json

file_list = ['softplus.txt','twisted0p50.txt',
             'twisted0p25.txt','twisted0p0.txt']

# the softplus example for which minimum mu is seen
nmin  =  908
ntest = 2000

for _file in file_list:
    with open(_file,'rt') as fin:
        dd = json.load(fin)
        min_ = dd['min_shear_mod']
        max_ = dd['max_shear_mod']
        error_list = dd['scaled_error_list']
        average_scaled_error = np.sum(error_list)/ntest
        print(f'average scaled error for {_file[:-4]}=',average_scaled_error,\
f'scaled_error for {_file[:-4]}=',error_list[nmin],\
f'{max_=},{min_=}'    
)
