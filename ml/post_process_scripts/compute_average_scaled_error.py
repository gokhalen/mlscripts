# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 08:54:29 2021

@author: aa
"""

import json
import numpy as np

filelist = ['logfile_softplus.txt',
            'logfile_twisted_p50.txt',
            'logfile_twisted_p25.txt',
            'logfile_twisted_p0.txt',
            ]


for filename in filelist:
    with open(filename,'rt') as fin:
        dd=json.load(fin)
        
        
        
    average_scaled_error = sum(dd['scaled_error_list'])/len(dd['scaled_error_list'])
    print(f'for {filename} {average_scaled_error=}')
    
    