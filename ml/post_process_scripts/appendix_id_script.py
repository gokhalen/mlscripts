# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:07:59 2021

@author: User
"""

import json
with open('logcnn2.txt','rt') as fin:
    dd1=json.load(fin)
nn=256
idx=dd1['sorted_idx_list'][nn]
error_list=dd1['scaled_error_list']
print('error = ',error_list[idx],'idx=',idx)