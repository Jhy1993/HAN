# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:04:37 2017

@author: Jhy_BUPT

README:

INPUT:

OUTPUT:

REFERENCE:

"""
from __future__ import print_function

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nltk.corpus import stopwords
term = {}
with open('./term.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip('\n').split('\t')
        term[line[0]] = line[1]
#print(term)
sp = stopwords.words('english')
#term_sp = [x for x in term if x not in sp]

with open('./term_modify.txt', 'w') as f:
    for k, v in term.items():
        f.write(k)
        f.write('\t')
        f.write(v)
        f.write('\n')
PT = {}
with open('./paper_term.txt', 'r') as f:
    for line in f.readlines():
       line = line.strip('\n').split('\t')
       PT[line[0]] = line[1]
       
    
with open('./paper_term_modify.txt', 'w') as f:
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
