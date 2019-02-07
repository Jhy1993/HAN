# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 21:40:04 2017

@author: Jhy
"""

#import pickle
#with open('DBLP_Triplets.pickle', 'rb') as f:
#    [ent_list, rel_list, trip_list] = pickle.load(f)
import tensorflow as tf 
a = tf.constant([[1, 3], [4, 5]])
sess = tf.InteractiveSession()
print(a.eval())
b = tf.reshape(a, [-1])
print(b.eval())