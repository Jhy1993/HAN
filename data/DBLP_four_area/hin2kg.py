# -*- coding: utf-8 -*-
# @Author: Jhy
# @Date:   2017-09-06 20:20:58
# @Last Modified by:   Jhy
# @Last Modified time: 2017-09-07 10:03:29
# 将HIN的数据转为为类似 FB15K的数据
#entity2id.txt
#relation2id.txt
# train.txt

import os
import pickle
from collections import Counter

import numpy as np 
import tensorflow as tf
a = './author.txt'
p = './paper.txt'
t = './term.txt'
c = './conf.txt'
a_list = []
with open(a, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        a_list.append(str('A' + line[0]))

p_list = []
with open(p, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        p_list.append(str('P' + line[0]))
t_list = []
with open(t, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        t_list.append(str('T' + line[0]))
c_list = []
with open(c, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        c_list.append(str('C' + line[0]))

ent_list = []
ent_list.extend(a_list)
ent_list.extend(p_list)
ent_list.extend(t_list)
ent_list.extend(c_list)
ent2id = dict()
for ent in ent_list:
    ent2id[ent] = len(ent2id)
    
rel_list = ['PA', 'PC', 'PT']
rel2id = {'PA':0, 'PC': 1, 'PT': 2}

pa = './paper_author.txt'
pc = './paper_conf.txt'
pt = './paper_term.txt'
pa_list = []
with open(pa, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        trip = [str('P' + line[0]), str('A' + line[1]), 'PA']
        pa_list.append(tuple(trip))
pc_list = []
with open(pc, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        trip = [str('P' + line[0]), str('C' + line[1]), 'PC']
        pc_list.append(tuple(trip))
pt_list = []
with open(pt, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n').split('\t')
        trip = [str('P' + line[0]), str('T' + line[1]), 'PT']
        pt_list.append(tuple(trip))
trip_list = []
trip_list.extend(pa_list)
trip_list.extend(pc_list)
trip_list.extend(pt_list)

with open('DBLP_Triplets.pickle', 'wb') as f:
    pickle.dump([ent_list, rel_list, trip_list], f)

'''
class DataTrans:
    """docstring for DataTrans"""
    def __init__(self):

        self.a = './author.txt'
        self.p = './paper.txt'
#        self.p = 'C:\\Users\Jhy\\Desktop\\gpu\\HINE\\datasets\\DBLP_four_area\\paper.txt'
        self.t = './term.txt'
        self.c = './conf.txt'
        self.entity2id = self.get_entity2id()
        self.id2entity = dict(zip(self.entity2id.values(), self.entity2id.keys()))
        self.relation2id = self.get_relation2id()

        self.pa = '/paper_author.txt'
        self.pc = '/paper_conf.txt'
        self.pt = './paper_term.txt'
        self.triplet = self.get_triplet()

    def get_entity2id(self):
        ent = []

        with open(self.a, 'r') as fa:
            lines = fa.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                tmp = str('A' + line[0])
                ent.append(tmp)
        print(len(ent))
        with open(self.p, 'r') as fp:
            line = fp.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                tmp =str('P' + line[0])
                ent.append(tmp)

        print(len(ent))
        with open(self.t, 'r') as ft:
            line = ft.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                tmp = str('T' + line[0])
                ent.append(tmp)


        with open(self.c, 'r') as fc:
            line = fc.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                tmp = str('C' + line[0])
                ent.append(tmp)

        count = Counter(ent)
        entity2id = dict()
        for k in list(count.keys()):
            entity2id[k] = len(entity2id)
        return entity2id
                

    def get_relation2id(self):
        d = {'PA': 0, 'PC': 1, 'PT': 2}
        return d

    def get_triplet(self, mode='train.txt'):
        f = open(mode, 'w')
        with open(self.pa, 'r') as f:


        pass


if __name__ == '__main__':
    DT = DataTrans()
    ent2id = DT.entity2id
    id2ent = DT.id2entity
    print(id2ent[31563])
    rel2id = DT.relation2id

'''