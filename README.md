# Paper
https://github.com/Jhy1993/Representation-Learning-on-Heterogeneous-Graph
# HAN

The source code of Heterogeneous Graph Attention Network (WWW-2019).

The source code is based on [GAT](https://github.com/PetarV-/GAT) 

# How to preprocess DBLP? 

Demo: preprocess_dblp.py

# Q&A

1. ACM_3025 in our experiments is based on the preprocessed version ACM in other paper (\data\ACM\ACM.mat). Subject is just like Neural Network, Multi-Object Optimization and Face Recognition. In ACM3025, PLP is actually PSP. You can find it in our code.
2. In ACM, train+val+test < node_num. That is because our model is a semi-supervised model which only need a few labels to optimize our model. The num of node can be found in meta-path based adj mat.
3.  "the model can generate node
embeddings for previous unseen nodes or even unseen graph" means the propose HAN can do inductive experiments. However, we cannot find such heterogeneous graph dataset. See experiments setting in Graphsage and GAT for details, especially on PPI dataset.
4. meta-path can be symmetric or asymmetric. HAN can deal with different types of nodes via project them into the same space.
5. Can we change the split of dataset and re-conduct some experiments? of course, you can split the dataset by yourself, as long as you use the same split for all models.

# Datasets

Preprocessed ACM can be found in:
https://pan.baidu.com/s/1V2iOikRqHPtVvaANdkzROw 
提取码：50k2 

Preprocessed DBLP can be found in:
https://pan.baidu.com/s/1Qr2e97MofXsBhUvQqgJqDg 
提取码：6b3h  
# Run

python ex_acm3025.py

# HAN in DGL
https://github.com/dmlc/dgl/tree/master/examples/pytorch/han

# Reference

If you make advantage of the HAN model in your research, please cite the following in your manuscript:

```
@article{han2019,
title={Heterogeneous Graph Attention Network},
author={Xiao, Wang and Houye, Ji and Chuan, Shi and  Bai, Wang and Peng, Cui and P. , Yu and Yanfang, Ye},
journal={WWW},
year={2019}
}
```
