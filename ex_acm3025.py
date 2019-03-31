import time
import numpy as np
import tensorflow as tf

from models import GAT, HeteGAT
from utils import process

# 禁用gpu
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#""0,1,2,3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

dataset = 'acm'
checkpt_file = 'pre_trained/{}/{}_allMP.ckpt'.format(dataset, dataset)

# training params
batch_size = 1
nb_epochs = 600
patience = 100
lr = 0.004  # learning rate
l2_coef = 0.001  # weight decay
att_size = 128
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu

model = HeteGAT

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

# jhy data
import scipy.io as sio
import scipy.sparse as sp


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_dblp(path='/home/jhy/allGAT/acm_hetesim/ACM3025.mat'):
    data = sio.loadmat(path)
    truelabels, truefeatures = data['label'], data['feature'].astype(float)
    N = truefeatures.shape[0]
    rownetworks = [data['PAP'] - np.eye(N), data['PLP'] - np.eye(N)] 

    y = truelabels
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))

    return rownetworks, truefeatures, y_train, y_val, y_test, train_mask, val_mask, test_mask


adj_list, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp()



nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

# adj = adj.todense()

features = features[np.newaxis]  # [1, nb_node, ft_size]
adj_list = [adj[np.newaxis] for adj in adj_list]

y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]

train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(
            batch_size, nb_nodes, ft_size))
        bias_in_list = [tf.placeholder(dtype=tf.float32, shape=(
            batch_size, nb_nodes, nb_nodes)) for _ in range(len(biases_list))]
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(
            batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        is_train = tf.placeholder(dtype=tf.bool, shape=())
    # forward
    logits, final_embedding, att_val, coef_list = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                                                  attn_drop, ffd_drop,
                                                                  bias_mat_list=bias_in_list,
                                                                  hid_units=hid_units, n_heads=n_heads,
                                                                  residual=residual, activation=nonlinearity,
                                                                  mp_att_size=att_size,  # 0805最后的挣扎
                                                                  return_coef=True)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0
    best_nmi = 0
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        saver.restore(sess, checkpt_file)
        print('load model from : {}'.format(checkpt_file))
        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_train[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: train_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}
            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            loss_value_ts, acc_ts, jhy_final_embedding, jhy_coef_list = sess.run(
                [loss, accuracy, final_embedding, coef_list],
                feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test l1oss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step)

        sess.close()

        print('start knn, kmean.....')
        xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]
        yy = y_test[test_mask]
        from jhyexps import my_KNN, my_Kmeans, my_TSNE, my_Linear

        my_KNN(xx, yy)
        my_Kmeans(xx, yy)

