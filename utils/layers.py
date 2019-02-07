import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d


def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False,
              return_coef=False):
    """[summary]
    
    [description]
    
    Arguments:
        seq {[type]} -- shape=(batch_size, nb_nodes, fea_size))

    """
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        # filter个数: out_sz
        # kernel size: 1
        # seq_fts, [batch_size, nb_nodes, out_sz]
        # seq (batch_size, nb_nodes, fea_size) 相当于h, 包括h_1~h_N
        # conv1d有out_sz个卷积核,相当于将seq中h经W投影到out_sz维度, 因为只需要投影, 所以这里不要bias
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # 上述操作相当于 Wh, 注意这里是对所有h同时投影

        # simplest self-attention possible
        # # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_j]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j], 注意这里转为+号
        # f_1, [batch_Size, nb_node, 1], 即[a_1]^T [Wh], 对所有h同时进行a^T
        # [batch_Size, nb_node, 1]中的[nb_node, 1]代表 att中对自己的att
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        # f_2, [batch_size, nb_node, 1], 即[a_2]^T [Wh], 对所有h同时进行a^T
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        # 根据上面的转换, f1 + f2等价于先拼接再乘a
        # [batch_size, nb_node, 1] + [batch_size, 1, nb_node] = [batch_size, nb_node, nb_node]
        # 经过广播后, logit[i, j] = [[a_1], [a_2]]^T [[Wh_i], [Wh_j]]= [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
        # 在进行softmax即可!
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # logits为甚等于 a^T[Wh_i||Wh_j]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq
        if return_coef:
            return activation(ret), coefs
        else:
            return activation(ret)  # activation


def attn_head_const_1(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    """[summary]

    [description]

    Arguments:
        seq {[type]} -- shape=(batch_size, nb_nodes, fea_size))
        bias_mat, 0,1邻接矩阵的转换， 0变为-e9， 1变为 0，为了计算softmax
        还原为邻接矩阵， adjmat = 1.0 - bias_mat / -e9
    """
    adj_mat = 1.0 - bias_mat / -1e9
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        # # filter个数: out_sz
        # # kernel size: 1
        # # seq_fts, [batch_size, nb_nodes, out_sz]
        # # seq (batch_size, nb_nodes, fea_size) 相当于h, 包括h_1~h_N
        # # conv1d有out_sz个卷积核,相当于将seq中h经W投影到out_sz维度, 因为只需要投影, 所以这里不要bias
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        #
        # # 上述操作相当于 Wh, 注意这里是对所有h同时投影
        #
        # # simplest self-attention possible
        # # # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_j]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j], 注意这里转为+号
        # # f_1, [batch_Size, nb_node, 1], 即[a_1]^T [Wh], 对所有h同时进行a^T
        # # [batch_Size, nb_node, 1]中的[nb_node, 1]代表 att中对自己的att
        # f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        # # f_2, [batch_size, nb_node, 1], 即[a_2]^T [Wh], 对所有h同时进行a^T
        # f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        # # 根据上面的转换, f1 + f2等价于先拼接再乘a
        # # [batch_size, nb_node, 1] + [batch_size, 1, nb_node] = [batch_size, nb_node, nb_node]
        # # 经过广播后, logit[i, j] = [[a_1], [a_2]]^T [[Wh_i], [Wh_j]]= [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
        # # 在进行softmax即可!
        # logits = f_1 + tf.transpose(f_2, [0, 2, 1])


        # 这里const——1 将GAT中权重全部设置为1
        logits = adj_mat #.ones([seq.shape[0], seq.shape[1], seq.shape[1]], tf.float32)
        # logits为甚等于 a^T[Wh_i||Wh_j]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation


def jhy_attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    """[summary]
    参考下文的attention
    18ArXiv_AGNN_Attention-based Graph Neural Network for Semi-supervised Lea

    Arguments:
        seq {[type]} -- shape=(batch_size, nb_nodes, fea_size))

    """
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        # filter个数: out_sz
        # kernel size: 1
        # seq_fts, [batch_size, nb_nodes, out_sz]
        # seq (batch_size, nb_nodes, fea_size) 相当于h, 包括h_1~h_N
        # conv1d有out_sz个卷积核,相当于将seq中h经W投影到out_sz维度, 因为只需要投影, 所以这里不要bias
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # 上述操作相当于 Wh, 注意这里是对所有h同时投影

        # simplest self-attention possible
        # # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_j]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j], 注意这里转为+号
        # f_1, [batch_Size, nb_node, 1], 即[a_1]^T [Wh], 对所有h同时进行a^T
        # [batch_Size, nb_node, 1]中的[nb_node, 1]代表 att中对自己的att
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        # f_2, [batch_size, nb_node, 1], 即[a_2]^T [Wh], 对所有h同时进行a^T
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        # 根据上面的转换, f1 + f2等价于先拼接再乘a
        # [batch_size, nb_node, 1] + [batch_size, 1, nb_node] = [batch_size, nb_node, nb_node]
        # 经过广播后, logit[i, j] = [[a_1], [a_2]]^T [[Wh_i], [Wh_j]]= [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
        # 在进行softmax即可!
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # logits为甚等于 a^T[Wh_i||Wh_j]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation


# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!


def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = tf.sparse_add(adj_mat * f_1, adj_mat *
                               tf.transpose(f_2, [0, 2, 1]))
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(
                                        coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation

# final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
#                                                      time_major=False,
#                                                      return_alphas=True)
def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas