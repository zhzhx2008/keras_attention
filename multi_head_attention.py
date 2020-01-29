# coding=utf-8
import os
import sys
from distutils.util import strtobool

import six
import tensorflow as tf
from keras.layers import *

# https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py

is_py2 = six.PY2

if not is_py2:
    basestring = str


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, basestring)


# 判断是tf.keras还是纯keras的标记
is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

if is_tf_keras:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K

    sys.modules['keras'] = keras
else:
    import keras
    import keras.backend as K


def gelu_erf(x):
    """基于Erf直接计算的gelu函数
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    """基于Tanh近似计算的gelu函数
    """
    cdf = 0.5 * (1.0 + K.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
    return x * cdf


def set_gelu(version):
    """设置gelu版本
    """
    version = version.lower()
    assert version in ['erf', 'tanh'], 'gelu version must be erf or tanh'
    if version == 'erf':
        keras.utils.get_custom_objects()['gelu'] = gelu_erf
    else:
        keras.utils.get_custom_objects()['gelu'] = gelu_tanh


def piecewise_linear(t, schedule):
    """分段线性函数
    其中schedule是形如{1000: 1, 2000: 0.1}的字典，
    表示 t ∈ [0, 1000]时，输出从0均匀增加至1，而
    t ∈ [1000, 2000]时，输出从1均匀降低到0.1，最后
    t > 2000时，保持0.1不变。
    """
    schedule = sorted(schedule.items())
    if schedule[0][0] != 0:
        schedule = [(0, 0.)] + schedule

    x = K.constant(schedule[0][1], dtype=K.floatx())
    t = K.cast(t, K.floatx())
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1. * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = K.constant(schedule[i][1], dtype=K.floatx())
        x = K.switch(t >= t_begin, x, x_begin)

    return x


def search_layer(inputs, name, exclude=None):
    """根据inputs和name来搜索层
    说明：inputs为某个层或某个层的输出；name为目标层的名字。
    实现：根据inputs一直往上递归搜索，直到发现名字为name的层为止；
         如果找不到，那就返回None。
    """
    if exclude is None:
        exclude = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude:
        return None
    else:
        exclude.add(layer)
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude)
                if layer is not None:
                    return layer


def sequence_masking(x, mask, mode=0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    mode: 如果是0，则直接乘以mask；
          如果是1，则在padding部分减去一个大正数。
    axis: 序列所在轴，默认为1；
    """
    if mask is None or mode not in [0, 1]:
        return x
    else:
        if axis is None:
            axis = 1
        if axis == -1:
            axis = K.ndim(x) - 1
        assert axis > 0, 'axis muse be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        for _ in range(K.ndim(x) - K.ndim(mask) - axis + 1):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 0:
            return x * mask
        else:
            return x - (1 - mask) * 1e12


def pool1d(x,
           pool_size,
           strides=1,
           padding='valid',
           data_format=None,
           pool_mode='max'):
    """向量序列的pool函数
    """
    x = K.expand_dims(x, 1)
    x = K.pool2d(x,
                 pool_size=(1, pool_size),
                 strides=(1, strides),
                 padding=padding,
                 data_format=data_format,
                 pool_mode=pool_mode)
    return x[:, 0]


def divisible_temporal_padding(x, n):
    """将一维向量序列右padding到长度能被n整除
    """
    r_len = K.shape(x)[1] % n
    p_len = K.switch(r_len > 0, n - r_len, 0)
    return K.temporal_padding(x, (0, p_len))


def swish(x):
    """swish函数（这样封装过后才有 __name__ 属性）
    """
    return tf.nn.swish(x)


def leaky_relu(x, alpha=0.2):
    """leaky relu函数（这样封装过后才有 __name__ 属性）
    """
    return tf.nn.leaky_relu(x, alpha=alpha)


def symbolic(f):
    """恒等装饰器（兼容旧版本keras用）
    """
    return f


class MultiHeadAttention(Layer):
    """多头注意力机制
    """

    def __init__(self,
                 heads,
                 head_size,
                 key_size=None,
                 pool_size=None,
                 kernel_initializer='glorot_uniform',
                 max_relative_position=None,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = heads * head_size
        self.key_size = key_size or head_size
        self.pool_size = pool_size or 1
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.max_relative_position = max_relative_position

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(units=self.key_size * self.heads,
                             kernel_initializer=self.kernel_initializer)
        self.k_dense = Dense(units=self.key_size * self.heads,
                             kernel_initializer=self.kernel_initializer)
        self.v_dense = Dense(units=self.out_dim,
                             kernel_initializer=self.kernel_initializer)
        self.o_dense = Dense(units=self.out_dim,
                             kernel_initializer=self.kernel_initializer)

        if self.max_relative_position is not None:
            if self.head_size != self.key_size:
                raise ValueError('head_size must be equal to key_size ' +
                                 'while use relative position embeddings.')

            def initializer(shape, dtype=None):
                vocab_size, depth = shape
                embeddings = np.zeros(shape)
                for pos in range(vocab_size):
                    for i in range(depth // 2):
                        theta = pos / np.power(10000, 2. * i / depth)
                        embeddings[pos, 2 * i] = np.sin(theta)
                        embeddings[pos, 2 * i + 1] = np.cos(theta)
                return embeddings

            shape = (2 * self.max_relative_position + 1, self.head_size)
            self.relative_embeddings = self.add_weight(name='relative_embeddings',
                                                       shape=shape,
                                                       initializer=initializer,
                                                       trainable=False)

    def call(self, inputs, q_mask=None, v_mask=None, a_mask=None):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        a_mask: 对attention矩阵的mask。
                不同的attention mask对应不同的应用。
        """
        q, k, v = inputs[:3]
        if a_mask:
            if len(inputs) == 3:
                a_mask = 'history_only'
            else:
                a_mask = inputs[3]
        if q_mask is not None:
            if not hasattr(self, 'q_mask_layer'):
                self.q_mask_layer = search_layer(q, q_mask)
            q_mask = self.q_mask_layer.output_mask
        if v_mask is not None:
            if not hasattr(self, 'v_mask_layer'):
                self.v_mask_layer = search_layer(v, v_mask)
            v_mask = self.v_mask_layer.output_mask
        # Pooling
        if self.pool_size > 1:
            is_self_attention = (q is k is v)
            q_in_len = K.shape(q)[1]
            q = sequence_masking(q, q_mask, 0)
            q = divisible_temporal_padding(q, self.pool_size)
            q = pool1d(q, self.pool_size, self.pool_size, pool_mode='avg')
            if is_self_attention:
                k = v = q
            else:
                k = sequence_masking(k, v_mask, 0)
                k = divisible_temporal_padding(k, self.pool_size)
                k = pool1d(k, self.pool_size, self.pool_size, pool_mode='avg')
                v = sequence_masking(v, v_mask, 0)
                v = divisible_temporal_padding(v, self.pool_size)
                v = pool1d(v, self.pool_size, self.pool_size, pool_mode='avg')
            if v_mask is not None:
                v_mask = v_mask[:, ::self.pool_size]
            if a_mask is not None and not is_string(a_mask):
                a_mask = a_mask[..., ::self.pool_size, ::self.pool_size]
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 相对位置编码
        if self.max_relative_position is not None:
            q_idxs = K.arange(0, K.shape(q)[1], dtype='int32')
            q_idxs = K.expand_dims(q_idxs, 1)
            v_idxs = K.arange(0, K.shape(v)[1], dtype='int32')
            v_idxs = K.expand_dims(v_idxs, 0)
            pos_ids = v_idxs - q_idxs
            pos_ids = K.clip(pos_ids, -self.max_relative_position,
                             self.max_relative_position)
            pos_ids = pos_ids + self.max_relative_position
            pos_embeddings = K.gather(self.relative_embeddings, pos_ids)
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, pos_embeddings)
        # Attention（续）
        a = a / self.key_size ** 0.5
        a = sequence_masking(a, v_mask, 1, -1)
        if a_mask is not None:
            if is_string(a_mask):
                ones = K.ones_like(a[:1, :1])
                a_mask = (ones - tf.linalg.band_part(ones, -1, 0)) * 1e12
                a = a - a_mask
            else:
                a = a - (1 - a_mask) * 1e12
        a = K.softmax(a)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', a, vw)
        if self.max_relative_position is not None:
            o = o + tf.einsum('bhjk,jkd->bjhd', a, pos_embeddings)
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.o_dense(o)
        # 恢复长度
        if self.pool_size > 1:
            o = K.repeat_elements(o, self.pool_size, 1)[:, :q_in_len]
        # 返回结果
        o = sequence_masking(o, q_mask, 0)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'key_size': self.key_size,
            'pool_size': self.pool_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'max_relative_position': self.max_relative_position,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
