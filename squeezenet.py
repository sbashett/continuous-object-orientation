from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d
from tensorflow.contrib.layers import batch_norm, l2_regularizer, xavier_initializer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
import numpy as np

@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with arg_scope([conv2d, max_pool2d]):
            net = _squeeze(inputs, squeeze_depth)
            net = _expand(net, expand_depth)
        return net


def _squeeze(inputs, num_outputs):
    return conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def _expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], 3)


class Squeezenet(object):
    """Original squeezenet architecture for 224x224 images."""
    name = 'squeezenet'

    def __init__(self, num_classes=1000, weight_decay=0.0, batch_norm_decay=0.9):
        self._num_classes = num_classes
        self._weight_decay = weight_decay
        self._batch_norm_decay = batch_norm_decay
        self._is_built = False

    def build(self, x, is_training):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(is_training,
                                      self._weight_decay,
                                      self._batch_norm_decay)):
                return self._squeezenet(x, self._num_classes)

    @staticmethod
    def _squeezenet(images, num_classes=1000):

        images = tf.reshape(images, [-1,224,224,3])

        net = conv2d(images, 64, [3, 3], stride=2, scope='conv1')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
        net = fire_module(net, 16, 64, scope='fire2')
        net_fire3 = fire_module(net, 16, 64, scope='fire3')
       # net = tf.add(net,net_fire3, name = "connect1")  ############
        net = max_pool2d(net_fire3, [3, 3], stride=2, scope='maxpool3')
        net = fire_module(net, 32, 128, scope='fire4')
        net_fire5 = fire_module(net, 32, 128, scope='fire5')
       # net = tf.add(net,net_fire5,name="connect2")   #################
        net = max_pool2d(net_fire5, [3, 3], stride=2, scope='maxpool5')
        net = fire_module(net, 48, 192, scope='fire6')
        net_fire7 = fire_module(net, 48, 192, scope='fire7')
        #net = tf.add(net,net_fire7,name="connect3")   ###############
        net = fire_module(net_fire7, 64, 256, scope='fire8')
        net_fire9 = fire_module(net, 64, 256, scope='fire9')
        #net = tf.add(net,net_fire9,name="connect4")  #################
        net = conv2d(net_fire9, num_classes, [1, 1], stride=1, scope='conv10')
        net = avg_pool2d(net, [13, 13], stride=1, scope='avgpool10')
        # logits = tf.squeeze(net, [2], name='logits')
        logits = avg_pool2d(net_fire7, [3,3], stride=1, scope='avgpool10')
        return logits

def _arg_scope(is_training, weight_decay, bn_decay):
    with arg_scope([conv2d],
                    weights_initializer = xavier_initializer(uniform=False),
                   weights_regularizer=l2_regularizer(weight_decay),
                   biases_initializer = tf.constant_initializer(0),
                   # normalizer_fn=batch_norm,
                   # normalizer_params={'is_training': is_training,
                   #                    'fused': True,
                   #                    'decay': bn_decay},
                                     trainable = is_training):
        with arg_scope([conv2d, avg_pool2d, max_pool2d, batch_norm],
                       data_format='NHWC') as sc:
                return sc

def load_params(sess,params_path):
    with tf.variable_scope("squeezenet", reuse = True):
        params_dict = np.load(params_path, encoding = 'bytes').item()

        fire_layers = np.array(["fire2", "fire3", "fire4", "fire5", "fire6", "fire7", "fire8", "fire9"])
        for layer_name in fire_layers:
            with tf.variable_scope(layer_name, reuse = True):
                with tf.variable_scope("squeeze", reuse = True):
                    for data in params_dict[layer_name+'_s']:
                        if len(data.shape) == 3:
                            var = tf.get_variable('biases')
                            sess.run(var.assign(np.squeeze(data)))
                        else:
                            var = tf.get_variable('weights')
                            sess.run(var.assign(data))

                with tf.variable_scope("expand/1x1", reuse = True):
                    for data in params_dict[layer_name+'_e1']:
                        if len(data.shape) == 3:
                            var = tf.get_variable('biases')
                            sess.run(var.assign(np.squeeze(data)))
                        else:
                            var = tf.get_variable('weights')
                            sess.run(var.assign(data))

                with tf.variable_scope("expand/3x3", reuse = True):
                    for data in params_dict[layer_name+'_e3']:
                        if len(data.shape) == 3:
                            var = tf.get_variable('biases')
                            sess.run(var.assign(np.squeeze(data)))
                        else:
                            var = tf.get_variable('weights')
                            sess.run(var.assign(data))

        conv_layers = np.array(["conv1","conv10"])
        for layer_name in conv_layers:
            with tf.variable_scope(layer_name, reuse=True):
                for data in params_dict[layer_name]:
                    if len(data.shape) == 3:
                        var = tf.get_variable('biases')
                        sess.run(var.assign(np.squeeze(data)))
                    else:
                        var = tf.get_variable('weights')
                        sess.run(var.assign(data))

    print("loading parameters successful")


'''
Network in Network: https://arxiv.org/abs/1312.4400
See Section 3.2 for global average pooling
'''
