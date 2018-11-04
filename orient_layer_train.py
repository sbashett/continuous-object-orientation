import tensorflow as tf
import numpy as np

class ObjectOrientLayer(object):

	def __init__(self,name, num_classes, trainable=True, reuse=False):
		self.name = name
		self.num_classes = num_classes
		self.trainable = trainable
		self.reuse = reuse

	def build_layer(self, x):
		with tf.variable_scope(self.name):
			# avg_pool = tf.contrib.layers.avg_pool2d(x, [3,3], stride = 1, scope = "avg_pool")

			flatten = tf.contrib.layers.flatten(x)#avg_pool)

			output = tf.contrib.layers.fully_connected(flatten, self.num_classes, reuse = self.reuse,
													trainable = self.trainable, scope = "fc_layer",
													weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
													)

			return output
