import sys
# append the path where the slim folder is located in your system
sys.path.append("/usr/local/lib/python3.5/dist-packages/tensorflow/contrib/slim/python/slim")

import tensorflow as tf
import squeezenet as net
import orient_layer_train as orient
from data_input import ImageDataGen
import mean_shift_layer as msl
import numpy as np
import time
import os
import cv2
import pandas as pan

from tensorflow.contrib import slim
from tensorflow.contrib.framework import arg_scope
from nets import resnet_v1

BATCH_SIZE = 32
MAX_STEPS = 1000000
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_SAMPLES = 2000
NUM_EPOCHS = 64
LOG_DEVICE_PLACEMENT = False
DISPLAY_STEP = 5
NUM_ANGLES = 8 # N value from paper
NUM_START_ANGLES = 9 # M value from paper
LEARNING_RATE = 0.000001
TRAIN = False
MAX_TO_KEEP = 4 # change this variable to mention the max number of latest checkpoints to keep while training
BASE_NETWORK = "resnet" # use = "squeezenet" to use squeezenet as base network
BASE_NETWORK_IS_TRAIN = False # make it true to finetune even the base network's weights.

'''
check out the below path variables for givng the checkpoint and summary path

checkpoint_path: the folder to store the trained checkpoints
restore_checkpoint_path: folder from where the checkpoint should be restored to validate the network
filewriter_path: location to store the tensorflow summaries
params_path: location to the base network's parameters file in .npy format to initialze the weights of base network_out
			I used it for loading squeezenet parameters. For resnet I used the parameters from tensorflow slim
'''

# restore_checkpoint_path = "/home/results/checkpoints/run5"
# filewriter_path = "/home/results/summaries/run5"
# checkpoint_path = "/home/results/checkpoints/run5"
# params_path = "squeezenet_v1.1_params.npy"

if not os.path.isdir(checkpoint_path):
	os.mkdir(checkpoint_path)

def run():

	global_step = tf.Variable(0,name="global_step", trainable=False)

	##### creating object for data generation
	train_data_gen = ImageDataGen(NUM_SAMPLES, BATCH_SIZE, NUM_START_ANGLES,NUM_ANGLES, mode='training')
	validate_data_gen = ImageDataGen(NUM_SAMPLES, BATCH_SIZE, NUM_START_ANGLES,NUM_ANGLES, mode='validation')

	iterator = tf.data.Iterator.from_structure(train_data_gen.data.output_types,train_data_gen.data.output_shapes)

	training_init_op = iterator.make_initializer(train_data_gen.data)
	valid_init_op = iterator.make_initializer(validate_data_gen.data)

	(current_data_batch, current_target_batch, angles) = iterator.get_next()

	if BASE_NETWORK == "squeezenet":
		squeezenet_model = net.Squeezenet()

	name = "object_orient_layer{}"
	orient_layer_obj = orient.ObjectOrientLayer(num_classes = NUM_ANGLES*NUM_START_ANGLES, name = "orient_layer_obj")

	##############################
	current_data_batch = tf.reshape(current_data_batch, [-1,224,224,3])

	if BASE_NETWORK == "resnet":
		with slim.arg_scope(resnet_v1.resnet_arg_scope()):
			 # use is_training = True for finetuning even the resnet weights or it just trains the orient layer weights
		 	logits,_ = resnet_v1.resnet_v1_101(current_data_batch, is_training=BASE_NETWORK_IS_TRAIN)
		prob = tf.nn.softmax(logits)

		# get model weights for resnet v1 with 101 layers. comment this if you want to train base net from scratch
		init_fn = slim.assign_from_checkpoint_fn('resnet_v1_101.ckpt', slim.get_model_variables('resnet_v1_101'))

	###############################
	if BASE_NETWORK == "resnet":
		network_out = tf.get_default_graph().get_tensor_by_name("resnet_v1_101/block3/unit_15/bottleneck_v1/Relu:0")
	elif BASE_NETWORK == "squeezenet":
		squeeze_op = squeezenet_model.build(current_data_batch, is_training = BASE_NETWORK_IS_TRAIN)

	total_loss = 0
	sofmax_name = "orient_softmax{}"
	loss_name = "loss_orient{}"

	orient_layer_logits = []
	orient_layer_loss = []

	combined_orient_layer_logits = orient_layer_obj.build_layer(network_out)

	for i in range(NUM_START_ANGLES):
		orient_layer_logits.append(combined_orient_layer_logits[:,NUM_ANGLES*i:(NUM_ANGLES*i + NUM_ANGLES)])

	for i in range(NUM_START_ANGLES):
		orient_layer_loss.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
									labels = current_target_batch[:,i],logits = orient_layer_logits[i], name = sofmax_name.format(i))))
		tf.summary.scalar(loss_name.format(i), orient_layer_loss[i])

	total_loss = tf.add_n(orient_layer_loss[:],"total_loss")

	tf.summary.scalar(total_loss.op.name, total_loss)

	batches_per_epoch = np.floor(NUM_SAMPLES/BATCH_SIZE).astype(np.int16)
	lr = tf.train.exponential_decay(LEARNING_RATE, global_step, batches_per_epoch*32, 0.1, staircase=True)

	with tf.variable_scope("train"):
		optimizer = tf.train.MomentumOptimizer(lr, 0.9)####OPTIMIZER###
		grads = optimizer.compute_gradients(total_loss)#, var_list=var_list)
		apply_grads_op = optimizer.apply_gradients(grads, global_step=global_step)

		for grad, var in grads:
			if grad is not None:
				tf.summary.histogram(var.op.name + '/gradients', grad)

		# with tf.control_dependencies([apply_grads_op]):
		# 	vars_mov_avg_obj = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name = "vars_mov_avg_obj")
		# 	var_mov_avg_op = vars_mov_avg_obj.apply(tf.trainable_variables())

	with tf.control_dependencies(orient_layer_loss[:]):
		with tf.control_dependencies([total_loss]):
			train_op = apply_grads_op

	with tf.variable_scope("validation"):
		# logits_validate = []
		crct_pred = []
		accuracy = []
		for i in range(NUM_START_ANGLES):
			pred = tf.equal(tf.argmax(orient_layer_logits[i], 1), tf.cast(current_target_batch[:,i], tf.int64))
			acc = tf.reduce_mean(tf.cast(pred,tf.float32))
			crct_pred.append(tf.reduce_sum(tf.cast(pred,tf.int32)))
			accuracy.append(acc)

		# logits_validate = tf.nn.softmax(orient_layer_logits[:])
		# logits_validate = tf.squeeze(logits_validate)


	# for variable in tf.trainable_variables():
	# 	variable_name = variable.name
	# 	print("parameter: %s:\n" %(variable_name), variable.shape)
	# 	# print(variable.shape)
	# 	tf.summary.histogram(variable.name, variable)

	merged_summary = tf.summary.merge_all()

	writer = tf.summary.FileWriter(filewriter_path + '/train')
	saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)
	with tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT)) as sess:
		sess.run(tf.global_variables_initializer())
		writer.add_graph(sess.graph)

		if BASE_NETWORK == "squeezenet":
			net.load_params(sess,params_path) # uncomment this to train the squeezent form scratch
		init_fn(sess)

		# training
		if TRAIN == True:
			for epoch in range(NUM_EPOCHS):
				print("starting epoch : ", epoch+1)
				step = 1
				sess.run(training_init_op)

				while  step < batches_per_epoch:
					_,c = sess.run([train_op,total_loss])

					if step % DISPLAY_STEP == 0:
						s = sess.run(merged_summary)
						writer.add_summary(s, epoch*batches_per_epoch + step)
					if step%5 == 0:
						#print("data_batch:",sess.run(), "\n target_batch:", sess.run(target_batch), "\nlogits:",sess.run(logits))
						pass
					step += 1

				print("epoch %d completed with total_loss %f" % (epoch+1, c))

				#######save checkpoint#######

				if epoch % 5 == 0:
					checkpoint_name = os.path.join(checkpoint_path, 'train_checkpoint_epoch{}.ckpt'.format(epoch+1))
					save_path = saver.save(sess, checkpoint_name)
					print("checkpoint for epoch %d saved" % (epoch+1))

		# validating
		else:
			# change the chekpoint filename as per your requirement
			saver.restore(sess,os.path.join(restore_checkpoint_path, 'train/train_epoch64.ckpt'))
			print("done restoring for validation")
			sess.run(valid_init_op)
			step = 1
			total_accuracy = []
			error = []
			while step < int(200/BATCH_SIZE):
				print("step %d" %(step))
				prob,angle = sess.run([logits_validate,angles])
				# prob = np.squeeze(np.array(prob))
				estimated_angle = msl.mean_shift_layer(2,prob,0.1)
				error.append(np.abs(estimated_angle-angle))

				total_accuracy.append(sess.run([accuracy]))
				print("accuracy:",total_accuracy[step-1])

				step += 1
			print("average mean error:", np.mean(error))

			total_accuracy = np.mean(total_accuracy)
			print("total_accuracy:", total_accuracy)

if __name__ == '__main__':
	run()
