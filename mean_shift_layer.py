from __future__ import division
import numpy as np
import math


###### ASSUME WEIGHTED_VOTES IS A VX2 ARRAY WITH V=MxN ANGLES AND THEIR RESPECTIVE PROB
def mean_shift_layer(v,weighted_votes, resol):
	theta = weighted_votes[0][np.argmax(weighted_votes[1])]
	count = 0

	while True:
		mean_shift = calc_mean_shift(theta, v, weighted_votes)
		if abs(mean_shift) <= resol:
			break
		else:
			theta += mean_shift
			count += 1

		if count > 100:
			print("count value exceeded limit")
			print(mean_shift, theta)
			print(calc_mean_shift(theta, v, weighted_votes))
			break

	print(mean_shift)
	print("estimated theta:", theta)
	print("num of iterations:", count)

	return theta


def calc_mean_shift(theta, v, weighted_votes):
	num = 0
	den = 0
	# for i in range(weighted_votes.shape[0]):
	gval_num = np.reshape( v * np.sin(np.abs(theta-weighted_votes[0,:])) * np.exp(v * np.cos(np.abs(theta-weighted_votes[0,:]))), (1,72))
	# print("gval_num:",gval_num)
	gval_den = np.reshape(2 * np.abs(theta-weighted_votes[0,:]), (1,72))
	# print("gval_den:",gval_den)
	gval = np.divide(gval_num, gval_den)
	gval[np.isnan(gval)] = 0
	print("gval:", gval)
	num = np.multiply(weighted_votes[1][:], weighted_votes[0][:])
	num = np.multiply(num, gval)
	den = np.multiply(weighted_votes[1][:], gval)

	mean_shift = np.divide(np.sum(num),np.sum(den))
	mean_shift -= theta

	return mean_shift
