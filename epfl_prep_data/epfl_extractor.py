import numpy as np
import os
import cv2

orig_data_path = "epfl-gims08/tripod-seq"
save_data_path = "epfl"

orig_fid = open(os.path.join(orig_data_path , "tripod-seq.txt"), "r")

lines = orig_fid.readlines()

num_sequences, width, height = [int(iter) for iter in lines[0].split()]

target_width = 224
target_height = 224

num_images = [int(iter) for iter in lines[4].split()]
posn_front = [int(iter) for iter in lines[5].split()]
rot_dir = [int(iter) for iter in lines[6].split()]

csv_file = open("data.csv","w")
for i in range(num_sequences):
	box_fid = open(os.path.join(orig_data_path, "bbox_%02d.txt" % (i+1)), "r")


	########calculating angles##############

	rot_per_img = 360.0/(float)(num_images[i])
	frst_img_angle = rot_dir[i]* (posn_front[i]-1) * rot_per_img
	if(frst_img_angle < 0):
		frst_img_angle = 360 - abs(frst_img_angle)

	if(rot_dir[i] < 0):
		angles = list(np.linspace(frst_img_angle, 360-rot_per_img, posn_front[i]-1)) + list(np.arange(0, frst_img_angle, rot_per_img))
	else:
		angles = list(np.arange(frst_img_angle, 0, -rot_per_img)) + list([0]) + list(np.arange(360-rot_per_img, frst_img_angle - rot_per_img, -rot_per_img))

	angles = angles[0:num_images[i]]
	##########################################
	
	for j in range(num_images[i]):
		img = cv2.imread(os.path.join(orig_data_path, "tripod_seq_%02d_%03d.jpg" % (i+1, j+1)))
		temp = [float(iter) for iter in box_fid.readline().split()]
		temp[1] -= 0.1*temp[1]
		temp[0] -= 0.1*temp[0]
		temp[2] *= 1.2
		temp[3] *= 1.2
		temp = [int(x) for x in temp]

		crop_img = img[temp[1]:(temp[1]+temp[3]), temp[0]:(temp[0] + temp[2])]
		save_img = cv2.resize(crop_img, dsize = (target_width,target_height))

		cv2.imwrite(os.path.join(save_data_path, "tripod_seq_%02d_%03d.jpg" % (i+1, j+1)), save_img)

		csv_file.write("epfl/tripod_seq_%02d_%03d.jpg,%f\n" % (i+1, j+1, angles[j]))
	box_fid.close()

csv_file.close()
orig_fid.close()




