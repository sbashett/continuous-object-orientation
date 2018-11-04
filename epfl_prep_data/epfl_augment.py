import numpy as np
import os
import cv2
import pandas as pan
import random

import numpy as np
import os
import cv2

save_data_path = "epfl_train_augmented"

csv_load = np.array(pan.read_csv("data.csv"))
csv_save = open("data_train.csv","w")
total_count = 1103

trains = []

count = 0
for i in range(total_count):#, csv_load.shape[0]):
	img = cv2.imread(csv_load[i][0])
	img_vert_flip = cv2.flip(img,0)

	img_name = os.path.join(save_data_path, "IMG%05d.jpg" %(count))
	cv2.imwrite(img_name, img)
	trains.append((img_name, csv_load[i][1]))
	count += 1

	img_name = os.path.join(save_data_path, "IMG%05d.jpg" %(count))
	cv2.imwrite(img_name, img_vert_flip)
	trains.append((img_name, csv_load[i][1]))
	count += 1

random.shuffle(trains)
for (img_name, angle) in trains:
	csv_save.write("%s,%f\n" % (img_name,angle))

# cv2.imshow("orig", img)
# cv2.imshow("flipped",img_vert_flip)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



csv_save.close()
