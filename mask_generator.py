import cv2
import os
 
data_folder = "/Users/harsh/Downloads/niflr/appy-png/"
mask_folder = "/Users/harsh/Downloads/niflr/appy-mask/"

ls = list((os.popen("ls "+data_folder).read()).split("\n"))

s = 512

try:
	for i in range(len(ls)):
		mask = cv2.imread(data_folder+ls[i],0)
		mask = cv2.resize(mask,(s,s))
		for m in range(s):
			for n in range(s):
				mask[m][n] = 255 if mask[m][n]<200 else 0
		cv2.imwrite(mask_folder+'mask_'+ls[i],mask)
except(cv2.error):
	{}

