import os
import numpy as np
import struct
from matplotlib import pyplot as plt
import sys
import random
import copy
import cv2
import tables
import scipy.io
import glob
images = {}

count=0

for filename in glob.iglob('**/*.jpg',recursive=True): #assuming gif
	img = cv2.imread(filename)
	if(np.shape(img)[0]>144 and np.shape(img)[1]>144):
		# change the size of the image here
		resized_img = cv2.resize(img, (32, 32)) 
		images[filename]=resized_img
		count=count+1
		# change the number of images here
		if(count>10000): 
			break


DB=[]
with open('meta.txt', 'rb') as myFile:
	for line in myFile:
		line=eval(line)
		if(line['valid']!=0 and line['fileName'] in images):
			myData=[]
			myData.append(images[line['fileName']])
			myData.append(line['fileName'])
			myData.append(line['age'])
			myData.append(line['gender'])
			DB.append(myData)
			
print(DB)


		






