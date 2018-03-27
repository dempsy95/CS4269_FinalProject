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


def getData(): 

	
	metaData=[]

	with open('meta.txt', 'rb') as myFile:
		for line in myFile:
			line=eval(line)
			myData=[]
			myData.append(line['fileName'])
			myData.append(line['age'])
			myData.append(line['gender'])
			metaData.append(myData)

	count=0	

	DB=[]
	for sample in metaData:
		img = cv2.imread(sample[0])
		if(np.shape(img)[0]>144 and np.shape(img)[1]>144):
		# change the size of the image here
			resized_img = cv2.resize(img, (32, 32)) 
			myData=[]
			myData.append(resized_img)
			myData.extend(sample)
			count=count+1
			DB.append(myData)

		# change the number of images here
			if(count>=10000): 
				break

	
	return DB


getData()

		






