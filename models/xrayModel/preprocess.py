import numpy as np
import tensorflow as tf
import os
from PIL import Image
import random
import datetime

def extract(images, labels, dirlist, namelist, prefix, dim1, dim2, stop):
	"""
	Populates the images and labels numpy arrays
	:param images: images numpy array
	:param labels: labels numpy array
	:param dirlist: a list containing lists of directory contents
	:param namelist: a list containing directory names
	:param prefix: prefix filepath
	:param dim1: the x dimension of the image to be created
	:param dim2: the y dimension of the image to be created
	:param stop: an integer indicating the position in a given dirlist to stop
	:return: the saved index positions of the next unread image in the directories
	"""
	idx = 0
	pos = 0
	pn = 0
	pp = 0
	while len(dirlist) > 0:
		i = 0
		for filename in dirlist[0]:
			i += 1
			if namelist[0] == "PNEUMONIA":
				pp += 1
			else:
				pn += 1
			if i == stop:
				break
			if filename.endswith(".jpeg"): 
				image = Image.open(prefix + "/" + namelist[0] + "/" + filename)
				image = image.resize((dim1, dim2))
				image = np.array(image).astype('float32') / 255.0
				image = np.reshape(image, (-1, 1, dim1, dim2))
				image = np.transpose(image, axes=[0,2,3,1])
				# For some reason a single image ends up size 3 on axis=0
				# The check below essentially turns it back to size 1 on axis=0
				if np.shape(image)[0]  == 3:
					image = image[0]
				label = np.array([0.0, 0.0]).astype('float32')
				label[pos] = 1.0
				labels[idx] = label
				images[idx] = image
			else:
				idx -= 1
			idx += 1
		pos += 1
		dirlist.pop(0)
		namelist.pop(0)
	return pn, pp, idx


def get_data(prefix, segment=240, positionN=0, positionP=0):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels and auxilliary values
	:param prefix: filepath
	:param segment: size of data chunk to read
	:param positionN: index to start reading from for NORMAL files
	:param positionP: index to start reading from for PNEUMONIA files
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes), two position values of saved progress
	of iteration through files, and a flag 'end' determining if dir end reached
	"""

	random.seed(datetime.time())

	stop = 0
	end = False
	directoryLisN = os.listdir(prefix + "/NORMAL")[positionN : ]
	directoryLisP = os.listdir(prefix + "/PNEUMONIA")[positionP : ]

	if (len(directoryLisN) + len(directoryLisP) - segment) <= 0 or (
		prefix[-5 : ] == "train" and (
		len(directoryLisN) < segment // 2 or len(directoryLisP) < segment // 2)):
		end = True 
	
	NUM_INPUTS = min(segment, len(directoryLisN) + len(directoryLisP))
	stop = NUM_INPUTS // 2


	images = np.zeros((NUM_INPUTS, 150, 150, 1))
	labels = np.zeros((NUM_INPUTS, 2))
	dirlist = [directoryLisP, directoryLisN]
	namelist = ["PNEUMONIA", "NORMAL"]

	pn, pp, idx = extract(images, labels, dirlist, namelist, prefix, 150, 150, stop)
	if idx == 0:
		images = np.array([])
		labels = np.array([])
	elif idx < np.shape(images)[0] - 1:
		images = images[0 : idx]
		labels = labels[0 : idx]
	positionN += pn
	positionP += pp

	return images, labels, positionN, positionP, end

