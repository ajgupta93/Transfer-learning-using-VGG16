import numpy as np
import math
import utilities as util
import custom_model as model
from keras.callbacks import History
from experiments_model import *
import gc, os
import pdb

os.system('ulimit -s unlimited')


def pick_n_examples(train_images, train_labels, test_images, test_labels, n):
	relevant_examples = []
	for i in range(0, len(train_labels), 16):
		for j in range(i, i+n):
			relevant_examples.append(j)


	train_images = train_images[relevant_examples]
	train_labels = train_labels[relevant_examples]

	#for test
	# test_n = int(math.ceil(n*20.0/100.0))
	# relevant_examples = []
	# for i in range(0, len(test_labels), 4):
	# 	for j in range(i, i+test_n):
	# 		relevant_examples.append(j)


	# test_images = test_images[relevant_examples]
	# test_labels = test_labels[relevant_examples]

	return train_images, train_labels, test_images, test_labels


def load_data(output_dim, example_class):
	train_images = np.load('../data/train_images.npy')
	train_labels = np.load('../data/train_labels.npy')
	test_images = np.load('../data/test_images.npy')
	test_labels = np.load('../data/test_labels.npy')

	#pick only some examples per class
	train_images, train_labels, test_images, test_labels = pick_n_examples(train_images, train_labels, test_images, test_labels, example_class)
	
	#preprocess
	train_images = util.vgg_preprocessing(train_images)
	train_labels = util.one_hot_encoding(train_labels, output_dim)
	test_images = util.vgg_preprocessing(test_images)
	test_labels = util.one_hot_encoding(test_labels, output_dim)


	return train_images, train_labels, test_images, test_labels


	

if __name__ == '__main__':
	#Output dim for our dataset
	output_dim = 256 #For Caltech256
	
	#load data
	train_images, train_labels, test_images, test_labels = load_data(output_dim, 4)
	
	#convoltion model
	#Experiments with last convolutional layer (softmax is attached after so as to reduce trainable parameters)
	#convolution_model(output_dim, "block5_pool", train_images, train_labels, test_images, test_labels)

	#Experiments with middle convolutional layer (softmax is attached after so as to reduce trainable parameters)
	#convolution_model(output_dim, "block3_pool", train_images, train_labels, test_images, test_labels)


	#fully connected model
	#fc_model(output_dim, train_images, train_labels, test_images, test_labels)

	#temperature based model
	temperature_model(output_dim, train_images, train_labels, test_images, test_labels)