import numpy as np
import math
import utilities as util
import custom_model as model
from keras.callbacks import History
from experiments_model import *
import pdb
import os
import gc
import scipy.misc

os.system('ulimit -s unlimited')


def pick_n_examples(train_images, train_labels, test_images, test_labels, n, unique_labels):
	trImgs = []
	trLbls = []
	perClassCount = np.zeros((1,len(unique_labels)))
	
	for i in range(np.size(train_labels,axis=0)):
            curr_label_idx = np.where(train_labels[i]==1)[0][0]
            if perClassCount[0][curr_label_idx]<n:
                    perClassCount[0][curr_label_idx] += 1
                    trImgs.append(train_images[i])
                    trLbls.append(train_labels[i])
                    #fname = "./test/file_" + str(i) + ".jpg"
                    #scipy.misc.imsave(fname, train_images[i])
        train_images = np.asarray(trImgs)
        train_labels = np.asarray(trLbls)
        #print np.shape(train_labels) 

	return train_images, train_labels, test_images, test_labels


def load_data(output_dim, example_class):
	train_images = np.load('../dataUT/train_images.npy')
	train_labels = np.load('../dataUT/train_labels.npy')
	test_images = np.load('../dataUT/test_images.npy')
	test_labels = np.load('../dataUT/test_labels.npy')
	classes = np.load('../dataUT/label_decode.npy')
	#pick only some examples per class
	train_images, train_labels, test_images, test_labels = pick_n_examples(train_images, train_labels, test_images, test_labels, example_class, classes)
	#preprocess
	train_images = util.vgg_preprocessing(train_images)
	#train_labels = util.one_hot_encoding(train_labels, output_dim)
	test_images = util.vgg_preprocessing(test_images)
	#test_labels = util.one_hot_encoding(test_labels, output_dim)


	return train_images, train_labels, test_images, test_labels



if __name__ == '__main__':
	#Output dim for our dataset
	output_dim = 11 #For Urban tribes
	
	#load data
	train_images, train_labels, test_images, test_labels = load_data(output_dim, 16)
	
	#convoltion model
	#Experiments with last convolutional layer (softmax is attached after so as to reduce trainable parameters)
	#convolution_model(output_dim, "block5_pool", train_images, train_labels, test_images, test_labels)

	#Experiments with middle convolutional layer (softmax is attached after so as to reduce trainable parameters)
	#convolution_model(output_dim, "block3_pool", train_images, train_labels, test_images, test_labels)


	#fully connected model
	#fc_model(output_dim, train_images, train_labels, test_images, test_labels)
