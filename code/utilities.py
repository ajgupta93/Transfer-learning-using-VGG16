import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils.np_utils import *
from keras.callbacks import *
from scipy.misc import imresize
from scipy.misc import imsave
from keras.models import Model
from matplotlib import pyplot as plt
import pdb

def get_early_stopping_callback():
	return EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=3, verbose=0, mode='auto')


def save_all_activations(array, size_percentage, filepath):
	for i in range(0, len(array)):
		filename = filepath+str(i)+".png"
		print filename
		imsave(filename, imresize(array[i], size_percentage))


def visualize_filters(model, input_img):
	imsave("input_image.png", np.squeeze(input_img))
	
	#first convolution layer 
	first_conv_model = Model(input=model.input, output=model.get_layer('block1_conv1').output)
	first_conv_output = first_conv_model.predict(input_img)
	
	first_conv_output = np.rollaxis(np.squeeze(first_conv_output), 2)
	save_all_activations(first_conv_output, 100, "../results/first_conv/")
	
	#last convoltution layers
	last_conv_model = Model(input=model.input, output=model.get_layer('block5_conv3').output)
	last_conv_output = last_conv_model.predict(input_img)

	last_conv_output = np.rollaxis(np.squeeze(last_conv_output), 2)
	save_all_activations(last_conv_output, 800, "../results/last_conv/")
	

def subtract_mean(images):
	mean = np.mean(images) #, axis = 0)
	processed_images = images - mean

	return processed_images
		
def vgg_preprocessing(images):
	return preprocess_input(images)

def one_hot_encoding(labels, n_classes):
	return to_categorical(labels, nb_classes=n_classes)

def plot_temperature_graphs(best_T, train_acc, test_acc):
	legend = ['training accuracy', 'test accuracy']
	labels = ['epochs', 'percent accuracy']
	title = "Temperature based softmax regression: Best T = ", best_T
	
	t = np.linspace(1, len(train_acc), len(train_acc))

	plt.gca().set_color_cycle(['red', 'blue'])
	plt.plot(t, train_acc, t, test_acc)
	plt.xlabel(labels[0])
	plt.ylabel(labels[1])
	plt.title(title)
	plt.legend(legend, loc='best')
	plt.show()
