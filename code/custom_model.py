import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Lambda
from keras.optimizers import SGD
from keras.optimizers import RMSprop
import pdb

def get_sgd_optimizer():
	return SGD(lr=0.001, momentum=0.5, decay=0.0, nesterov=True)

def get_rmsp_optimizer():
	return RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

def get_model(output_dim):
	vgg_model = VGG16(weights='imagenet', include_top=True)

	#print vgg_model.summary()

	#now take ouput of all but last layer
	vgg_out = vgg_model.layers[-2].output

	#now lets add a softmax layer tailored for our task
	softmax_layer = Dense(output_dim, activation='softmax', name='softmax_256')(vgg_out)

	# this is the model we will train
	model = Model(input=vgg_model.input, output=softmax_layer)

	#Freeze all layers of base VGG16 model
	for layer in vgg_model.layers:
		layer.trainable = False

	# compile the model (should be done *after* setting layers to non-trainable)
	rmsp = get_rmsp_optimizer()
	model.compile(optimizer=rmsp, metrics=['accuracy'], loss='categorical_crossentropy')

	#verify that model is updated and appropiate
	#print model.summary()

	return model

def get_convolution_model(output_dim, layer_name):
	vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

	#now take ouput of all but last layer
	vgg_out = vgg_model.get_layer(layer_name).output

	vgg_out = Flatten()(vgg_out)

	#now lets add a softmax layer tailored for our task
	softmax_layer = Dense(output_dim, activation='softmax', name='softmax_256')(vgg_out)

	# this is the model we will train
	model = Model(input=vgg_model.input, output=softmax_layer)

	#Freeze all layers of base VGG16 model
	for layer in vgg_model.layers:
		layer.trainable = False

	# compile the model (should be done *after* setting layers to non-trainable)
	rmsp = get_rmsp_optimizer()
	model.compile(optimizer=rmsp, metrics=['accuracy'], loss='categorical_crossentropy')

	#verify that model is updated and appropiate
	print model.summary()

	return model

def get_temperature_model(output_dim, temperature):
	vgg_model = VGG16(weights='imagenet', include_top=True)

	#now take ouput of all but last layer
	vgg_out = vgg_model.layers[-2].output

	#now add a lambda layer for temperature variable
	vgg_out = Lambda(lambda x: x * 1.0/ temperature)(vgg_out)

	#add back the softmax layer of VGG
	vgg_out = vgg_model.layers[-1](vgg_out)

	#now lets add a softmax layer tailored for our task
	softmax_layer = Dense(output_dim, activation='softmax', name='softmax_256')(vgg_out)

	# this is the model we will train
	model = Model(input=vgg_model.input, output=softmax_layer)

	#Freeze all layers of base VGG16 model
	for layer in vgg_model.layers:
		layer.trainable = False

	# compile the model (should be done *after* setting layers to non-trainable)
	rmsp = get_rmsp_optimizer()
	model.compile(optimizer=rmsp, metrics=['accuracy'], loss='categorical_crossentropy')

	#verify that model is updated and appropiate
	print model.summary()

	return model
