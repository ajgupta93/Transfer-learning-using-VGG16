import numpy as np
import math
import utilities as util
import custom_model as model
from keras.callbacks import History
import gc
import pdb

def train(model, train_images, train_labels):
	early_stopping = util.get_early_stopping_callback()
	hist = History()
	
	model.fit(train_images, train_labels, batch_size=16, nb_epoch=10, verbose=1, callbacks=[hist],
		validation_split=0.1, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

	print hist.history
	return hist

def test(model, test_images, test_labels):
	test_eval = model.evaluate(test_images, test_labels, batch_size=16, verbose=1)

	#predict labels
	predicted = model.predict(test_images, batch_size=16, verbose=1)

	predicted_labels = np.argmax(predicted, axis=1) 
	print predicted_labels

	return test_eval[1]


def fc_model(output_dim, train_images, train_labels, test_images, test_labels):
	caltech_model = model.get_model(output_dim)

	#visualize filters
	# img = test_images[5]
	# img = img.reshape((1,)+img.shape)
	# util.visualize_filters(caltech_model, img) 
    
    #Train the model
	train(caltech_model, train_images, train_labels)

    #Test the model
	print test(caltech_model, test_images, test_labels)


def convolution_model(output_dim, layer_name, train_images, train_labels, test_images, test_labels):
	caltech_model = model.get_convolution_model(output_dim, layer_name)

	#Train the model
	train(caltech_model, train_images, train_labels)

    #Test the model
	print test(caltech_model, test_images, test_labels)



def temperature_model(output_dim, train_images, train_labels, test_images, test_labels):
	
	T = [1.0, 2.0, 4.0, 8.0, 16.0]
	best_model = None
	best_loss = None
	best_T = 1

	train_accuracy = []
	test_accuracy = []

	for temp in T:
		print "----------------T = ", temp, "-----------------------------------------------------"
		caltech_model = model.get_temperature_model(output_dim, temp)
		#Train the model
		hist = train(caltech_model, train_images, train_labels)
		hist = hist.history

		if best_loss == None or hist['loss'][len(hist['loss'])-1] < best_loss: 
			best_loss = hist['loss'][len(hist['loss'])-1]
			best_model = caltech_model
			best_T = temp

		train_accuracy.append(hist['acc'][len(hist['loss'])-1]*100.0)

		#Test the model
		test_acc = test(best_model, test_images, test_labels)*100.0

		test_accuracy.append(test_acc)



	print "--------------------------------------------------------------------------------------"
	print "best T found", best_T

	util.plot_temperature_graphs(best_T, train_accuracy, test_accuracy)

	return best_T, train_accuracy, test_accuracy
