import os
from keras.preprocessing import image
import numpy as np
import pdb

def read_image(filename, target_size):
	img = image.load_img(filename, target_size=target_size)
	img = image.img_to_array(img)
	return img.reshape((1,)+img.shape)

def concatenate_images(images, labels, img, label):
	if images is not None:
		images = np.concatenate((images, img))
		labels = np.concatenate((labels, label))
	else:
		images = img
		labels = label

	return images, labels

def read_caltech256(root, target_size):
	train_images = None
	train_labels = None
	test_images = None
	test_labels = None

	i = -1

	for dirname, subdirs, fileList in os.walk(root):
		#we should find 256 directories inside root
		print "Found directory", dirname
		
		if "clutter" in dirname:
			continue

		if len(fileList) >= 20:
			np.random.shuffle(fileList)
			i += 1
		else:
			continue

		#get training and test images
		for j in range(0, 20):
			filename = fileList[j]
			if filename.endswith(".jpg") or filename.endswith(".jpeg"): 
		        # store in image matrix
				img = read_image(os.path.join(dirname,filename), target_size)
				label = np.array([i])
				
				if j < 16:
					train_images, train_labels = concatenate_images(train_images, train_labels, img, label)
					print "training data", i, j, train_images.shape, img.shape
				else:
					test_images, test_labels = concatenate_images(test_images, test_labels, img, label)
					print "test data", i, j, test_images.shape, img.shape

		
	return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
	target_size=(224, 224)
	train_images, train_labels, test_images, test_labels = read_caltech256('../data/256_ObjectCategories', target_size)

	#save
	np.save('../data/train_images.npy', train_images)
	np.save('../data/train_labels.npy', train_labels)
	np.save('../data/test_images.npy', test_images)
	np.save('../data/test_labels.npy', test_labels)