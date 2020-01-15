import sys
import os
import pickle
from os import listdir
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

features = dict()
if(os.path.isfile(sys.argv[2])):
	if(os.stat(sys.argv[2]).st_size):
		features = pickle.loads(open(sys.argv[2],"rb").read())

def extract_features(directory):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	print(model.summary())
	for name in listdir(directory):
		image_id = name.split('.')[0]
		if(image_id not in features):
			filename = directory + '/' + name
			image = load_img(filename, target_size=(224, 224))
			image = img_to_array(image)
			image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
			image = preprocess_input(image)
			feature = model.predict(image, verbose=0)
			features[image_id] = feature
			print('>%s' % name)
	return features

directory = sys.argv[1]
features = extract_features(directory)
pickle.dump(features, open(sys.argv[2], 'wb'))
print('Extracted features.')
