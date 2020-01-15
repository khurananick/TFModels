# things we need in general
import sys
import pickle
import json
import ijson.backends.yajl2 as ijson
# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# things we need for Tensorflow
import os
import numpy as np
import tflearn
import tensorflow as tf
import random

#get all the inputs
training_input    = sys.argv[1]
training_logs     = sys.argv[2]
model_output      = sys.argv[3]
training_data_file= sys.argv[4]
words_file        = str(training_data_file) + ".words"
classes_file      = str(training_data_file) + ".classes"
documents_file    = str(training_data_file) + ".documents"
training_text_file= str(training_data_file) + ".txt"

# helper methods
def load_json(filepath):
    return json.load(open(filepath, "r"))

def save_json(data, filepath):
    json.dump(data, open(filepath, "w" ))

def load_ijson(filepath, itempath):
    return ijson.items(open(filepath, "rb"), itempath)

def append_file(filepath, lines):
    f = open(filepath, "a")
    for line in lines:
        if line is not None:
            f.write(json.dumps(line) + "\n")
    f.close()

words = []
classes = []
documents = []
ignore_words = ['?']

# check if there is an existing json file with data
if not os.path.isfile(words_file):
    # import our intents file
    intents = load_ijson(training_input, 'intents.item')
    # loop through each sentence in our intents patterns to get words and classes
    for intent in intents:
        while intent['patterns']:
            pattern = intent['patterns'].pop(0)
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    # dump it all to a file and release the variables.
    save_json(words, words_file)
    save_json(classes, classes_file)

words = None
classes = None
intents = None

if not os.path.isfile(documents_file):
    # import our intents file
    intents = load_ijson(training_input, 'intents.item')
    # loop through each sentence in our intents patterns to get documents
    for intent in intents:
        while intent['patterns']:
            pattern = intent['patterns'].pop(0)
            w = nltk.word_tokenize(pattern)
            documents.append((w, intent['tag']))
    # dump it all to a file and release the variables.
    intents = None
    save_json(documents, documents_file)

intents = None
documents = None

# stem and lower each word and remove duplicates
words = load_ijson(words_file, 'item')
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
save_json(words, words_file)
words = None

# remove duplicates
classes = load_json(classes_file)
classes = sorted(list(set(classes)))
save_json(classes, classes_file)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# creates training data from documents and words arrays in batches of 10
def bag_em_up():
    def do_bagging(documents):
        for doc in documents:
            words = load_ijson(words_file, 'item')
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            # append to training array
            return [bag, output_row]

    documents = load_ijson(documents_file, 'item')
    worklist = range(2000000)
    batchsize = 200
    for i in range(0, len(worklist), batchsize):
        batch = worklist[i:i+batchsize]
        new_batch = []
        for b in batch:
            new_batch.append(do_bagging(documents))
        if len(new_batch) == 0:
            return
        append_file(training_text_file, new_batch)

if not os.path.isfile(training_data_file):
    # training set, bag of words for each sentence
    if not os.path.isfile(training_text_file):
        bag_em_up()

    with open(training_text_file) as f:
        for line in f:
            try:
                training.append(json.loads(line))
            except:
                print("Error loading json.")

    words = None
    documents = None

    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)

    # create train and test lists
    train_x = list(training[:,0])
    train_y = list(training[:,1])

    # save all of our data structures
    save_json({'train_x':train_x, 'train_y':train_y}, training_data_file)
else:
    data    = load_json(training_data_file)
    train_x = data.pop('train_x', [])
    train_y = data.pop('train_y', [])
    data    = None

classes = None

# Reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir=(training_logs))
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=32, show_metric=True)
#score = model.evaluate(train_x, train_y)
model.save(model_output)
print("Done")
