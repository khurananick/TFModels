import os
import numpy
import pandas
import sys
import pickle
import make_keras_picklable as mkp
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import model_from_json

training_input = sys.argv[1]
model_output   = sys.argv[2]

mkp.make_keras_picklable()

# seed.. I don't know what it's for.
seed = 7
numpy.random.seed(seed)

# read and parse data from csv file.
df = pandas.read_csv(training_input, header=0)
feature_columns = list(df.columns.values)
label_column = feature_columns.pop()
features = df[feature_columns].values
labels_df = df[label_column]
labels = labels_df.values
num_columns = len(feature_columns)
num_classes = len(labels_df.unique())
num_rows = len(features)

def baseline_model():
    model = Sequential()
    model.add(Dense((num_columns*2), input_dim=num_columns, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = None
if(os.path.isfile(model_output)):
        if(os.stat(model_output).st_size):
                estimator = pickle.loads(open(model_output,"rb").read())

if(estimator == None):
    estimator = KerasClassifier(build_fn=baseline_model, epochs=(num_rows), batch_size=100, verbose=0)

estimator.fit(features, labels, epochs=num_rows, batch_size=100, verbose=0)

open(model_output, "wb").write(pickle.dumps(estimator))
print("Done")
