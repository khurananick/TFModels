import os
import sys
import pickle
import numpy
import pandas
import make_keras_picklable as mkp
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

training_input = sys.argv[1]
model_output   = sys.argv[2]

mkp.make_keras_picklable()

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
    model.add(Dense((num_columns*12), input_dim=num_columns, kernel_initializer='normal', activation='relu'))
    model.add(Dense((num_columns*10), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

seed = 7
numpy.random.seed(seed)

pipeline = None

if(os.path.isfile(model_output)):
        if(os.stat(model_output).st_size):
                pipeline = pickle.loads(open(model_output,"rb").read())

if(pipeline == None):
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=num_rows, batch_size=100, verbose=0)))
    pipeline = Pipeline(estimators)

pipeline.fit(features, labels)

open(model_output, "wb").write(pickle.dumps(pipeline))
print("Done")

