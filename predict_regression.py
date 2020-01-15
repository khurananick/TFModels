import sys
import json
import numpy as np
import pickle
import make_keras_picklable as mkp

model_file = sys.argv[1]
uinputs    = json.loads(sys.argv[2])  # user inputs
finputs    = [] # formatted inputs

for uinput in uinputs:
  finputs.append([float(i) for i in uinput])

mkp.make_keras_picklable()

def baseline_model():
    model = Sequential()
    model.add(Dense((num_columns*12), input_dim=num_columns, kernel_initializer='normal', activation='relu'))
    model.add(Dense((num_columns*10), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

estimator = pickle.loads(open(model_file,"rb").read())
predictions = []

for finput in finputs:
    npa = np.array([finput])
    predictions.append(estimator.predict(npa).item(0))

print(json.dumps(predictions))
