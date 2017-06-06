import os
import numpy as np
import pandas as pd
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense

# Load data
data_path = 'bike-sharing-dataset/hour.csv'
rides = pd.read_csv(data_path)

# Preprocessing

## Create dummie variables for categorical variables
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

## Drop fields
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',\
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)

## Standardize Variables
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', \
                  'windspeed']
### Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean) / std

## Train/Test Split
test_data = data[-21*24:]
data = data[:-21*24]

### Seperate data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1),\
        data['cnt']
test_features, test_targets = test_data.drop(target_fields, axis=1), \
        test_data['cnt']

test_features = np.array(test_features, ndmin=2)
test_targets = np.array(test_targets, ndmin=2).T

### Split training data into train and validation sets
n_records = features.shape[0]
split = np.random.choice(features.index, size=int(n_records*0.8),\
                         replace=False)
train_features, train_targets = features.ix[split], targets.ix[split]
val_features, val_targets = features.drop(split), targets.drop(split)

# Convert inputs list to 2d array
train_features = np.array(train_features, ndmin=2)
train_targets = np.array(train_targets, ndmin=2).T
val_features = np.array(val_features, ndmin = 2)
val_targets = np.array(val_targets, ndmin=2).T

#Epoch 10000/10000
#13500/13500 [==============================] - 0s - loss: 0.0473 - val_loss: 0.0578

# Hyperparameters
epoch = 250
input_nodes = train_features.shape[1]
hidden_nodes = 25
output_nodes = 1
learning_rate = 0.3

# Create NN
model = Sequential()
model.add(Dense(hidden_nodes, input_dim=input_nodes, init='uniform', activation='sigmoid'))
model.add(Dense(output_nodes, init='normal'))

# Compile Model
sgd = SGD(lr=learning_rate)
model.compile(loss="mean_squared_error", optimizer=sgd)

# Fit model
model.fit(train_features, train_targets, validation_data=(val_features, val_targets), \
          nb_epoch=epoch, batch_size=128)

# Evaluate on test
print("\nModel MSE on test set: {:.2f} ".format(model.evaluate(test_features,
                                                               test_targets)*100))

# Serialize model to JSON
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
model.save_weights("weights.h5")
print("\nSaved model to disk")
