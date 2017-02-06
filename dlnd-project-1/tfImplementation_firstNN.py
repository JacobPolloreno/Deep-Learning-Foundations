import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn

# Load data
data_path = 'bike-sharing-dataset/hour.csv'
rides = pd.read_csv(data_path)

# Preprocessing

# Create dummie variables for categorical variables
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

# Drop fields
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)

# Standardize Variables
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum',
                  'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean) / std

# Train/Test Split
test_data = data[-21 * 24:]
data = data[:-21 * 24]

# Seperate data into features and targets
#   target variable is 'cnt'; drop rest
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1),\
    data['cnt']
test_features, test_targets = test_data.drop(target_fields, axis=1), \
    test_data['cnt']

# Tranform test_features and test_targets into NP array
test_features = test_features.values
test_targets = np.array(test_targets, ndmin=2).T

# Split training data into train and validation sets
# Validation set is for hyperparameter tuning
n_records = features.shape[0]
split = np.random.choice(features.index, size=int(n_records * 0.8),
                         replace=False)
train_features, train_targets = features.ix[split], targets.ix[split]
val_features, val_targets = features.drop(split), targets.drop(split)

# Parameters
train_records = train_features.shape[0]
input_nodes = train_features.shape[1]
output_nodes = 1

training_epoch = 110
hidden_nodes = 17
learning_rate = 0.3
batch_size = 128
display_step = 10

# Convert features and targets into tensor variables
X = tf.placeholder(tf.float32, [None, input_nodes])
Y = tf.placeholder(tf.float32, [None, output_nodes])

# Set model weights
weights = {
    'h1': tf.Variable(tf.random_normal([input_nodes, hidden_nodes])),
    'out': tf.Variable(tf.random_normal([hidden_nodes, output_nodes]))
}

# Convert features and test list to 2d NP arrays
train_features = train_features.values
train_targets = np.array(train_targets, ndmin=2).T
val_features = val_features.values
val_targets = np.array(val_targets, ndmin=2).T

# Create model


def multilayer_perceptron(x, weights):
    # Hidden layer with Sigmoid activation
    layer_1 = tf.matmul(x, weights['h1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out'])

    return out_layer


# Construct model
pred = multilayer_perceptron(X, weights)

# Define loss and optimizer
cost = tf.reduce_mean(tf.square(pred - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Track Cost function for training and validation sets
    losses = {'train': [], 'validation': []}

    # Fit training data
    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(train_records / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch = np.random.choice(range(train_records), size=128)
            batch_x = train_features[batch][:]
            batch_y = train_targets[batch][:]
            # Run optimization backprop and get loss value(cost)
            _, c = sess.run([optimizer, cost], feed_dict={
                            X: batch_x, Y: batch_y})

            avg_cost += c / total_batch

        # Display loss per epoch cycle
        if(epoch + 1) % display_step == 0:
            # Calculate losses for training and test sets
            train_loss = sess.run(
                cost, feed_dict={
                    X: train_features, Y: train_targets})
            val_loss = sess.run(
                cost, feed_dict={
                    X: val_features, Y: val_targets})
            losses['train'].append(train_loss)
            losses['validation'].append(val_loss)

            # Print out the losses as the network is training
            print("Epoch:", '%04d' % (epoch + 1), "Avg loss =",
                  "{:.9f}".format(avg_cost))
            print("\tTrain Loss=", "{:.9f}".format(train_loss))
            print("\tVal Loss=", "{:.9f}".format(val_loss))

    print("Optimization Finished\n")

    training_cost = sess.run(
        cost,
        feed_dict={
            X: train_features,
            Y: train_targets})
    print("Training Loss=", training_cost)

    # Test example
    testing_cost = sess.run(
        cost,
        feed_dict={
            X: test_features,
            Y: test_targets})
    print("Testing Loss=", testing_cost)

    # Learning curve
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.plot(losses['train'][1:], label='Training loss')
    plt.plot(losses['validation'][1:], label='Validation loss')
    plt.legend()
    plt.show()

    # Testing Example Graph
    fig, ax = plt.subplots(figsize=(10, 4))

    mean, std = scaled_features['cnt']
    predictions = sess.run(pred, feed_dict={X: test_features})
    predictions = predictions * std + mean
    ax.plot(predictions, label='Prediction')
    ax.plot(test_targets * std + mean, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)
    plt.show()
