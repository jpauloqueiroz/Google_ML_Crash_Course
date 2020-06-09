# from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow import keras

from matplotlib import pyplot as plt 

# Adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.3f}".format
# tf.keras.backend.set_floatx('float32')


# Load the dataset.
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
# Shuffles the training examples.
train_df = train_df.reindex(np.random.permutation(train_df.index))


# Calculate the Z-scores of each column in the training set and write those Z-scores into a new pandas DataFrame named train_df_norm.
train_df_mean = train_df.mean() # Get the mean of train_df
train_df_std = train_df.std() # Get the standart deviation of train_df.
train_df_norm = (train_df - train_df_mean) / train_df_std # Calculate the Z-scores.

# Calculate the Z-scores of each column in the test set and write those Z-scores into a new pandas DataFrame named test_df_norm.
test_df_mean = test_df.mean() # Get the mean of test_df.
test_df_std = test_df.std() # Get the standart deviation of test_df.
test_df_norm = (test_df - test_df_mean) / test_df_std # Calculate the Z-scores.


threshold = 265000 # This is the 75th percentile for the median house values.
train_df_norm ["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float)
test_df_norm ["median_house_value_is_high"] = (test_df["median_house_value"] > threshold).astype(float)


""" Implement feature crosses. """
# Create an empty list that will eventually hold all feature columns.
feature_column = []

# Create a numerical feature column to represent median_income.
median_income = tf.feature_column.numeric_column("median_income")
feature_column.append(median_income)

# Create a numerical feature column to represent total_rooms.
total_rooms = tf.feature_column.numeric_column("total_rooms")
feature_column.append(total_rooms)

# Convert the list of feature columns into a layer that will ultimately become part of the model.
feature_layer = keras.layers.DenseFeatures(feature_column)

""" Define functions to create and train a model, and a plotting function. """
def create_model(my_learning_rate, feature_layer, my_metrics):
    """ Create and compile a simple linear regression model. """
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(feature_layer)

    # Add one linear layer to the model to yield a simple linear regression.
    model.add(tf.keras.layers.Dense(units = 1, input_shape = (1,),
                                    activation =  tf.sigmoid),)

    # Construct the layers into a model that TensorFlow can execute.
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = my_learning_rate),
                  loss = tf.keras.losses.BinaryCrossentropy(),
                  metrics = my_metrics)

    return model

def train_model(model, dataset, epochs, batch_size, label_name):
    """ Feed a dataset into the model in order to train it. """
    # Define Features
    features = {name:np.array(value) for name, value in dataset.items()}
    # Define Label
    label = np.array(features.pop(label_name))
    # Train
    history = model.fit(x = features, y = label, batch_size = batch_size, epochs = epochs, shuffle = True)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    
    return epochs, hist

def plot_curve(epochs, hist, list_of_metrics):
    """ Plot a curve of one or more classification metrics vs. epoch. """  

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label = m)

    plt.legend()

    plt.show()

# Hyperparameters.
learning_rate = 0.001
epochs = 20
batch_size = 100
classification_threshold = 0.35
label_name = "median_house_value_is_high"

# Establish the metrics the model will measure.
METRICS = [tf.keras.metrics.BinaryAccuracy(name = "accuracy", threshold = classification_threshold),
           tf.keras.metrics.Precision(name = "precision", thresholds = classification_threshold),
           tf.keras.metrics.Recall(name = "recall", thresholds = classification_threshold),
           tf.keras.metrics.AUC(num_thresholds = 100, name = 'auc')]

# Establish the model's topography.
my_model = create_model(learning_rate, feature_layer, METRICS)

# Train the model on the training set.
epochs, hist = train_model(my_model, train_df_norm, epochs, batch_size, label_name)

# Plot a graph of the metric(s) vs. epochs.
list_of_metrics_to_plot = ["accuracy", "precision", "recall", "auc"]
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate the model against the test set.
test_features = {name:np.array(value) for name, value in test_df_norm.items()}
test_labels = np.array(test_features.pop(label_name))
my_model.evaluate(x = test_features, y = test_labels, batch_size = batch_size)

