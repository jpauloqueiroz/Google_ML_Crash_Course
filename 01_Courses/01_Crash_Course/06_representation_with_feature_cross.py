import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow import keras

from matplotlib import pyplot as plt 

# Adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
tf.keras.backend.set_floatx('float32')

# Load the dataset.
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# Scale the labels.
scale_factor = 1000.0
# Scale the training set's label.
train_df["median_house_value"] /= scale_factor
# Scale the test set's label.
test_df["median_house_value"] /= scale_factor

# Shuffles the training examples
train_df = train_df.reindex(np.random.permutation(train_df.index))


""" Implement feature crosses. """
# Create an empty list that will eventually hold all feature columns.
feature_column = []

# Size of each bucket(bin).
resolution_in_degrees = 0.4

# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df["latitude"])),
                                     int(max(train_df["latitude"])),
                                     resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,latitude_boundaries)
feature_column.append(latitude)

# Create a bucket feature column for longitude
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df["longitude"])),
                                     int(max(train_df["longitude"])),
                                     resolution_in_degrees))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,longitude_boundaries)
feature_column.append(longitude)

# Convert the list of feature columns into a layer that will ultimately become part of the model.
buckets_feature_layer = keras.layers.DenseFeatures(feature_column)


# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude,longitude], hash_bucket_size = 100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_column.append(crossed_feature)

feature_cross_feature_layer = keras.layers.DenseFeatures(feature_column)

# Features using floating-point values to represent latitude and longitude.
""" 
# Create a numerical feature column to represent latitude.
latitude = tf.feature_column.numeric_column("latitude")
feature_column.append(latitude)
# Create a numerical feature column to represent longitude.
longitude = tf.feature_column.numeric_column("longitude")
feature_column.append(longitude)

# Convert the list of feature columns into a layer that will ultimately become part of the model.
fp_feature_layer = keras.layers.DenseFeatures(feature_column)
"""

""" Define functions to create and train a model, and a plotting function. """
def create_model(my_learning_rate, feature_layer):
    """ Create and compile a simple linear regression model. """
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(feature_layer)

    # Add one linear layer to the model to yield a simple linear regression.
    model.add(tf.keras.layers.Dense(units = 1, input_shape = (1,)))

    # Construct the layers into a model that TensorFlow can execute.
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = my_learning_rate),
                  loss = "mean_squared_error",
                  metrics = [tf.keras.metrics.RootMeanSquaredError()])

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
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse

def plot_the_loss_curve(epochs, rmse):
    """ Plot the curve of loss vs. epoch. """

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root mean Squared Error")

    plt.plot(epochs, rmse, label = "Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.94, rmse.max() * 1.05])
    plt.show()


# Hyperparameters.
learning_rate = 0.07
epochs = 50
batch_size = 50
label_name = "median_house_value"

# Create and compile the mode'ls topography
my_model = create_model(learning_rate, feature_cross_feature_layer)

# Train the model on the training set.
epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

plot_the_loss_curve(epochs, rmse)

print("\n: Evaluate the new model against the test set:")
test_features = {name:np.array(value) for name, value in test_df.items()}
test_label = np.array(test_features.pop(label_name))
my_model.evaluate(x = test_features, y = test_label, batch_size = batch_size)

