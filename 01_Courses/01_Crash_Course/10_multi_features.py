import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import seaborn as sns

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.3f}".format


""" Load the dataset.------------------------------------------------------------------------------------- """
# Load the dataset.
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
# Shuffles the training examples.
train_df = train_df.reindex(np.random.permutation(train_df.index))


""" Normalize values.----------------------------------------------------------------------------------- """
# Calculate the Z-scores of each column in the training set and write those Z-scores into a new pandas DataFrame named train_df_norm.
train_df_mean = train_df.mean() # Get the mean of train_df
train_df_std = train_df.std() # Get the standart deviation of train_df.
train_df_norm = (train_df - train_df_mean) / train_df_std # Calculate the Z-scores.

# Calculate the Z-scores of each column in the test set and write those Z-scores into a new pandas DataFrame named test_df_norm.
test_df_mean = test_df.mean() # Get the mean of test_df.
test_df_std = test_df.std() # Get the standart deviation of test_df.
test_df_norm = (test_df - test_df_mean) / test_df_std # Calculate the Z-scores.


""" Create a feature layer.---------------------------------------------------------------------------- """
# Create an empty list that will eventually hold all feature columns.
feature_column = []

# As we scale dthe columns into their Z-scores, instead of picking a resolution in degrees, we're going to use resolution_in_Zs.
# A resolution_in_zs of 1 correspond to a full standard deviation.
resolution_in_Zs = 0.3
#Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")
latitude_boundaries = list(np.arange(int(min(train_df_norm["latitude"])),
                                     int(max(train_df_norm["latitude"])),
                                     resolution_in_Zs))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)
#Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df_norm["longitude"])),
                                     int(max(train_df_norm["longitude"])),
                                     resolution_in_Zs))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column, longitude_boundaries)
#Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude,longitude], hash_bucket_size = 100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_column.append(crossed_feature)

# Represent median_income as a floating-point value.
median_income = tf.feature_column.numeric_column("median_income")
feature_column.append(median_income)

# Represent population as a floating-point value.
population = tf.feature_column.numeric_column("population")
feature_column.append(population)

# Represent total_rooms as a floating-point value.
total_rooms = tf.feature_column.numeric_column("total_rooms")
feature_column.append(total_rooms)

# Represent households as a floating-point value.
households = tf.feature_column.numeric_column("households")
feature_column.append(households)

# Convert the list of feature columns into a layer that will later be fed into the model.
my_feature_layer = tf.keras.layers.DenseFeatures(feature_column)


# Define the plotting function.
def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Mean Squared Error")

  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min()*0.95, mse.max() * 1.03])
  plt.show()  


""" Build a deep neural net model. ------------------------------------------------------------------------ """
def create_model(my_learning_rate, my_feature_layer):
    model = tf.keras.models.Sequential()
    
    # Add the layer containing the feature column to the model. 
    model.add(my_feature_layer)

    # Describe the topography of the model by calling tf.keras.layers.Dense method once for each layer.
    #   * units specifies the number of nodes in this layer.
    #   * activation specifies the activation function (Rectified Linear Unit).
    #   * name is just a string that can be useful when debugging.

    # Define the first hidden layer with 36 nodes.
    model.add(tf.keras.layers.Dense(units = 36, activation = "relu", name = "Hidden1"))

    model.add(tf.keras.layers.Dropout(rate = 0.2))

    # Define the second hidden layer with 18 nodes.
    model.add(tf.keras.layers.Dense(units = 18, activation = "relu", name = "Hidden2"))
    
    # Define the outputlayer.
    model.add(tf.keras.layers.Dense(units = 1, name = "Output"))

    model.compile(optimizer = tf.keras.optimizers.Adam(lr = my_learning_rate),
                  loss = "mean_squared_error",
                  metrics = [tf.keras.metrics.MeanSquaredError()])

    return model

def train_model(model, dataset, epochs, label_name, batch_size = None, ):
    """ Train the model by feeding it data. """

    # Split the dataset into features and label.
    features = {name:np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x = features, y = label, epochs = epochs, batch_size = batch_size, shuffle = True)

    epochs = history.epoch

    # To trach the progreesion of training, gather a snapshot of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]
    return epochs, mse


""" Call the functions to build and train.--------------------------------------------------------------------- """
# The following variables are the hyperparameters.
learning_rate = 0.011
epochs = 120
batch_size = 600

# Specify the label.
label_name = "median_house_value"

# Establish the model's topography.
my_model = create_model(learning_rate, my_feature_layer)

# Train the model on the normalized training set.
epochs, mse = train_model(my_model, train_df_norm, epochs, label_name, batch_size)

plot_the_loss_curve(epochs, mse)

test_features = {name:np.array(value) for name, value in test_df_norm.items()}
test_label = np.array(test_features.pop(label_name)) # isolate the label
print("\n Evaluate the linear regression model against the test set:")
my_model.evaluate(x = test_features, y = test_label, batch_size=batch_size)

