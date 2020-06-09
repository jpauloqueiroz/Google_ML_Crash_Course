import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.3f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth = 200)

# Load the dataset
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Output example #2917 of the training set.
#print(x_train[2917])

# Use false colors to visualize the array.
#plt.imshow(x_train[2917])
#plt.show()

# Output row #10 of example #2917.
#print(x_train[2917][10])

# Output pixel #16 of row #10 of example #2900.
#x_train[2917][10][16]


# Normalize features values.
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

# Define the plotting function
def plot_curve(epochs, hist, list_of_metrics):
    # list_of_metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label = m)

    plt.legend()
    plt.show()


""" Create and train the model. ------------------------------------------------------------------- """
# Create a deep neural net model
def create_model(my_learning_rate):
    model = tf.keras.models.Sequential()

    # The features are stored in a two-dimensional 28X28 array. Flatten that two-dimensional array into a a one-dimensional
    # 784-element array.
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Define the first hiddel layer.
    model.add(tf.keras.layers.Dense(units = 256, activation = "relu"))

    # Define a dropout regularization layer.
    model.add(tf.keras.layers.Dropout(rate = 0.2))

    # Define the second hiddel layer.
    model.add(tf.keras.layers.Dense(units = 128, activation = "relu"))

    # Define the output layer. The units parameter is set to 10 because the model must choose among 10 possible output values (representing
    # the digits from 0 to 9, inclusive).
    model.add(tf.keras.layers.Dense(units = 10, activation = "softmax"))

    # Notice that the loss function for multi-class classification is different than the loss function for binary classification.
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = my_learning_rate),
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    
    return model

# Train the model
def train_model(model, train_features, train_label, epochs, batch_size = None, validation_split = 0.1):
    history = model.fit(x = train_features, y = train_label, epochs = epochs, batch_size = batch_size, shuffle = True,
                        validation_split = validation_split)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

""" Invoke the previous functions. --------------------------------------------------------------- """
leaning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

# Establish the model's topography.
my_model = create_model(leaning_rate)

# Train the model on the normalized training set.
epochs, hist = train_model(my_model, x_train_normalized, y_train, epochs, batch_size, validation_split)

# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ["accuracy"]
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate the model against the test set.
my_model.evaluate(x = x_test_normalized, y = y_test, batch_size = batch_size)





