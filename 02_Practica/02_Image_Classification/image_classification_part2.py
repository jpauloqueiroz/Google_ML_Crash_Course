""" 1 - Explore how data augmentation works by making random transformations to training images.
    2 - Add data augmentation to our data preprocessing.
    3 - Add dropout to the convnet.
    4 - Retrain the model and evaluate loss and accuracy.

    - rotation_range is a value in degrees (0–180), a range within which to randomly rotate pictures.
    - width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
    - shear_range is for randomly applying shearing transformations.
    - zoom_range is for randomly zooming inside pictures.
    - horizontal_flip is for randomly flipping half of the images horizontally. This is relevant when there are no assumptions of horizontal assymmetry (e.g. real-world pictures).
    - fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
""" 

#%% Imports.
import tensorflow as tf
import PIL.Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tensorflow.keras import layers
from tensorflow.keras import Model

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


#%% Load and unzip data.
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# Directory with our training cat pictures.
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with our training dog pictures.
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat pictures.
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with our validation dog pictures.
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)


#%%
print(train_cat_fnames)
print(train_dog_fnames)


# %% Add Data Augmentation to the Preprocessing Step.
# Adding rescale, rotation_range, width_shift_range, height_shift_range,
# shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 32 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# %% Apply the datagen transformations to a cat image from the training set to produce five random variants.
img_path = os.path.join(train_cats_dir, train_cat_fnames[7])
img = load_img(img_path, target_size = (150,150)) # this is a PIL image
x = tf.keras.preprocessing.image.img_to_array(img) # numpy array with shape(150, 150, 3)
x = x.reshape((1,) + x.shape) # numpy array with shape(1, 150, 150, 3)

# The .flow() command below generates batches of randomly transformed images
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in train_datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(tf.keras.preprocessing.image.array_to_img(batch[0]))
    i += 1
    if i % 5 == 0:
        break
""" If we train a new network using this data augmentation configuration, our network will never see the same input twice. 
    However the inputs that it sees are still heavily intercorrelated, so this might not be quite enough to completely get rid of overfitting."""


# %% Building the convnet and adding Dropout
# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Convolution2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dimension tensor
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation = "relu")(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation since is a binary classification
output = layers.Dense(1, activation = "sigmoid")(x)

# Configure and compile out model
model = Model(img_input, output)
model.compile(loss = "binary_crossentropy",
              optimizer = tf.keras.optimizers.RMSprop(lr = 0.001),
              metrics = ["acc"])


# %% Train the model.
history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=1)

""" Note that with data augmentation in place, the 2,000 training images are randomly transformed each time a new training epoch runs, 
    which means that the model will never see the same image twice during training."""


# %% Evaluate the Results
# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

plt.show()


#%% Clean
import os, signal
os.kill(os.getpid(), signal.SIGKILL)