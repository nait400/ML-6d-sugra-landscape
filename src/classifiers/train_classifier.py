import scipy.io
import tensorflow as tf
import random
import csv
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# This file trains a classifier on gram matrix data.


# Here are variables one might want to change

# Use this many solutions for validation
numVal = 4000
# Train for this many epochs
max_epochs = 500
# Set batch size
batches = 100
# Which activation function to use
af = 'relu'
# Folder to save network snapshots + images
folder = 'classifier' + str(round(time.time()))


unit_data = []
labels = []
labels_shuffled = []
indices = []
data_shuffled = []

# Load the Gram matrix data into memory
mat0 = scipy.io.loadmat('grams_0.mat')
mat1 = scipy.io.loadmat('grams_1.mat')

# Choose a number of unit labeled solutions which matches the zero labeled ones
unit_chosen = random.sample(range(len(mat1['matrix_name'])),len(mat0['matrix_name']))

for i in range(len(unit_chosen)):
    unit_data.append(mat1['matrix_name'][unit_chosen[i]])
    labels.append(0)
    indices.append(i)
for i in range(len(unit_chosen)):
    labels.append(1)
    indices.append(i+len(unit_chosen))

data = [*mat0['matrix_name'],*unit_data]

print(len(data))

# Shuffles the data
indices = random.sample(indices,k=len(indices))

for i in range(len(indices)):
     data_shuffled.append(data[indices[i]])
     labels_shuffled.append(labels[indices[i]])

# Set aside validation data
data_val = data_shuffled[:numVal]
labels_val = labels_shuffled[:numVal]

data_train = data_shuffled[numVal:]
labels_train = labels_shuffled[numVal:]

gm_features = len(data_train[0])

# Convert the data to tensors, free up the previous lists
data_train = tf.convert_to_tensor(data_train)
data_val = tf.convert_to_tensor(data_val)
labels_train = tf.convert_to_tensor(labels_train)
labels_val = tf.convert_to_tensor(labels_val)

# Define the network architecture
input = tf.keras.layers.Input(shape=(gm_features))
dense = tf.keras.layers.Dense(64, activation=af)(input)
# dense = tf.keras.layers.Dense(32, activation=af)(dense)
dense = tf.keras.layers.Dense(16, activation=af)(dense)
# dense = tf.keras.layers.Dense(8, activation=af)(dense)
dense = tf.keras.layers.Dense(4, activation=af)(dense)
# dense = tf.keras.layers.Dense(2, activation=af)(dense)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

model = tf.keras.Model(
    inputs = input,
    outputs = output
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics = tf.keras.metrics.BinaryAccuracy()
)

# Uncomment this if you want it to print the architecture
# model.summary()

history = model.fit(
    x = data_train,
    y = labels_train,
    validation_data=(data_val,labels_val),
    epochs=max_epochs,
    batch_size=batches,
    verbose=1
)

# Save a copy of the network
model.save(folder)

# # Output a plot of the losses
# plt.plot(history.history['binary_accuracy'], 'bo', label='Training Accuracy')
# plt.plot(history.history['val_binary_accuracy'], 'rs', label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.9, 1])
# plt.legend(loc='lower right')
# plt.savefig(folder + '/loss_plot.png')