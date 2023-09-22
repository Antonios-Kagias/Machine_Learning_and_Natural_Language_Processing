# -*- coding: utf-8 -*-
"""### Import libraries"""

from tensorflow import keras #remember that keras is now included in tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import numpy as np
from keras import datasets, layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

"""### Download and prepare the CIFAR10 dataset"""

print('>>> Downloading dataset...')
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print('>>> Dataset downloaded...')

print(f'>>> We have {len(train_labels)} train images and {len(test_labels)} test images...')

# Normalize pixel values to be between 0 and 1
print('>>> Normalizing pixel values...')
train_images, test_images = train_images / 255.0, test_images / 255.0
print('>>> Pixel values normalized...')

# Create validation data
print('>>> Creating validation set...')
X_train, val_images, y_train, val_labels = train_test_split (train_images, train_labels, test_size=0.25, random_state=1) # 0.25 x 0.83 = 0.21
print('>>> Validation set created...')
print(f'>>> We have {len(y_train)} train images, {len(val_labels)} validation images and {len(test_labels)} test images...')

"""## Version 1.0



---

### Plot 4 images of each class

Before doing anything else, let's plot 4 images of each class
"""

print('>>> Plotting 4 images of each class based on actual class...')

# Store the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class_to_demonstrate = 0
while (sum(train_labels == class_to_demonstrate) > 4):
    tmp_idxs_to_use = np.where(train_labels == class_to_demonstrate)

    # create new plot window
    # small figsize to see images more clearly
    plt.figure(figsize=(3,3))

    plt.subplot(221)
    plt.imshow(train_images[tmp_idxs_to_use[0][0]])
    plt.subplot(222)
    plt.imshow(train_images[tmp_idxs_to_use[0][1]])
    plt.subplot(223)
    plt.imshow(train_images[tmp_idxs_to_use[0][2]])
    plt.subplot(224)
    plt.imshow(train_images[tmp_idxs_to_use[0][3]])
    tmp_title = 'Images belonging to class <' + str(class_names[class_to_demonstrate] + '>')
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1

"""### Create the convolutional base

The 6 lines of code below define the convolutional base using a common pattern: a stack of Conv2D and MaxPooling2D layers.

As input, a CNN takes tensors of shape (`image_height`, `image_width`, `color_channels`), ignoring the batch size. The `color_channels` argument refers to (R,G,B) so the CNN will be configured to process inputs of shape (32, 32, 3), which is the format of CIFAR images. This is done by passing the argument `input_shape` to the first layer.


"""

print('>>> Creating the model...')
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

"""### Add Dense layers on top
To complete the model, we will feed the last output tensor from the convolutional base (of shape (4, 4, 64)) into one or more Dense layers to perform classification. Dense layers take vectors as input (which are 1D), while the current output is a 3D tensor. First, we will flatten (or unroll) the 3D output to 1D, then add one or more Dense layers on top. CIFAR has 10 output classes, so we will use a final Dense layer with 10 outputs.
"""

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

"""Here's the complete architecture of the model:"""

print('>>> Model created...')
print('>>> Checking model architecture...')
model.summary()

"""### Compile and train the model

*duration of this cell execution: 10-15 minutes*
"""

print('>>> Compiling the model...')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print('>>> Model compiled...')

print('>>> Training the model...')
print('>>> This may take a while...')
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(val_images, val_labels))
print('>>> Model training done...')

"""### Evaluate the model"""

print('>>> Evaluating the model...')
print('>>> Calculating model loss and accuracy...')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model 1.0')

test_loss, test_acc = model.evaluate(val_images, val_labels, verbose=2)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

"""### Save the trained model"""

print('>>> Saving the trained model...')
model_name = 'CIFAR10_CNN.h5'
model.save(model_name)
print('>>> Model saved...')

