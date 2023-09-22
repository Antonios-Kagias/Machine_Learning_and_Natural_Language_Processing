# -*- coding: utf-8 -*-
"""
# **Assignment 4**
Μη επιβλεπόμενη μάθηση – Συσταδοποίηση

### Import libraries
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation, Flatten
from keras import backend as K
from sklearn import cluster

"""### Define a performance evaluation function"""

def performance_score(input_values, cluster_indexes):
    try:
        silh_score = metrics.silhouette_score(input_values, cluster_indexes)
        print(' .. Silhouette Coefficient score is {:.2f}'.format(silh_score))
        #print( ' ... -1: incorrect, 0: overlapping, +1: highly dense clusts.')
    except:
        print(' .. Warning: could not calculate Silhouette Coefficient score.')
        silh_score = -999

    try:
        ch_score =\
         metrics.calinski_harabasz_score(input_values, cluster_indexes)
        print(' .. Calinski-Harabasz Index score is {:.2f}'.format(ch_score))
        #print(' ... Higher the value better the clusters.')
    except:
        print(' .. Warning: could not calculate Calinski-Harabasz Index score.')
        ch_score = -999

    try:
        db_score = metrics.davies_bouldin_score(input_values, cluster_indexes)
        print(' .. Davies-Bouldin Index score is {:.2f}'.format(db_score))
        #print(' ... 0: Lowest possible value, good partitioning.')
    except:
        print(' .. Warning: could not calculate Davies-Bouldin Index score.')
        db_score = -999

    return silh_score, ch_score, db_score

"""### Import the dataset"""

print('>>> Downloading dataset...')
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print('>>> Dataset downloaded...')

"""### Split into train, validation and test sets"""

print('>>> Splitting dataset...')
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=1) # 0.2 x 0.86 = 0.172
print('>>> Splitting done...')
print(f'>>> We have {len(y_train)} train images, {len(y_validate)} validation images and {len(y_test)} test images...')

"""### Create a CNN autoencoder"""

print('>>> Creating the CNN autoencoder...')
model = Sequential()
model.add(Conv2D(14, kernel_size=3, padding='same',\
                 activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu'))
model.add(UpSampling2D((2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))

print('>>> Model created...')
print('>>> Checking model architecture...')
model.summary()

"""### Compile and train the autoencoder"""

print('>>> Compiling the model...')
model.compile(optimizer='adam',
              loss=tf.keras.losses.mse,
              # categorical_crossentropy
              metrics=['accuracy'])
print('>>> Model compiled...')

print('>>> Training the model...')
print('>>> This may take a while...')
model.fit(X_train, X_train, epochs=3, validation_data=(X_validate, X_validate))
print('>>> Model training done...')

"""### Use the autoencoder over test data"""

print('>>> Using the trained autoencoder over test data...')
restored_testing_dataset = model.predict(X_test)
print('>>> Autoencoder used over test data...')

"""### Plot original and reconstructed images"""

print('>>> Plotting original (top) and reconstructed (bottom) images...')
plt.figure(figsize=(20,5))
for i in range(10):
    index = y_test.tolist().index(i)
    plt.subplot(2, 10, i+1)
    plt.imshow(X_test[index].reshape((28,28)))
    plt.gray()
    plt.subplot(2, 10, i+11)
    plt.imshow(restored_testing_dataset[index].reshape((28,28)))
    plt.gray()

"""### Exctract the encoder block"""

print('>>> Extracting encoder...')
encoder = K.function([model.layers[0].input],[model.layers[4].output])
print('>>> Encoder extracted...')

"""### Normalizing pixel values"""

# Normalize pixel values to be between 0 and 1
print('>>> Normalizing pixel values...')
train_norm_images, test_norm_images = X_train / 255.0, X_test / 255.0
print('>>> Pixel values normalized...')

"""### Convert images to projected data"""

print('>>> Converting images to projected data...')
test_encoded_images = encoder([X_test])[0].reshape(-1,7*7*7)
print('>>> Images converted...')

"""Normalized pixel values"""

print('>>> Converting images to projected data...')
test_norm_images = encoder([test_norm_images])[0].reshape(-1,7*7*7)
test_norm_images = test_norm_images / np.max(test_norm_images)
print('>>> Images converted...')

"""### Mini-Batch K-Means clustering (encoded pixel values)"""

print('>>> Mini-Batch K-Means clustering with encoded pixel values...')

mbkm = cluster.MiniBatchKMeans(n_init=3)
mbkm.fit(test_encoded_images)
mbkm_labels_enc = mbkm.labels_
mbkm_pred_labels_enc = mbkm.predict(test_encoded_images)

print('>>> Mini-Batch K-Means clustering with encoded pixel values finished...')

"""#### Visualizations"""

fig = plt.figure(figsize=(20,20))
for clusterIdx in range(len(mbkm_labels_enc)):
    for c, val in enumerate(X_test[mbkm_pred_labels_enc == clusterIdx][0:10]):
        fig.add_subplot(10, 10, 10*clusterIdx+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('clothes: '+str(clusterIdx))

"""#### Performance scores"""

print('>>> Calculating performance scores for Mini-Batch K-Means clustering with encoded pixel values...')
mbkm_silh_score_enc, mbkm_ch_score_enc, mbkm_db_score_enc = performance_score(y_test.reshape(-1, 1), mbkm_labels_enc)
mbkm_vm_score_enc = metrics.v_measure_score(mbkm_labels_enc, mbkm_pred_labels_enc)
print(f' .. V-measure score is {mbkm_vm_score_enc}')

"""### Mini-Batch K-Means clustering (normalized pixel values)"""

print('>>> Mini-Batch K-Means clustering with normalized pixel values...')

mbkm = cluster.MiniBatchKMeans(n_init=3)
mbkm.fit(test_norm_images)
mbkm_labels_norm = mbkm.labels_
mbkm_pred_labels_norm = mbkm.predict(test_norm_images)

print('>>> Mini-Batch K-Means clustering with normalized pixel values finished...')

"""#### Visualizations"""

fig = plt.figure(figsize=(20,20))
for clusterIdx in range(len(mbkm_labels_norm)):
    for c, val in enumerate(X_test[mbkm_pred_labels_norm == clusterIdx][0:10]):
        fig.add_subplot(10, 10, 10*clusterIdx+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('clothes: '+str(clusterIdx))

"""#### Performance scores"""

print('>>> Calculating performance scores for Mini-Batch K-Means clustering with normalized pixel values...')
mbkm_silh_score_norm, mbkm_ch_score_norm, mbkm_db_score_norm = performance_score(y_test.reshape(-1, 1), mbkm_labels_norm)
mbkm_vm_score_norm = metrics.v_measure_score(mbkm_labels_norm, mbkm_pred_labels_norm)
print(f' .. V-measure score is {mbkm_vm_score_norm}')

"""### Hierarchical clustering (encoded pixel values)"""

print('>>> Hierarchical clustering with encoded pixel values...')

aggl = cluster.AgglomerativeClustering()
aggl.fit(test_encoded_images)
aggl_labels_enc = aggl.labels_
aggl_pred_labels_enc = aggl.fit_predict(test_encoded_images)

print('>>> Hierarchical clustering with encoded pixel values finished...')

"""#### Visualizations"""

fig = plt.figure(figsize=(20,20))
for clusterIdx in range(len(aggl_labels_enc)):
    for c, val in enumerate(X_test[aggl_pred_labels_enc == clusterIdx][0:10]):
        fig.add_subplot(10, 10, 10*clusterIdx+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('clothes: '+str(clusterIdx))

"""#### Performance scores"""

print('>>> Calculating performance scores for Hierarchical clustering with encoded pixel values...')
aggl_silh_score_enc, aggl_ch_score_enc, aggl_db_score_enc = performance_score(y_test.reshape(-1, 1), aggl_labels_enc)
aggl_vm_score_enc = metrics.v_measure_score(aggl_labels_enc, aggl_pred_labels_enc)
print(f' .. V-measure score is {aggl_vm_score_enc}')

"""### Hierarchical clustering (normalized pixel values)"""

print('>>> Hierarchical clustering with normalized pixel values...')

aggl = cluster.AgglomerativeClustering()
aggl.fit(test_norm_images)
aggl_labels_norm = aggl.labels_
aggl_pred_labels_norm = aggl.fit_predict(test_norm_images)

print('>>> Hierarchical clustering with normalized pixel values finished...')

"""#### Visualizations"""

fig = plt.figure(figsize=(20,20))
for clusterIdx in range(len(aggl_labels_norm)):
    for c, val in enumerate(X_test[aggl_pred_labels_norm == clusterIdx][0:10]):
        fig.add_subplot(10, 10, 10*clusterIdx+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('clothes: '+str(clusterIdx))

"""#### Perfomance scores"""

print('>>> Calculating performance scores for Hierarchical clustering with normalized pixel values...')
aggl_silh_score_norm, aggl_ch_score_norm, aggl_db_score_norm = performance_score(y_test.reshape(-1, 1), aggl_labels_norm)
aggl_vm_score_norm = metrics.v_measure_score(aggl_labels_norm, aggl_pred_labels_norm)
print(f' .. V-measure score is {aggl_vm_score_norm}')

"""### Birch clustering (encoded pixel values)"""

print('>>> Birch clustering with encoded pixel values...')

birch = cluster.Birch(threshold=0.1)
birch.fit(test_encoded_images)
birch_labels_enc = birch.labels_
birch_pred_labels_enc = birch.fit_predict(test_encoded_images)

print('>>> Birch clustering with encoded pixel values finished...')

"""#### Visualizations"""

fig = plt.figure(figsize=(20,20))
for clusterIdx in range(len(birch_labels_enc)):
    for c, val in enumerate(X_test[birch_pred_labels_enc == clusterIdx][0:10]):
        fig.add_subplot(10, 10, 10*clusterIdx+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('clothes: '+str(clusterIdx))

"""#### Performance scores"""

print('>>> Calculating performance scores for Birch clustering with encoded pixel values...')
birch_silh_score_norm, birch_ch_score_norm, birch_db_score_norm = performance_score(y_test.reshape(-1, 1), birch_labels_enc)
birch_vm_score_norm = metrics.v_measure_score(birch_labels_enc, birch_pred_labels_enc)
print(f' .. V-measure score is {birch_vm_score_norm}')

"""### Birch clustering (normalized pixel values)"""

print('>>> Birch clustering...')

birch = cluster.Birch(threshold=0.1)
birch.fit(test_norm_images)
birch_labels_norm = birch.labels_
birch_pred_labels_norm = birch.fit_predict(test_norm_images)

print('>>> Birch clustering finished...')

"""#### Visualizations"""

fig = plt.figure(figsize=(20,20))
for clusterIdx in range(len(birch_labels_norm)):
    for c, val in enumerate(X_test[birch_pred_labels_norm == clusterIdx][0:10]):
        fig.add_subplot(10, 10, 10*clusterIdx+c+1)
        plt.imshow(val.reshape((28,28)))
        plt.gray()
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('clothes: '+str(clusterIdx))

"""#### Performance scores"""

print('>>> Calculating performance scores for Birch clustering with normalized pixel values...')
birch_silh_score_norm, birch_ch_score_norm, birch_db_score_norm = performance_score(y_test.reshape(-1, 1), birch_labels_norm)
birch_vm_score_norm = metrics.v_measure_score(birch_labels_norm, birch_pred_labels_norm)
print(f' .. V-measure score is {birch_vm_score_norm}')
