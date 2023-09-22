# -*- coding: utf-8 -*-
"""### Load the trained model and use it over test data"""

print('>>> Loading the trained model...')
loaded_model = keras.models.load_model(model_name)
print('>>> Model loaded...')

print('>>> Using the trained model over test data...')
y_test_predictions_vectorized = loaded_model.predict(test_images)
y_test_predictions = np.argmax(y_test_predictions_vectorized, axis=1)
print('>>> Predictions made over test data...')

"""### Calculate metric scores
Accuracy, precision, recall, f1 score
"""

# calculate the scores
print('>>> Calculating metric scores...')
acc_test = accuracy_score(test_labels, y_test_predictions)
pre_test = precision_score(test_labels, y_test_predictions, average='macro')
rec_test = recall_score(test_labels, y_test_predictions, average='macro')
f1_test = f1_score(test_labels, y_test_predictions, average='macro')

# print the scores
print('Accuracy score of the classifier is: {:.2f}.'.format(acc_test))
print('Precision score of the classifier is: {:.2f}.'.format(pre_test))
print('Recall score of the classifier is: {:.2f}.'.format(rec_test))
print('F1 score of the classifier is: {:.2f}.'.format(f1_test))

"""### Confusion matrix

Show TP, TN, FP, FN
"""

# compute confusion matrix
print('>>> Computing the confusion matrix...')
conf_matrix = confusion_matrix(test_labels, y_test_predictions)

print('>>> Calculating TP, TN, FP, FN for each class...')
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)

# print results
for i in range(len(TP)):
    print(f"TP for class {i}: {TP[i]}\t TN for class {i}: {TN[i]}\t FP for class {i}: {FP[i]}\t FN for class {i}: {FN[i]}\t")

"""### Illustrate a few results

Based on the classifier prediction
"""

print('>>> Illustrating a few results...')
print('>>> Plotting 4 images of each class based on classifier prediction...')

# Store the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class_to_demonstrate = 0
while (sum(y_test_predictions == class_to_demonstrate) > 4):
    tmp_idxs_to_use = np.where(y_test_predictions == class_to_demonstrate)

    # create new plot window
    # small figsize to see images more clearly
    plt.figure(figsize=(3,3))

    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(test_images[tmp_idxs_to_use[0][0]])
    plt.subplot(222)
    plt.imshow(test_images[tmp_idxs_to_use[0][1]])
    plt.subplot(223)
    plt.imshow(test_images[tmp_idxs_to_use[0][2]])
    plt.subplot(224)
    plt.imshow(test_images[tmp_idxs_to_use[0][3]])
    tmp_title = 'Images considered as <' + str(class_names[class_to_demonstrate] + '>')
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1

"""## Version 2.0

Using a different loss function during model training

---

### Transform to one-hot-encoding
"""

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()
one_hot_train = onehotencoder.fit_transform(y_train[:,0:1]).toarray()
one_hot_val = onehotencoder.fit_transform(val_labels[:,0:1]).toarray()

# print(len(one_hot_train))
# print(len(one_hot_val))

"""### Duplicate the model"""

model2 = model

"""### Compile and train the model

*duration of this cell execution: 10-15 minutes*
"""

print('>>> Compiling model version 2.0...')
model2.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalHinge(),
              metrics=['accuracy'])
print('>>> Model 2.0 compiled...')

print('>>> Training the model 2.0...')
print('>>> This may take a while...')
history = model2.fit(X_train, one_hot_train, epochs=10,
                    validation_data=(val_images, one_hot_val))
print('>>> Model 2.0 training done...')

"""### Evaluate the model"""

print('>>> Evaluating the model...')
print('>>> Calculating model loss and accuracy...')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model 2.0')

test_loss2, test_acc2 = model2.evaluate(val_images, one_hot_val, verbose=2)
print('Test loss:', test_loss2)
print('Test accuracy:', test_acc2)

"""### Save the trained model"""

print('>>> Saving the trained model...')
model_name_2 = 'CIFAR10_CNN_2.h5'
model2.save(model_name_2)
print('>>> Model saved...')

"""### Load the trained model and use it over test data"""

print('>>> Loading the trained model...')
loaded_model = keras.models.load_model(model_name_2)
print('>>> Model loaded...')

print('>>> Using the trained model over test data...')
y_test_predictions_vectorized = loaded_model.predict(test_images)
y_test_predictions = np.argmax(y_test_predictions_vectorized, axis=1)
print('>>> Predictions made over test data...')

"""### Calculate metric scores
Accuracy, precision, recall, f1 score
"""

# calculate the scores
print('>>> Calculating metric scores...')
acc_test = accuracy_score(test_labels, y_test_predictions)
pre_test = precision_score(test_labels, y_test_predictions, average='macro')
rec_test = recall_score(test_labels, y_test_predictions, average='macro')
f1_test = f1_score(test_labels, y_test_predictions, average='macro')

# print the scores
print('Accuracy score of the classifier is: {:.2f}.'.format(acc_test))
print('Precision score of the classifier is: {:.2f}.'.format(pre_test))
print('Recall score of the classifier is: {:.2f}.'.format(rec_test))
print('F1 score of the classifier is: {:.2f}.'.format(f1_test))

"""### Confusion matrix

Show TP, TN, FP, FN
"""

# compute confusion matrix
print('>>> Computing the confusion matrix...')
conf_matrix = confusion_matrix(test_labels, y_test_predictions)

print('>>> Calculating TP, TN, FP, FN for each class...')
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)

# print results
for i in range(len(TP)):
    print(f"TP for class {i}: {TP[i]}\t TN for class {i}: {TN[i]}\t FP for class {i}: {FP[i]}\t FN for class {i}: {FN[i]}\t")

"""### Illustrate a few results

Based on the classifier prediction
"""

print('>>> Illustrating a few results...')
print('>>> Plotting 4 images of each class based on classifier prediction...')

# Store the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class_to_demonstrate = 0
while (sum(y_test_predictions == class_to_demonstrate) > 4):
    tmp_idxs_to_use = np.where(y_test_predictions == class_to_demonstrate)

    # create new plot window
    # small figsize to see images more clearly
    plt.figure(figsize=(3,3))

    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(test_images[tmp_idxs_to_use[0][0]])
    plt.subplot(222)
    plt.imshow(test_images[tmp_idxs_to_use[0][1]])
    plt.subplot(223)
    plt.imshow(test_images[tmp_idxs_to_use[0][2]])
    plt.subplot(224)
    plt.imshow(test_images[tmp_idxs_to_use[0][3]])
    tmp_title = 'Images considered as <' + str(class_names[class_to_demonstrate] + '>')
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1
# cmap=plt.get_cmap('gray')

"""## Version 3.0

Using a more complicated model architecture

---

### Create the model 3.0
"""

print('>>> Creating the model 3.0...')
model3 = models.Sequential()
model3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(64, (3, 3), activation='relu'))


model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model3.add(layers.MaxPooling2D((2, 2)))


model3.add(layers.Flatten())
model3.add(layers.Dense(64, activation='relu'))
model3.add(layers.Dense(10))

"""Here's the complete architecture of the model:"""

print('>>> Model created...')
print('>>> Checking model architecture...')
model3.summary()

"""### Compile and train the model

*duration of this cell execution: 10-15 minutes*
"""

print('>>> Compiling the model 3.0...')
model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print('>>> Model 3.0 compiled...')

print('>>> Training the model 3.0...')
print('>>> This may take a while...')
history = model3.fit(X_train, y_train, epochs=10,
                    validation_data=(val_images, val_labels))
print('>>> Model 3.0 training done...')

"""### Evaluate the model"""

print('>>> Evaluating the model 3.0...')
print('>>> Calculating model loss and accuracy...')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model 3.0')

test_loss, test_acc = model3.evaluate(val_images, val_labels, verbose=2)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

"""### Save the trained model"""

print('>>> Saving the trained model 3.0...')
model_name = 'CIFAR10_CNN_3.h5'
model3.save(model_name)
print('>>> Model 3.0 saved...')

"""### Load the trained model and use it over test data"""

print('>>> Loading the trained model 3.0...')
loaded_model = keras.models.load_model(model_name)
print('>>> Model 3.0 loaded...')

print('>>> Using the trained model 3.0 over test data...')
y_test_predictions_vectorized = loaded_model.predict(test_images)
y_test_predictions = np.argmax(y_test_predictions_vectorized, axis=1)
print('>>> Predictions made over test data...')

"""### Calculate metric scores
Accuracy, precision, recall, f1 score
"""

# calculate the scores
print('>>> Calculating metric scores...')
acc_test = accuracy_score(test_labels, y_test_predictions)
pre_test = precision_score(test_labels, y_test_predictions, average='macro')
rec_test = recall_score(test_labels, y_test_predictions, average='macro')
f1_test = f1_score(test_labels, y_test_predictions, average='macro')

# print the scores
print('Accuracy score of the classifier is: {:.2f}.'.format(acc_test))
print('Precision score of the classifier is: {:.2f}.'.format(pre_test))
print('Recall score of the classifier is: {:.2f}.'.format(rec_test))
print('F1 score of the classifier is: {:.2f}.'.format(f1_test))

"""### Confusion matrix

Show TP, TN, FP, FN
"""

# compute confusion matrix
print('>>> Computing the confusion matrix...')
conf_matrix = confusion_matrix(test_labels, y_test_predictions)

print('>>> Calculating TP, TN, FP, FN for each class...')
FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
TP = np.diag(conf_matrix)
TN = conf_matrix.sum() - (FP + FN + TP)

# print results
for i in range(len(TP)):
    print(f"TP for class {i}: {TP[i]}\t TN for class {i}: {TN[i]}\t FP for class {i}: {FP[i]}\t FN for class {i}: {FN[i]}\t")

"""### Illustrate a few results

Based on the classifier prediction
"""

print('>>> Illustrating a few results...')
print('>>> Plotting 4 images of each class based on classifier prediction...')

# Store the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class_to_demonstrate = 0
while (sum(y_test_predictions == class_to_demonstrate) > 4):
    tmp_idxs_to_use = np.where(y_test_predictions == class_to_demonstrate)

    # create new plot window
    # small figsize to see images more clearly
    plt.figure(figsize=(3,3))

    # plot 4 images as gray scale
    plt.subplot(221)
    plt.imshow(test_images[tmp_idxs_to_use[0][0]])
    plt.subplot(222)
    plt.imshow(test_images[tmp_idxs_to_use[0][1]])
    plt.subplot(223)
    plt.imshow(test_images[tmp_idxs_to_use[0][2]])
    plt.subplot(224)
    plt.imshow(test_images[tmp_idxs_to_use[0][3]])
    tmp_title = 'Images considered as <' + str(class_names[class_to_demonstrate] + '>')
    plt.suptitle(tmp_title)

    # show the plot
    plt.show()
    plt.pause(2)

    # update the class to demonstrate index
    class_to_demonstrate = class_to_demonstrate + 1