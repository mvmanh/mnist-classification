#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MNIST Classification dùng CNN

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Dropout, MaxPool2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from IPython.display import SVG
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
import os

np.random.seed(100)


def show_history(acc, loss):
    plt.plot(acc)
    plt.plot(loss)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'test_accuracy'], loc='best')
    plt.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train đang có dạng (60000, 28, 28): 60k là số lượng sample, 28x28 là size ảnh

# chuyển input từ (6000,28,28) thành (60000,28,28,1) trong đó 1 là số channel
# tensorflows conv2d yêu cầu input 4d
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

m = x_train[0]

# chuyển input thành float để normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize
x_train /= 255
x_test /= 255

# label đang là (60000,1), chuyển sang dạng one-hot vector (60000,10)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

if os.path.exists('mnist_cnn.h5'):
    print('Model existed! Load model from file')
    model = load_model('mnist_cnn.h5')
else:
    print('Train new model')

    model = Sequential()

    # input shape có dạng width x height x channel thì không cần chiều của sample
    model.add(Conv2D(filters=32,kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))

    plot_model(model, to_file='model.png',show_shapes=True)

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=2, validation_data=(x_train, y_train))
    model.save('mnist_cnn.h5')

    acc_data = np.array(history.history['acc'])
    loss_data = np.array(history.history['acc'])

    np.save(open('data1.txt', 'w'), acc_data)
    np.save(open('data2.txt', 'w'), loss_data)

x = np.load(open('data1.txt'))
y = np.load(open('data2.txt'))
show_history(x,y)


print('Evaluating model')

score = model.evaluate(x_test, y_test, verbose=1)
prediction = model.predict(x_test)


prediction = np.argmax(prediction, axis=1)
y_test = np.argmax(y_test,axis=1)

matrix = confusion_matrix(y_test, prediction)
f1score = f1_score(y_test, prediction, average='weighted')
precision = precision_score(y_test, prediction,average='weighted')
#auc_value = roc_auc_score(y_test, precision)


print('Test score {0}'.format(score))
print('F1 score {0}'.format(f1score))
print('Precision score {0}'.format(precision))
#print('AUC score {0}'.format(auc_value))
print('Confusion matrix:\n{0}'.format(matrix))


