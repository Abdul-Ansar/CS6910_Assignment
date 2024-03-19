# -*- coding: utf-8 -*-
"""Question_10 .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xlmUmTcWFfA2uNHkFb7yg9P8L_Mr9frU
"""

import wandb  # Importing WandB for experiment tracking
import numpy as np  # For numerical operations
import os  # For operating system related functionalities
from activations import Sigmoid, Tanh, Relu, Softmax  # Custom activation functions
from layers import Input, Dense  # Custom layer implementations
from optimizers import Momentum, Nesterov, AdaGrad, RMSProp, Adam, Nadam  # Custom optimizer implementations
from network import NeuralNetwork  # Custom neural network implementation
from loss import CrossEntropy  # Custom loss functions
from helper import OneHotEncoder, MinMaxScaler  # Helper functions for data preprocessing

from sklearn.model_selection import train_test_split  # For splitting dataset
from keras.datasets import mnist  # Importing MNIST dataset
import matplotlib.pyplot as plt  # For plotting

###############################################################
# Loading and preprocessing data
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # Loading data
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # Printing dataset shapes

X_scaled = X_train / 255  # Scaling training data
X_test_scaled = X_test / 255  # Scaling test data

X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1] * X_scaled.shape[2]).T  # Reshaping training data
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1] * X_test_scaled.shape[2]).T  # Reshaping test data

encoder = OneHotEncoder()  # Initializing OneHotEncoder
t_train = encoder.fit_transform(y_train, 10)  # Encoding training labels
t_test = encoder.fit_transform(y_test, 10)  # Encoding test labels

###############################################################
# Model Configuration 1
layers1 = [
    Input(data=X_scaled),
    Dense(size=64, activation='Tanh', name='HL1'),
    Dense(size=10, activation='Sigmoid', name='OL')
]

# Creating and training neural network model
nn_model1 = NeuralNetwork(layers=layers1, batch_size=1024, optimizer='Adam', initialization='XavierUniform',
                          epochs=10, t=t_train, X_val=X_test_scaled, t_val=t_test, loss="CrossEntropy")
nn_model1.forward_propagation()
nn_model1.backward_propagation()

# Evaluating accuracy on test set
acc_test1, _, _ = nn_model1.check_test(X_test_scaled, y_test)
print('Accuracy on test set for Configuration 1 =', acc_test1)

###############################################################
# Model Configuration 2
layers2 = [
    Input(data=X_scaled),
    Dense(size=32, activation='Tanh', name='HL1'),
    Dense(size=10, activation='Sigmoid', name='OL')
]

# Creating and training neural network model
nn_model2 = NeuralNetwork(layers=layers2, batch_size=128, optimizer='Adam', initialization='XavierUniform',
                          epochs=10, t=t_train, X_val=X_test_scaled, t_val=t_test, loss="CrossEntropy")
nn_model2.forward_propagation()
nn_model2.backward_propagation()

# Evaluating accuracy on test set
acc_test2, _, _ = nn_model2.check_test(X_test_scaled, y_test)
print('Accuracy on test set for Configuration 2 =', acc_test2)

###############################################################
# Model Configuration 3
layers3 = [
    Input(data=X_scaled),
    Dense(size=32, activation='Relu', name='HL1'),
    Dense(size=10, activation='Sigmoid', name='OL')
]

# Creating and training neural network model
nn_model3 = NeuralNetwork(layers=layers3, batch_size=1024, optimizer='Adam', initialization='XavierUniform',
                          epochs=10, t=t_train, X_val=X_test_scaled, t_val=t_test, loss="CrossEntropy")
nn_model3.forward_propagation()
nn_model3.backward_propagation()

# Evaluating accuracy on test set
acc_test3, _, _ = nn_model3.check_test(X_test_scaled, y_test)
print('Accuracy on test set for Configuration 3 =', acc_test3)