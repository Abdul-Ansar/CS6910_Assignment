
print("Importing packages... ", end="")

# Importing necessary packages
import numpy as np  # For numerical operations
from activations import Sigmoid, Tanh, Relu, Softmax  # Custom activation functions
from layers import Input, Dense  # Custom layer implementations
from optimizers import Normal, Momentum, Nesterov, AdaGrad, RMSProp, Adam, Nadam  # Custom optimizer implementations
from network import NeuralNetwork  # Custom neural network implementation
from loss import CrossEntropy  # Custom loss function
from helper import OneHotEncoder, MinMaxScaler  # Helper functions for data preprocessing

import matplotlib.pyplot as plt  # For plotting
from keras.datasets import fashion_mnist  # Importing Fashion MNIST dataset
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # Evaluation metrics
import seaborn as sns  # For visualization

import warnings  # For handling warnings
warnings.filterwarnings("ignore")  # Suppressing warnings
print("Done!")  # Indicating successful package import

# Dictionary mapping optimizer names to their corresponding classes
map_optimizers = {
    "Normal": Normal(),
    "Momentum": Momentum(),
    "Nesterov": Nesterov(),
    "AdaGrad": AdaGrad(),
    "RMSProp": RMSProp(),
    "Adam": Adam(),
    "Nadam": Nadam()
}

# Loading Fashion MNIST dataset
print("Loading data... ", end="")
[(x_train, y_train), (x_test, y_test)] = fashion_mnist.load_data()  # Loading data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2)  # Splitting data
print("Done!")  # Indicating successful data loading

# Printing sizes of training and validation datasets
print("Size of Training data:", x_train.shape)
print("Size of Validation data:", x_val.shape)

# Performing scaling and encoding transformations on the data
print("Performing Scaling and Encoding transformations on the data... ", end="")
X_scaled = scaler.transform(x_train)  # Scaling training data
X_val_scaled = scaler.transform(x_val)  # Scaling validation data
X_test_scaled = scaler.transform(x_test)  # Scaling test data

# Reshaping data for compatibility with neural network
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1] * X_scaled.shape[2]).T
X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1] * X_val_scaled.shape[2]).T
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1] * X_test_scaled.shape[2]).T

# Encoding labels using OneHotEncoder
encoder = OneHotEncoder()
t = encoder.fit_transform(y_train, 10)  # Encoding training labels
t_val = encoder.fit_transform(y_val, 10)  # Encoding validation labels
t_test = encoder.fit_transform(y_test, 10)  # Encoding test labels
print("Done!")  # Indicating successful transformations

# Limiting data size for faster testing
X_scaled = X_scaled[:, :2000]
X_val_scaled = X_val_scaled[:, :500]
t = t[:, :2000]
t_val = t_val[:, :500]

# Creating neural network layers
layers = [
    Input(data=X_scaled),  # Input layer with scaled data
    Dense(size=64, activation="Sigmoid", name="HL1"),  # Hidden layer with Sigmoid activation
    Dense(size=10, activation="Sigmoid", name="OL")  # Output layer with Sigmoid activation
]

# Initializing neural network model
model = NeuralNetwork(layers=layers, batch_size=2000, optimizer="Normal", \
                      initialization="RandomNormal", loss="CrossEntropy", \
                      epochs=int(100), t=t, X_val=X_val_scaled, t_val=t_val, \
                      use_wandb=False)

# Performing forward and backward propagation
model.forward_propogation()  # Forward propagation
first_pass_y = model.layers[-1].y  # Getting output of untrained network
model.backward_propogation()  # Backward propagation

# Evaluating performance on validation and test datasets
acc_val, loss_val, _ = model.check_test(X_val_scaled, t_val)  # Validation set evaluation
acc_test, loss_test, _ = model.check_test(X_test_scaled, t_test)  # Test set evaluation

# Printing evaluation results
print("=" * 50)
print("Training Data")
print("Fraction Correctly classified in untrained network:", np.sum(np.argmax(first_pass_y, axis=0) == y_train[:2000]) / 2000)
print("Fraction Correctly classified in trained network:", np.sum(np.argmax((model.layers[-1].y), axis=0) == y_train[:2000]) / 2000)

print("=" * 50)
print("Validation Data")
print("Fraction Correctly classified in trained network:", acc_val / t_val.shape[1])

print("=" * 50)
print("Testing Data")
print("Fraction Correctly classified in trained network:", acc_test / t_test.shape[1])

# Plotting accuracy and loss curves
plt.figure()
plt.plot(np.array(model.accuracy_hist_val) / 500, label="training accuracy")
plt.plot(np.array(model.accuracy_hist) / 2000, label="validation accuracy")
plt.title("Accuracy of the model")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(np.array(model.loss_hist) / 2000, label="training loss")
plt.plot(np.array(model.loss_hist_val) / 500, label="validation loss")
plt.title("Loss of the model")
plt.legend()
plt.grid()
plt.show()