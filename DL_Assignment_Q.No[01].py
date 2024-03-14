
!pip install wandb
# Importing necessary packages
print("Importing packages... ", end="")
import wandb  # We import the Weights & Biases library for experiment tracking
import numpy as np  # NumPy is imported for numerical operations
from keras.datasets import fashion_mnist  # Importing Fashion MNIST dataset from Keras
import matplotlib.pyplot as plt  # Matplotlib is used for plotting

# Initializing W&B project for experiment tracking
wandb.init(project="trail-1")
print("Done!")  # Indicates successful package import and W&B initialization

# Loading the Fashion MNIST dataset
print("Loading data... ", end="")
# Load the dataset
[(x_train, y_train), (x_test, y_test)] = fashion_mnist.load_data()

# Defining number of classes and their name mappings
num_classes = 10
class_mapping = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
print("Done!")  # Indicates successful data loading

# Plotting a figure from each class
plt.figure(figsize=[12, 5])  # Setting up figure size
img_list = []  # List to store images
class_list = []  # List to store class names

# Looping through each class
for i in range(num_classes):
    # Finding the first occurrence of an image of the current class
    position = np.argmax(y_train == i)
    image = x_train[position, :, :]  # Extracting the image
    plt.subplot(2, 5, i + 1)  # Setting up subplots
    plt.imshow(image)  # Displaying the image
    plt.title(class_mapping[i])  # Adding title with class name
    img_list.append(image)  # Appending image to the list
    class_list.append(class_mapping[i])  # Appending class name to the list

# Logging images and their corresponding captions to W&B
wandb.log({"Question 1": [wandb.Image(img, caption=caption) for img, caption in zip(img_list, class_list)]})