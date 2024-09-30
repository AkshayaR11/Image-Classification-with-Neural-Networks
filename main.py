import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

# Load the pre-trained model
model = models.load_model('image_classifier.h5')

# Load and process the input image
img = cv.imread('plane.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Convert the color from BGR to RGB
img = cv.resize(img, (32, 32))  # Resize the image to 32x32

# Normalize the image before prediction
img = np.array([img]) / 255.0

# Display the image
plt.imshow(img[0])
plt.show()

# Make a prediction
prediction = model.predict(img)
index = np.argmax(prediction)

# Class names for CIFAR-10
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Print the predicted class
print(f'Prediction is: {class_names[index]}')
