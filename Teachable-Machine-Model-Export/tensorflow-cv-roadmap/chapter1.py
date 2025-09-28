import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load datatenso
(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

# Normalise - 0 to 1
xtrain, xtest = xtrain/255.0, xtest/255.0

# Model Building
mizu = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),     # flatten 2d image to 1d array
    tf.keras.layers.Dense(128, activation='relu'),    # hl with 128 neurons
    tf.keras.layers.Dense(10, activation='softmax')   # output layer for 10 classes
])

# Model configuration with optimizer, loss functions and metrics
mizu.compile(optimizer='adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

# Train Model
mizu.fit(xtrain, ytrain, epochs=5)
print("Mizu is getting trained...")

# Evaluate Model (loss and metrics test data)
test_loss, test_acc = mizu.evaluate(xtest, ytest, verbose=2)
print(f'Test accuracy: {test_acc}')

# Making predictions using model (take first image from test set)
image = xtest[0]
label = ytest[0]

# Adding a batch dimension to the image
image = np.expand_dims(image, axis=0)

# Make a prediction
predictions = mizu.predict(image)
pred_class = np.argmax(predictions[0])

print()
print(f'True Label: {label}')
print(f'Predicted Label: {pred_class}')