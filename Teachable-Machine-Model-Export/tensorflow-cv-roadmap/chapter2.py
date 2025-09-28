import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense 
import matplotlib.pyplot as plt 
import numpy as np

# Loading fashion MNIST dataset 
(xtrain, ytrain), (xtest, ytest) = fashion_mnist.load_data()

# Defining class names for labels
cn = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalise the data
xtrain, xtest = xtrain/255.0, xtest/255.0

# Displaying some sample images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(xtrain[i], cmap=plt.cm.binary)
    plt.xlabel(cn[ytrain[i]])
plt.show()

# Building model
milo = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
milo.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

# Train the model
milo.fit(xtrain, ytrain, epochs=10)

# Evaluating model
test_loss, test_acc = milo.evaluate(xtest, ytest, verbose=2)

print("\n", "-"*50)
print(f'Test Accuracy: {test_acc}')

# Predictions
img = xtest[0]
img = np.expand_dims(img,0)   # Adding a batch dimension

predictions = milo.predict(img)
pred = np.argmax(predictions[0])
true = ytest[0]

print("-"*50)
print("Prediction: ", cn[pred])
print("Actual Label: ", cn[true])
print("-"*50)