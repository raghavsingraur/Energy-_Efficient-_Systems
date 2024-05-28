import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import pyJoules  
import random
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

# Function to create the CNN model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Function to train the CNN model
def train_model(model, train_data, train_labels, epochs, batch_size):
    model.compile(loss=losses.sparse_categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=1)

# Function to make predictions using the trained model
def predict(model, test_data):
    predictions = model.predict(test_data)
    return predictions

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def main():
    # Generate example data
    num_classes = 10  # Number of classes
    input_shape = (28, 28, 1)  # Input shape for MNIST dataset
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32') / 255
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32') / 255

    # Create the model
    model = create_model(input_shape, num_classes)

    # Train the model
    train_model(model, train_data, train_labels, epochs=5, batch_size=32)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
    print(f'Test accuracy: {test_acc}')

    # Make predictions
    predictions = predict(model, test_data)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()

csv_handler.save_data()
