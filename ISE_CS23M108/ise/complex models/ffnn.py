import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import pyJoules  
import random
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

# Define the architecture of the neural network
def create_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model

# Define a function to train the neural network
def train_model(model, inputs, targets, epochs):
    model.compile(loss=losses.MeanSquaredError(),
                  optimizer=optimizers.Adam())
    model.fit(inputs, targets, epochs=epochs, verbose=1)

# Define a function to make predictions using the trained model
def predict(model, inputs):
    predictions = model.predict(inputs)
    return predictions

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def main():
    # Generate example data
    import numpy as np
    inputs = np.random.randn(100, 10)  # 100 samples with 10 features
    targets = np.random.randn(100, 1)  # Regression targets

    # Create the model
    model = create_model()

    # Train the model
    train_model(model, inputs, targets, epochs=10)

    # Make predictions
    predictions = predict(model, inputs)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()

csv_handler.save_data()

