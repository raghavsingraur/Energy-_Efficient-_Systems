import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import pyJoules  
import random
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler


# Function to create the RNN model
def create_model(input_shape, hidden_units, output_units):
    model = tf.keras.Sequential([
        layers.SimpleRNN(hidden_units, input_shape=input_shape, activation='relu'),
        layers.Dense(output_units)
    ])
    return model

# Function to train the RNN model
def train_model(model, inputs, targets, epochs, batch_size):
    model.compile(loss=losses.MeanSquaredError(),
                  optimizer=optimizers.Adam())
    model.fit(inputs, targets, epochs=epochs, batch_size=batch_size, verbose=1)

# Function to make predictions using the trained model
def predict(model, inputs):
    predictions = model.predict(inputs)
    return predictions

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def main():
    # Generate example data
    input_shape = (10, 3)  # Sequence length of 10 with 3 features
    hidden_units = 64  # Number of units in the RNN layer
    output_units = 1  # Regression task with one output
    inputs = np.random.randn(100, 10, 3)  # 100 samples of sequences with 10 timesteps and 3 features each
    targets = np.random.randn(100, 1)  # Regression targets
    epochs = 10
    batch_size = 32

    # Create the model
    model = create_model(input_shape, hidden_units, output_units)

    # Train the model
    train_model(model, inputs, targets, epochs, batch_size)

    # Make predictions
    predictions = predict(model, inputs)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()

csv_handler.save_data()
