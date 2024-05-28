import pyJoules  
import random
from pyJoules.energy_meter import measure_energy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
from pyJoules.handler.csv_handler import CSVHandler

def load_data():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Initialize and train the model
    model = LogisticRegression(max_iter=1000)  # Example model (you can use any model)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def random_sampling(X, y, test_size):
    # Randomly shuffle the indices
    indices = np.random.permutation(len(X))

    # Calculate the number of samples for the test set
    num_test_samples = int(len(X) * test_size)

    # Split the data into training and testing sets
    X_train = X[indices[num_test_samples:]]
    y_train = y[indices[num_test_samples:]]
    X_test = X[indices[:num_test_samples]]
    y_test = y[indices[:num_test_samples]]

    return X_train, X_test, y_train, y_test

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def main():
    # Load data
    X, y = load_data()

    # Define the proportion of data to be used for testing
    test_size = 0.2

    # Define target time in seconds (4 minutes)
    target_time = 10

    start_time = time.time()
    elapsed_time = 0

    # Perform random sampling until elapsed time exceeds target time
    while elapsed_time < target_time:
        X_train, X_test, y_train, y_test = random_sampling(X, y, test_size)
        accuracy = train_and_evaluate(X_train, y_train, X_test, y_test)
        elapsed_time = time.time() - start_time

    # Print the accuracy
    #print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()

csv_handler.save_data()
