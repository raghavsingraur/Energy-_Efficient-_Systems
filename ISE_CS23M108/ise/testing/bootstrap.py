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

    # Calculate accuracy for this bootstrap sample
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def bootstrap(X, y, num_samples, sample_size):
    # Initialize an empty list to store accuracy scores
    accuracy_scores = []

    # Perform bootstrapping
    for _ in range(num_samples):
        # Randomly sample with replacement from the dataset
        indices = np.random.choice(len(X), size=sample_size, replace=True)
        X_sampled = X[indices]
        y_sampled = y[indices]

        # Split data into training and testing sets
        X_train = X_sampled
        y_train = y_sampled
        X_test = np.delete(X, indices, axis=0)
        y_test = np.delete(y, indices)

        # Train and evaluate the model for this bootstrap sample
        accuracy = train_and_evaluate(X_train, y_train, X_test, y_test)

        # Append accuracy score to the list
        accuracy_scores.append(accuracy)

    # Calculate the average accuracy across all bootstrap samples
    average_accuracy = np.mean(accuracy_scores)
    return average_accuracy

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def main():
    # Load data
    X, y = load_data()

    # Define the number of bootstrap samples and the sample size
    num_samples = 100
    sample_size = len(X)

    # Define target time in seconds (4 minutes)
    target_time = 10

    start_time = time.time()
    elapsed_time = 0

    # Perform bootstrapping until elapsed time exceeds target time
    while elapsed_time < target_time:
        average_accuracy = bootstrap(X, y, num_samples, sample_size)
        elapsed_time = time.time() - start_time

    # Print the average accuracy
    #print("Average accuracy:", average_accuracy)

if __name__ == "__main__":
    main()

csv_handler.save_data()
