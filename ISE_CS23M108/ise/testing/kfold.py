
import pyJoules  
import random
from pyJoules.energy_meter import measure_energy
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
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

    # Calculate accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def k_fold_cross_validation(X, y, k):
    # Initialize an empty list to store accuracy scores
    accuracy_scores = []

    # Initialize a KFold object
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Iterate over each fold
    for train_index, test_index in kf.split(X):
        # Split data into training and testing sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train and evaluate the model for this fold
        accuracy = train_and_evaluate(X_train, y_train, X_test, y_test)

        # Append accuracy score to the list
        accuracy_scores.append(accuracy)

    # Calculate the average accuracy across all folds
    average_accuracy = np.mean(accuracy_scores)
    return average_accuracy

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def main():
    # Load data
    X, y = load_data()

    # Define the number of folds for cross-validation
    k = 5

    # Define target time in seconds (4 minutes)
    target_time = 10

    start_time = time.time()
    elapsed_time = 0

    # Perform k-fold cross-validation until elapsed time exceeds target time
    while elapsed_time < target_time:
        average_accuracy = k_fold_cross_validation(X, y, k)
        elapsed_time = time.time() - start_time

    # Print the average accuracy
    #print("Average accuracy:", average_accuracy)

if __name__ == "__main__":
    main()

csv_handler.save_data()
