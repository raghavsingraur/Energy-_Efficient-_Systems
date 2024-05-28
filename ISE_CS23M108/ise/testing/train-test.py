import pyJoules  
import random
from pyJoules.energy_meter import measure_energy
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import platform
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


csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def main():
    # Load data
    X, y = load_data()
    #print(platform.python_version())
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define target time in seconds (4 minutes)
    target_time = 10

    start_time = time.time()
    elapsed_time = 0

    # Train and evaluate the model until elapsed time exceeds target time
    while elapsed_time < target_time:
        # Train and evaluate the model
        accuracy = train_and_evaluate(X_train, y_train, X_test, y_test)
        elapsed_time = time.time() - start_time

    # Print the accuracy
    #print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()


csv_handler.save_data()

