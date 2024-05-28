import random
import pyJoules
from pyJoules.energy_meter import measure_energy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from pyJoules.handler.csv_handler import CSVHandler

def load_data():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def train_classifier(X_train, y_train, X_test, y_test, n_neighbors=5):
    # Initialize KNN classifier
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def main():
    # Load data
    X, y = load_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define target time in seconds (4 minutes)
    target_time = 10

    start_time = time.time()
    elapsed_time = 0

    # Train the KNN classifier until elapsed time exceeds target time
    while elapsed_time < target_time:
        # Train the KNN classifier
        accuracy = train_classifier(X_train, y_train, X_test, y_test)
        
        elapsed_time = time.time() - start_time

    # Print the accuracy
    #print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()

csv_handler.save_data() 
