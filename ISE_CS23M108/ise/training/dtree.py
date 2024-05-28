import pyJoules  
import random
from pyJoules.energy_meter import measure_energy
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import time
from pyJoules.handler.csv_handler import CSVHandler


def run_decision_tree(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate a decision tree classifier.

    Parameters:
    - X: Input features
    - y: Target labels
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Random seed for reproducibility

    Returns:
    - accuracy: Accuracy of the classifier
    - classification_report: Classification report containing precision, recall, and F1-score
    """

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the DecisionTreeClassifier
    clf = DecisionTreeClassifier()

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred)

    return accuracy, report

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Define target time in seconds (4 minutes)
    target_time = 10

    start_time = time.time()
    elapsed_time = 0

    # Run the decision tree classifier until elapsed time exceeds target time
    while elapsed_time < target_time:
        accuracy, report = run_decision_tree(X, y)
        elapsed_time = time.time() - start_time

    # Print results
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()

csv_handler.save_data()    

