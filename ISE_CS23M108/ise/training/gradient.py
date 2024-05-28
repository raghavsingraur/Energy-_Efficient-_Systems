import random
from pyJoules.energy_meter import measure_energy
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time
from pyJoules.handler.csv_handler import CSVHandler

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def run_gradient_boosting(X, y, test_size=0.3, test_cases=100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    for test in range(test_cases):
        data_estimators = [1, 10, 20, 70]
        for n in data_estimators:
            if test == 0:
                clf = GradientBoostingClassifier(n_estimators=n)
            else:
                clf = GradientBoostingRegressor(n_estimators=n)
            clf.fit(X_train, y_train)
            """
            if test == 0:
                print('Gradient Boosting Classifier', n, round(clf.score(X_train, y_train), 2), round(clf.score(X_test, y_test), 2))
            else:
                print('Gradient Boosting Regressor', n, round(clf.score(X_train, y_train), 2), round(clf.score(X_test, y_test), 2))
            """

# Load data
X, y = load_digits(return_X_y=True)

# Run the function
run_gradient_boosting(X, y)

csv_handler.save_data()


