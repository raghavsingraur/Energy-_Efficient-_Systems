import random
from pyJoules.energy_meter import measure_energy
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time
from pyJoules.handler.csv_handler import CSVHandler

csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def train_random_forest(X, y, test_size=0.3, depths=[40, 80], estimators=[400, 800]):
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    elapsed_time = time.time() - start_time
    target_time = 10  # Target execution time in seconds (4 minutes)
    
    while elapsed_time < target_time:
        for eachDepth in depths:
            for eachEstimators in estimators:
                clf = RandomForestClassifier(n_estimators=eachEstimators, max_depth=eachDepth)
                clf.fit(X_train, y_train)
                elapsed_time = time.time() - start_time
                if elapsed_time >= target_time:
                    break
        if elapsed_time >= target_time:
            break

# Load data
X, y = load_digits(return_X_y=True)

# Call the function
train_random_forest(X, y)

csv_handler.save_data() 
