import numpy as np
import pyJoules  
from pyJoules.energy_meter import measure_energy
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
from pyJoules.handler.csv_handler import CSVHandler


# Define the function to measure energy consumption
csv_handler = CSVHandler('result.csv')
@measure_energy(handler=csv_handler)
def run_linear_regression():
    # Generate some random data for demonstration purposes
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)   # Generate 100 random numbers between 0 and 2
    y = 4 + 3 * X + np.random.randn(100, 1)  # Linear equation with some noise

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define target time in seconds (4 minutes)
    target_time = 4 * 60

    start_time = time.time()
    elapsed_time = 0

    # Run the linear regression process until elapsed time exceeds target time
    while elapsed_time < target_time:
        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        #print("Mean Squared Error:", mse)

        # Plot the training data and the linear regression line
        plt.scatter(X_train, y_train, color='blue', label='Training Data')
        plt.scatter(X_test, y_test, color='red', label='Testing Data')
        plt.plot(X_test, y_pred, color='green', linewidth=3, label='Linear Regression')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear Regression')
        plt.legend()
        #plt.show()

        elapsed_time = time.time() - start_time

# Call the function to measure energy consumption
run_linear_regression()

csv_handler.save_data()    

