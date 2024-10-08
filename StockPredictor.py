import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

# Fill initialized lists with relevant data from CSV file
def get_data(filename):
    try:
        with open(filename, 'r') as csvfile:
            csvFileReader = csv.reader(csvfile)
            next(csvFileReader)  
            for row in csvFileReader:
                if len(row) < 2: 
                    continue
                try:
                    date = int(row[0].split('_')[0])
                    price = float(row[1]) 
                    dates.append(date)
                    prices.append(price)
                except ValueError as e:
                    print(f"Value error for row {row}: {e}")
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def predict_prices(dates, prices, x):
    # Reshape dates for SVR
    dates = np.reshape(dates, (len(dates), 1))  # Reshape list into a 2D array for SVR

    # Create Support Vector Regression models
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # Train the models
    svr_rbf.fit(dates, prices)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)

    # Plot the graph
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    # Return predictions for the given date x
    return svr_rbf.predict([[x]])[0], svr_lin.predict([[x]])[0], svr_poly.predict([[x]])[0]

# Load the data
get_data('aapl.csv')


if dates and prices:
    
    predicted_price = predict_prices(dates, prices, 29)

    # Print predicted prices
    print(predicted_price)
else:
    print("No data available for predictions.")
