import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

co2_dataFile = r"FuelConsumptionCo2.csv"

def main():
    # read the data
    co2_data = pd.read_csv(co2_dataFile)

    # select certain fields of the data
    cdf = co2_data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
              'CO2EMISSIONS']]

    # Plot the emissions wrt engine size
    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.savefig("emissions_vs_enginsize.png")

    # Randomly select data for training and testing purposes
    xx = np.random.rand(len(co2_data)) <0.8
    train = cdf[xx]
    test = cdf[xx]

    regr = linear_model.LinearRegression()
    x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(x,y)
    print(f"Regression Coefficients: {regr.coef_}")

    # Predict using the test data set
    y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print("Mean Squared Error (MSE) : %.2f"  % np.mean((y_hat - y) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x, y))


if __name__ == "__main__":
    main()