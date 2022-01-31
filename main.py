import pandas as pd

# predict max amount of money that person is ready to pay for new car


# 1. Read data and print last five rows
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
dataset_car_purchase = pd.read_csv('datasets/car_purchase.csv')
print(dataset_car_purchase.tail())


# 2. Show dataset info

print(dataset_car_purchase.info())
print(dataset_car_purchase.describe())






