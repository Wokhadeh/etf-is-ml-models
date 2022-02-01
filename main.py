import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# predict max amount of money that person is ready to pay for new car


# 1. Read data and print last five rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
dataset_car_purchase = pd.read_csv('datasets/car_purchase.csv')
print(dataset_car_purchase.tail())


# 2. Show dataset info

print(dataset_car_purchase.info())
print(dataset_car_purchase.describe())

# 3. Show correlation between parameters

sns.heatmap(dataset_car_purchase.corr())
plt.show()

# Strong correlation (age, max_purchase_amount), (salary, max_purchase_amount)
# Medium correlation (net_worth, max_purchase_amount)
# Weak correlation (credit_card_debt, max_purchase_amount) !?!? and  (customer_id, max_purchase_amount)
# There is no other important correlation between potential parameters

# 4. Choose parameters for model

# Not important parameters: customer_id
# Maybe not important parameters: credit_card_debt !?!? , age !?!? (decided to include it in model anyway)
data_train = dataset_car_purchase[['age', 'annual_salary', 'credit_card_debt', 'net_worth']]
labels = dataset_car_purchase['max_purchase_amount']

# 5. Prediction using linear regression from sklearn

lr_skl_model = LinearRegression()

X_train, X_test, Y_train, Y_test = train_test_split(data_train, labels, random_state=1)

lr_skl_model.fit(X_train.values.reshape(-1, 4), Y_train.values)
predicted = lr_skl_model.predict(X_test.values.reshape(-1, 4))
print('Model: ',end='')
for pair in zip(lr_skl_model.coef_[:-1], data_train.columns):
    print(str(pair[0]) + ' * ' + pair[1] + ' + ',end='')
print(lr_skl_model.coef_[-1])
print('Mean squared error function: ' + str((1/(2*len(predicted)))*sum(Y_test.sub(predicted)**2)))
print('Score: ' + str(lr_skl_model.score(X_test.values.reshape(-1, 4), Y_test.values)))


#ser_predicted = pd.Series(data=predicted, name='predicted', index=X_test.index)
# print(ser_predicted)
# new_df = pd.concat([Y_test, ser_predicted], axis=1)
# print(new_df)






