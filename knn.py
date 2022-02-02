import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from knn_manhattan import KNNManhattan

# predict state of car using knn algorithm

# 1. Read data and print last five rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
dataset_car_state = pd.read_csv('datasets/car_state.csv')
print(dataset_car_state.tail())

# 2. Show dataset info
print(dataset_car_state.info())
print(dataset_car_state.describe())

# 3. Show dependency between output and input parameters (showing two discrete parameters => countplot)

for col in dataset_car_state.columns[:-1]:
    cp = sns.countplot(x=col, hue="status", data=dataset_car_state)
    #plt.show()

# 4. Choose parameters for model and transform data

# Potential not important parameters are doors and trunk_size (decided to include it in model anyway)

# Transform all categorical attributes

ohe = OneHotEncoder(dtype=int, sparse=False)
for col in dataset_car_state.columns[:-1]:
    ohe_matrix = ohe.fit_transform(dataset_car_state[col].to_numpy().reshape(-1, 1))
    dataset_car_state.drop(columns=[col],inplace=True)
    dataset_car_state=dataset_car_state.join(pd.DataFrame(data=ohe_matrix, columns=ohe.get_feature_names_out([col])))

#print(dataset_car_state)

data_train = dataset_car_state[dataset_car_state.columns[1:]]
labels = dataset_car_state[dataset_car_state.columns[0]]

# 5. Prediction using KNN algorithm from sklearn

X_train, X_test, Y_train, Y_test = train_test_split(data_train, labels, random_state=1)

knn_skl_model = KNeighborsClassifier()
knn_skl_model.fit(X_train,Y_train)
predicted = knn_skl_model.predict(X_test)

# ser_predicted = pd.Series(data=predicted, name='predicted', index=X_test.index)
# new_df = pd.concat([Y_test, ser_predicted], axis=1)
# print(new_df)


print('Score: ' + str(knn_skl_model.score(X_test, Y_test)))

# 6. Prediction using self-implemented KNN algorithm

knn_my_model = KNNManhattan()
knn_my_model.fit(X_train, Y_train)
predicted_my_model=knn_my_model.predict(X_test)
new_df = pd.concat([Y_test, predicted_my_model], axis=1)
#print(new_df)
df1 = new_df[(new_df['status'] == new_df['prediction'])]
print(df1)
print('Score: ' + str(len(df1)/len(new_df)))

