import math

import pandas as pd
import numpy as np


class KNNManhattan:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_train = None
        self.k = None

    def fit(self, x_train: pd.DataFrame, y_train):
        self.x_train = x_train.copy(deep=True)
        self.y_train = y_train.copy(deep=True)

    def predict(self, x_test: pd.DataFrame):

        x_train_distance = self.x_train.join(pd.Series(dtype=int, data=np.zeros(len(self.x_train)), name='distance'))
        x_test_final = x_test.join(pd.Series(dtype=str, name='prediction'))

        self.k = np.ceil(math.sqrt(len(x_test) + len(self.x_train))) // 2 * 2 + 1

        for index_test, row_test in x_test.iterrows():
            for index_train, row_train in self.x_train.iterrows():
                x_train_distance.at[index_train, 'distance'] = sum(abs(row_test - row_train))
            x_train_distance = pd.concat([x_train_distance, self.y_train], axis=1)
            sorted_df = x_train_distance.sort_values(by=['distance'])
            sorted_df = sorted_df.head(int(self.k))
            counts = sorted_df['status'].value_counts()
            x_train_distance.drop(columns=['status'], inplace=True)
            prediction = counts.idxmax()
            x_test_final.at[index_test, 'prediction'] = prediction

        return x_test_final['prediction']