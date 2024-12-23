from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class General_Processor:
    def __init__(self):
        self.length = 0
        self.model = MinMaxScaler()

    def fit(self, data):
        res = self.model.fit_transform(data.reshape(-1, 1))
        self.length += res.shape[1] # sum of feature_dim

    def transform(self, data):
        return self.model.transform(data.reshape(-1, 1))

    def inverse_transform(self, data):
        res = self.model.inverse_transform(data)
        res = res.reshape(-1, 1)
        return res


class Processor:
    def __init__(self):
        self.model = MinMaxScaler()
        self.names, self.models, self.dim = [], [], 0

    def fit(self, data):
        self.names = []
        self.models = []
        self.dim = 0
        values_matrix = data.values

        for i, col in enumerate(data.columns):
            value = values_matrix[:, i]
            self.names.append(col)
            model = General_Processor()
            model.fit(value)
            self.models.append(model)
            self.dim += model.length

    def transform(self, data):
        cols = []
        values_matrix = data.values
        for i, col in enumerate(data.columns):
            value = values_matrix[:, i]
            preprocessed_col = self.models[i].transform(value)
            cols.append(preprocessed_col)
        return np.concatenate(cols, axis=1)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        res = []
        j = 0
        for model in self.models:
            value = data[:, j:j + model.length]
            x = model.inverse_transform(value)
            res.append(x)
            j += model.length
        matrix = np.concatenate(res, axis=1)
        return pd.DataFrame(matrix, columns=self.names)

