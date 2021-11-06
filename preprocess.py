import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
current_path = os.path.abspath(__file__)
logger = logging.getLogger('crgan')


class Dataprocess():
    def __init__(self, id, config):
        self.train = "data/train.csv"
        self.test = "data/test.csv"
        self.id = id
        self.config = config
        self.X_train = None
        self.y_train = []
        self.X_test = None
        self.y_test = []
        self.test_label = None  # read from csv

    def load_data(self):
        if self.config.train:
            csv_data = pd.read_csv(self.train, index_col=0)  #read csv
            df_data = pd.DataFrame(csv_data)  # save as dataframe
            df_data = df_data.fillna(df_data.mean())  # fill none
            data = df_data.values  # transform dataframe to list
            scaler = StandardScaler(with_mean=0, with_std=1)  # the mean and variance of train dataset
            scaler = scaler.fit(data.astype(float))
            data = scaler.transform(data)
            data = self.data_process(data)
            # self.plotting(data)
            self.shape_data(data)
        if self.config.predict:
            csv_data = pd.read_csv(self.test, index_col=0)  # read csv
            df_data = pd.DataFrame(csv_data)  # save as dataframe
            df_data = df_data.fillna(df_data.mean())  # fill none
            data = df_data.values  # transform dataframe to list
            scaler = StandardScaler(with_mean=0, with_std=1)  # the mean and variance of train dataset
            scaler = scaler.fit(data.astype(float))
            data = scaler.transform(data)
            data = self.data_process(data)
            self.shape_data(data, train=False)

    def shape_data(self, arr, train=True):
        data = []
        for i in range((arr.shape[0] - self.config.l_s) // self.config.s_w + 1):
            data.append(arr[i*self.config.s_w: i*self.config.s_w + self.config.l_s])
        if (arr.shape[0] - self.config.l_s) % self.config.s_w != 0:
            data.append(arr[-self.config.l_s:])
        data = np.array(data)
        m, n, p = data.shape
        assert len(data.shape) == 3
        if train:
            # np.random.shuffle(data)
            self.X_train = np.reshape(data, (m, n, p, 1))
        else:
            self.X_test = np.reshape(data, (m, n, p, 1))

    def shape_label(self, label):
        test_label = []
        label = np.reshape(label, (1, -1)).tolist()
        for i in range((len(label[0]) - self.config.l_s) // self.config.s_w + 1):
            test_label += label[0][i * self.config.s_w: i * self.config.s_w + self.config.l_s]
        if (len(label[0]) - self.config.l_s) % self.config.s_w != 0:
            test_label += label[0][-self.config.l_s:]
        return test_label

    def data_process(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(data.astype(float))
        data = scaler.transform(data)
        return data

    def plotting(self, data):
        plt.figure()
        data = data[1000:4000, 24:33]
        font1 = {"family": "Times New Roman", "fontsize": 20}
        data_1 = data
        for i in range(len(data_1)):
            for j in range(len(data_1[0])):
                data_1[i][j] += j
        plt.plot(data_1)
        plt.tick_params(labelsize=18)
        plt.xlabel("Time steps", font1)
        plt.ylabel("Parameters", font1)
        plt.show()
        plt.savefig(os.path.abspath(current_path) + "/../results/" + self.id + "/figures/paras.png")






