import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
current_path = os.path.abspath(__file__)
logger = logging.getLogger('crgan')


class Dataprocess():
    def __init__(self, id, config):
        self.train = "data/train.csv"
        self.test = "data/test3.csv"
        self.id = id
        self.config = config
        self.X_train = None
        self.y_train = []
        self.X_test = None
        self.y_test = []
        self.test_label = None

    def load_data(self):
        if self.config.train:
            csv_data = pd.read_csv(self.train, index_col=0)  #read csv
            df_data = pd.DataFrame(csv_data)  # save as dataframe
            df_data = df_data.fillna(df_data.mean())  # fill none
            data = df_data.values  # transform dataframe to list
            scaler = StandardScaler(with_mean=0, with_std=1)  # the mean and variance of train dataset
            scaler = scaler.fit(data.astype(float))
            data = scaler.transform(data)
            # np.save('train.npy', data)  # data save as .npy, using for train、offline test
            # file = np.load(self.train, "r")
            # data = file[:, 0:-1]  # the last column is label
            data = self.data_process(data)
            # self.plotting(data)
            self.shape_data(data)
        if self.config.predict:
            # file = np.load(self.test, "r")
            # data = file[:, 0:-1]  # the last column is label
            # label = file[:, -1]
            # self.test_label = self.shape_label(label)
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

    def pca(self, X_n, n_components):
        pca = PCA(n_components, svd_solver='full')
        pca.fit(X_n)
        ex_var = pca.explained_variance_ratio_
        pc = pca.components_
        # projected values on the principal component
        T_n = np.matmul(X_n, pc.transpose(1, 0))
        return T_n

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

        # m, n = data.shape
        # f, ax = plt.subplots(figsize=(100, n))
        # sns.heatmap(data.transpose(), ax=ax, vmax=1, vmin=0, cbar=True)
        # ax.set_xlabel('Time steps', font1)
        # ax.set_ylabel('Parameters', font1)
        # plt.xticks(())
        # plt.tick_params(labelsize=18)
        # # plt.yticks(())
        # plt.show()






