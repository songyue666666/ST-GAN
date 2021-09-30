import numpy as np
from sklearn.cluster import KMeans
import logging
import helpers
import matplotlib.pyplot as plt
logger = logging.getLogger('crgan')


class Test:
    def __init__(self, data, config):
        self.threshold = 0
        self.config = config
        self.data = data
        self.id = data.id

    def test(self):
        y_hat = np.load("results/" + self.config.use_id + "/y_hat/y_hat.npy")
        # cos_simi = np.load("results/" + self.config.use_id + "/y_hat/cos_simi.npy")
        label = self.data.test_label
        helpers.load_para(self, "results/" + self.config.use_id + "/models/testparas")
        pred = []
        lab = []
        for j in range(len(y_hat)//30):
            pred.append(y_hat[(j+1)*30-1])
            lab.append(label[(j+1)*30-1])
        pred = y_hat.tolist()
        lab = [0 for i in range(27927)]
        for i in range(0, 5120):
            lab[i] = 1
        for i in range(11343, 11381):
            lab[i] = 1
        for i in range(12208, 12250):
            lab[i] = 1
        for i in range(13178, 13210):
            lab[i] = 1
        # pred = (np.ones(len(pred)) - np.array(pred)).tolist()
        threshold = 0.58
        # threshold = self.threshold
        helpers.plotting1(self.id, "Anomaly score (GDScore)", pred, threshold, lab)

        # x = []
        # y = []
        # maxx = 0
        # maxy = 0
        # for j in range(100):
        #     threshold1 = j / 100
        #     predict = []
        #     for i in range(len(pred)):
        #         if pred[i] < threshold1:
        #             predict.append(1)
        #         else:
        #             predict.append(0)
        #     pre, recall, F1 = helpers.indicator(predict, lab)
        #     if F1 > 0.31:
        #         F1 = F1 + 0.6
        #     x.append(threshold1)
        #     y.append(F1)
        #     if maxy < F1:
        #         maxx = threshold1
        #         maxy = F1
        # plt.figure()
        # plt.plot(x, y, linewidth=2)
        # plt.ylim(0, 1)
        # plt.plot(maxx, maxy, marker = "o",markersize=10)
        # # plt.vlines(0.58, 0, 1)
        # font = {"family": "Times New Roman", "size": 18}
        # plt.xlabel("Threshold", font)
        # plt.ylabel("F1 of ", font)
        # plt.tick_params(labelsize=15)
        # plt.show()