import numpy as np
from sklearn.cluster import KMeans
import logging
import helpers
import matplotlib.pyplot as plt
logger = logging.getLogger('crgan')


class Test:
    def __init__(self, data, config):
        self.config = config
        self.data = data
        self.id = data.id

    def test(self):
        GDscore = np.load("results/" + self.config.use_id + "/y_hat/GDscore.npy")
        label = self.data.test_label # read from csv
        helpers.load_para(self, "results/" + self.config.use_id + "/models/testparas")
        threshold = self.threshold
        helpers.plotting1(self.id, "Anomaly score (GDScore)", GDscore, threshold, label)
        helpers.indicator(GDscore, label)
