from keras.models import load_model
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import helpers
logger = logging.getLogger('crgan')


class Score:
    def __init__(self, model, config, data):
        self.data = data
        self.config = config
        self.run_id = data.id
        self.latent_dim = model.latent_dim
        self.cos_simi = []
        self.anomaly_score = []

        if self.config.train:
            self.discriminator = model.discriminator
            self.generator = model.generator
            self.g_weight = model.g_weight  # the weight of generator and discriminator
            self.g_threshold = model.g_threshold  # the threshold used to judge positive sample
            self.d_threshold = model.d_threshold
            self.threshold = model.threshold
            self.median_y_hat = model.median_y_hat
        else:
            self.load()
            helpers.load_para(self, "results/" + self.config.use_id + "/models/testparas")

    def load(self):
        """
        Load model for channel.
        """
        logger.info('Loading pre-trained model')
        self.discriminator = load_model(os.path.join('results', self.config.use_id,
                                                     'models', 'discriminator.h5'))
        # self.combined = load_model(os.path.join('results', self.config.use_id,
        #                                      'models', 'combined_model.h5'))
        self.generator = load_model(os.path.join('results', self.config.use_id,
                                                'models', 'generator.h5'))

    def batch_predict(self):
        num_batches = self.data.X_test.shape[0] // self.config.batch_size
        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'
                             .format(self.config.l_s, self.data.X_test.shape[0]))
        # simulate data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size
            # noise = np.random.normal(0, 1, (self.config.batch_size, self.latent_dim))
            if i == num_batches:
                idx = self.data.X_test.shape[0]
                # noise = np.random.normal(0, 1, (idx-prior_idx, self.latent_dim))
            X_test_batch = self.data.X_test[prior_idx:idx]
            y_hat_batch = self.discriminator.predict(X_test_batch)
            y_hat_batch = np.reshape(y_hat_batch, (1, -1)).tolist()[0]  # the probability of the sample
            # logger.info("predicted by discriminator: ")
            # logger.info(y_hat_batch)
            y_hat = []
            y_hat += y_hat_batch
            y_hat = np.reshape(y_hat, (-1,)).tolist()
            self.data.y_test += y_hat
            gen_samples = self.generator.predict(X_test_batch)
            cos_simi = np.abs(helpers.cos_similarity(gen_samples, X_test_batch)) # the cosine similarity of generator
            # logger.info("cosine similarity calculated by generator: ")
            # logger.info(cos_simi)
            self.cos_simi += cos_simi.tolist()
            # the aggregative indicator
            GDScore = np.array(y_hat)*(1-self.g_weight) + np.array(np.mean(cos_simi))*self.g_weight
            # logger.info("the indicator value calculated by weights: ")
            # logger.info(GDScore)
            self.anomaly_score += (np.ones(len(GDScore)) - GDScore).tolist()
        np.save(os.path.join('results', self.run_id, 'y_hat', 'y_hat.npy'), self.data.y_test)
        np.save(os.path.join('results', self.run_id, 'y_hat', 'cos_simi.npy'), self.cos_simi)
        helpers.plotting2(self.run_id, "GDScore", self.anomaly_score, self.threshold)
        np.save(os.path.join('results', self.run_id, 'y_hat', 'GDScore.npy'), self.anomaly_score)
        return self.data


