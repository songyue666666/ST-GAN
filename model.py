from keras.models import Sequential, load_model, Model
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.layers import Reshape, BatchNormalization, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import os
import logging
import helpers

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('crgan')


class crnn_model:
    def __init__(self, config, run_id, data):
        self.config = config
        self.data = data
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        self.latent_dim = self.config.latent_dim # the dimension of the latent vector
        self.g_weight = 0 # the weight of generator used in model test
        self.g_threshold = 0 # the threshold used in model test
        self.d_threshold = 0  # the threshold used in model test
        self.median_y_hat = 0
        if self.config.train:
            self.input_shape = (self.config.l_s, self.data.X_train.shape[2], 1)  # (ls, 38, 1)
            # Build and compile the discriminator
            self.discriminator = self.discriminator()
            self.discriminator.compile(loss=self.config.loss_metric,
                                       optimizer=eval(self.config.discriminator_optimizer),
                                       metrics=['accuracy'])
            # Build the generator
            self.generator = self.generator()
            # The generator takes noise as input and generates imgs
            z = Input(shape=(self.latent_dim,))
            sample = self.generator(z)
            # For the combined model we will only train the generator
            self.discriminator.trainable = False
            # The discriminator takes generated images as input and determines validity
            valid = self.discriminator(sample)
            # The combined model  (stacked generator and discriminator)
            # Trains the generator to fool the discriminator
            self.combined = Model(z, valid)
            self.combined.compile(loss=self.config.loss_metric, optimizer=eval(self.config.generator_optimizer))
            self.train_new()
            self.save()

    def generator(self):
        model = Sequential()
        model.add(Dense(self.config.l_s * self.data.X_train.shape[2], activation="relu", input_dim=self.latent_dim))
        model.add(Reshape(self.input_shape))
        model.add(UpSampling2D())
        # Convolution layer
        model.add(Conv2D(64, (2, 5), padding='same', kernel_initializer='he_normal'))  # (None, ls, 38, 64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        # model.add(MaxPooling2D(pool_size=(2, 2), name='max1'))  # (None, ls/2, 19, 64)
        model.add(Conv2D(64, (5, 2), padding='same', kernel_initializer='he_normal'))  # (None, ls/2, 19, 64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(UpSampling2D())
        # CNN to RNN
        model.add(Reshape(target_shape=((self.config.l_s, -1))))  # (None, ls, -1)
        model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))  # (None, ls, 32)
        # RNN layer
        model.add(LSTM(80, input_shape=(None, 32), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(40, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(self.config.l_s * self.data.X_train.shape[2], activation='tanh'))
        model.add(Reshape(self.input_shape))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        gen_sample = model(noise)
        return Model(noise, gen_sample)

    def discriminator(self):
        model = Sequential()
        # Convolution layer
        model.add(Conv2D(32, (3, 3), input_shape=self.input_shape, padding='same', kernel_initializer='he_normal'))  # (None, ls, 38, 32)
        # model.add(MaxPooling2D(pool_size=(2, 2)))  # (None, ls/2, 19, 64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (2, 5), padding='same', kernel_initializer='he_normal'))  # (None, ls, 38, 64)
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (5, 2), padding='same', kernel_initializer='he_normal'))  # (None, ls/2, 19, 64)
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # CNN to RNN
        model.add(Reshape(target_shape=(self.config.l_s, -1), name='reshape'))  # (None, ls, -1)
        model.add(Dense(32, activation='relu', kernel_initializer='he_normal', name='dense1'))  # (None, ls, 32)
        # RNN layer
        model.add(LSTM(80, input_shape=(None, 32), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(40, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        sample = Input(shape=self.input_shape)
        validity = model(sample)
        return Model(sample, validity)

    def train_new(self):
        # Adversarial ground truths
        valid = np.ones((self.config.lstm_batch_size, 1))
        fake = np.zeros((self.config.lstm_batch_size, 1))
        for epoch in range(self.config.epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            X_train = self.data.X_train
            idx = np.random.randint(0, X_train.shape[0], self.config.lstm_batch_size)
            samples = X_train[idx]
            noise = np.random.normal(0, 1, (self.config.lstm_batch_size, self.latent_dim))
            # Generate a batch of new images
            gen_samples = self.generator.predict(noise)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(samples, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_samples, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (self.config.lstm_batch_size, self.latent_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
            # ---------------------
            #  claculate the aggregative indicator
            # ---------------------
            acc = 100 * d_loss[1] # the accuracy metric of discriminator
            # calculate the cosine similarity
            cos_simi_list = np.abs(helpers.cos_similarity(gen_samples, samples))
            cos_simi = 100 * np.mean(cos_simi_list) # the cosine similarity of generator
            g_d = 0.5 * (acc + cos_simi) # the aggregative indicator
            # Plot the progress
            logger.info("epoch: %d [D loss: %f] [G loss: %f] [D accuracy: %.2f%%] [G cosine_similarity: %.2f%%] [G cosine_similarity + D accuracy: %.2f%%]" % (epoch, d_loss[0], g_loss, acc, cos_simi, g_d))
            # judge whether reach to epochs and whether to early stop
            if epoch == self.config.epochs - 1 or (self.config.early_stop and g_d > self.config.early_stop_threshold):
                # save the parameters used in model test
                self.g_weight = cos_simi / (acc + cos_simi)
                y_hat = self.discriminator.predict(samples)
                min_y_hat = np.min(y_hat)
                max_y_hat = np.max(y_hat)
                self.median_y_hat = np.median(y_hat)
                mean_y_hat = np.mean(y_hat)
                min_cos_simi = np.min(cos_simi_list)
                self.g_threshold = min_cos_simi
                self.d_threshold = np.min([max_y_hat - self.median_y_hat, self.median_y_hat - min_y_hat])
                logger.info("generator weight: %.2f, y_hat: %.2f, cosine similarity: %.2f" % (self.g_weight, min_y_hat, min_cos_simi))
                info = {"g_weight" : self.g_weight, "g_threshold" : self.g_threshold, "d_threshold" : self.d_threshold, "median_y_hat" : self.median_y_hat}
                np.save("results/" + self.run_id + "/models/testparas.npy", info)
                break

    def save(self):
        """
        Save trained model.
        """
        self.combined.save(os.path.join('results'
                                     , self.run_id, 'models',
                                     'combined_model.h5'))
        self.discriminator.save(os.path.join('results'
                                        , self.run_id, 'models',
                                        'discriminator.h5'))
        self.generator.save(os.path.join('results'
                                        , self.run_id, 'models',
                                        'generator.h5'))


