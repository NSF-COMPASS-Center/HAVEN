import gc

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from LanguageModel.utils import *
from TransferModel.Analysis import Evaluation
from TransferModel.Analysis.Callbacks import TrainingPlot
from TransferModel.Analysis.Callbacks.AUROC import AUROCCallback


class TargetModel(object):
    def __init__(self, seed=None):
        self.seq_len = None
        if seed is not None:
            tf.random.set_seed(seed)

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        # physical_devices = tf.config.list_physical_devices('GPU')
        # try:
        #    for device in physical_devices:
        #        tf.config.experimental.set_memory_growth(device, True)
        # except:
        #    # Invalid device or cannot modify virtual devices once initialized.
        #    pass

    def split_and_pad(self, *args, **kwargs):
        raise NotImplementedError('Use LM instantiation instead '
                                  'of base class.')

    def gpu_gc(self):
        gc.collect()
        clear_session()

    # X_cats comes in the form of 1 large array, but has beginning and ending vocab separators
    # lengths is length of each sequence
    def fit(self, X, y, valX, valY, date, yVocab):
        X = self.split_and_pad(X)
        y = y.to_numpy()
        valX = self.split_and_pad(valX)
        valY = valY.to_numpy()

        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                   amsgrad=False)
        self.model_.compile(
            loss='sparse_categorical_crossentropy', optimizer=opt,
            metrics=['accuracy']
        )

        dirname = '{}/checkpoints/{}'.format(self.cache_dir_,
                                             self.model_name_)
        mkdir_p(dirname)

        # Callbacks:
        checkpoint = ModelCheckpoint(
            '{}/{}_{}'
            .format(dirname, self.model_name_, self.hidden_dim_) +
            '-{epoch:02d}.hdf5',
            save_best_only=False, save_weights_only=False,
            mode='auto', save_freq='epoch',
        )
        roc = AUROCCallback(X, y, valX, valY, self.batch_size_, date, yVocab)
        plotter = TrainingPlot.TrainingPlot(date)

        # tf fit
        history = self.model_.fit(
            X, y, epochs=self.n_epochs_, batch_size=self.batch_size_,
            validation_data=(valX, valY),
            shuffle=True, verbose=self.verbose_,
            callbacks=[checkpoint, roc, plotter],
        )

        return history

    def predict(self, X, sparse=False):
        X = self.split_and_pad(X)
        y_pred = self.model_.predict(X, batch_size=self.inference_batch_size_)
        if not sparse:
            y_pred = y_pred.argmax(y_pred, axis=1)
        return y_pred

    def transform(self, X):
        X = self.split_and_pad(X)

        tmpModel = Model(inputs=self.model_.input, outputs=self.model_.get_layer('embed_layer').output)

        X_embeddings = tmpModel.predict(X, batch_size=self.inference_batch_size_)

        self.gpu_gc()

        return X_embeddings
