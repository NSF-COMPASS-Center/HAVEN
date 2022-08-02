import numpy as np

from LanguageModel.utils import tprint, iterate_lengths
from TransferModel.Models.TargetModel import TargetModel

from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BiLSTMTargetModel(TargetModel):
    def __init__(
            self,
            seq_len,
            parentModel,
            vocab_size,
            target_size,
            embedding_dim=20,
            hidden_dim=256,
            n_hidden=2,
            dff=512,
            n_epochs=1,
            batch_size=1000,
            inference_batch_size=1500,
            cache_dir='.',
            figDir='.',
            model_name='bilstm_transfer',
            seed=None,
            verbose=False
    ):
        super().__init__(seed=seed, )

        # Init stuff
        self.seq_len_ = seq_len
        self.vocab_size_ = vocab_size
        self.target_size_ = target_size
        self.embedding_dim_ = embedding_dim
        self.hidden_dim_ = hidden_dim
        self.n_hidden_ = n_hidden
        self.dff_ = dff
        self.n_epochs_ = n_epochs
        self.batch_size_ = batch_size
        self.inference_batch_size_ = inference_batch_size
        self.cache_dir_ = cache_dir
        self.model_name_ = model_name
        self.verbose_ = verbose
        self.figDir = figDir


        # Generate model
        if not parentModel:
            input_pre = Input(shape=(seq_len - 1,))
            input_post = Input(shape=(seq_len - 1,))

            embed = layers.Embedding(vocab_size + 1, embedding_dim,
                              input_length=seq_len - 1)
            x_pre = embed(input_pre)
            x_post = embed(input_post)

            for _ in range(n_hidden - 1):
                lstm = layers.LSTM(hidden_dim, return_sequences=True)
                x_pre = lstm(x_pre)
                x_post = lstm(x_post)
            lstm = layers.LSTM(hidden_dim)
            x_pre = lstm(x_pre)
            x_post = lstm(x_post)

            x = layers.concatenate([x_pre, x_post],
                            name='embed_layer')

            output = self.attachTransferHead(x)

            self.model_ = Model(inputs=[input_pre, input_post],
                                outputs=output)
        else:
            # Replace classifier
            x = parentModel.layers[-3].output
            parentModel.trainable = False

            predictions = self.attachTransferHead(x)

            self.model_ = Model(inputs=parentModel.inputs, outputs=predictions)
            assert self.model_.layers[0].trainable == False
            assert self.model_.layers[1].trainable == False
            assert self.model_.layers[2].trainable == False

    # May want to
    # Change this to however you want, it'll be saved in the hdf5 file
    # Maybe resnet is good?
    def attachTransferHead(self, x):
        for sz in (512, 256, 128, 64):
            x = layers.Dense(sz, activation='swish')(x)
        print(self.target_size_)
        return layers.Dense(self.target_size_, activation='softmax', dtype='float32')(x)

    def split_and_pad(self, X):
        # Convert concatenated format back to array separated sequences
        X_pre = X.copy()
        X_post = X.copy()

        if self.verbose_ > 1:
            tprint('Padding {} splitted...'.format(len(X_pre)))

        X_pre = pad_sequences(
            X_pre, maxlen=self.seq_len_-1,
            dtype='int8', padding='pre', truncating='pre', value=0
        )
        if self.verbose_ > 1:
            tprint('Padding {} splitted again...'.format(len(X_pre)))

        # tf post padding
        X_post = pad_sequences(
            X_post, maxlen=self.seq_len_-1,
            dtype='int8', padding='post', truncating='post', value=0
        )

        if self.verbose_ > 1:
            tprint('Flipping...')

        # Reverse order of each
        X_post = np.flip(X_post, 1)
        X = [X_pre, X_post]

        return X
