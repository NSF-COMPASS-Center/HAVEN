import numpy as np

from LanguageModel.utils import tprint, iterate_lengths
from TransferModel.Models.host_model import HostLanguageModel

from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BiLSTMHostModel(HostLanguageModel):
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
            model_name='bilstm_host',
            seed=None,
            verbose=False
    ):
        super().__init__(seed=seed, )

        if not parentModel:
            input_pre = Input(shape=(seq_len - 1,))
            input_post = Input(shape=(seq_len - 1,))

            embed = Embedding(vocab_size + 1, embedding_dim,
                              input_length=seq_len - 1)
            x_pre = embed(input_pre)
            x_post = embed(input_post)

            for _ in range(n_hidden - 1):
                lstm = LSTM(hidden_dim, return_sequences=True)
                x_pre = lstm(x_pre)
                x_post = lstm(x_post)
            lstm = LSTM(hidden_dim)
            x_pre = lstm(x_pre)
            x_post = lstm(x_post)

            x = concatenate([x_pre, x_post],
                            name='embed_layer')

            # x = Dense(dff, activation='relu')(x)
            x = Dense(vocab_size + 1)(x)
            output = Activation('softmax', dtype='float32')(x)

            self.model_ = Model(inputs=[input_pre, input_post],
                                outputs=output)
        else:
            # Replace classifier
            x = parentModel.layers[-3].output
            parentModel.trainable = False

            for sz in (512, 256, 128, 64):
                x = layers.Dense(sz, activation='swish')(x)

            predictions = layers.Dense(2, activation='softmax', dtype='float32')(x)
            self.model_ = Model(inputs=parentModel.inputs, outputs=predictions)
            assert self.model_.layers[0].trainable == False
            assert self.model_.layers[1].trainable == False
            assert self.model_.layers[2].trainable == False

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
        print(f"seq_len: {self.seq_len_}")

    def split_and_pad(self, X):
        # Convert concatenated format back to array separated sequences
        X_pre = X.copy()
        X_post = X.copy()

        if self.verbose_ > 1:
            tprint('Padding {} splitted...'.format(len(X_pre)))

        X_pre = pad_sequences(
            X_pre, maxlen=self.seq_len_,
            dtype='int8', padding='pre', truncating='pre', value=0
        )
        if self.verbose_ > 1:
            tprint('Padding {} splitted again...'.format(len(X_pre)))

        # tf post padding
        X_post = pad_sequences(
            X_post, maxlen=self.seq_len_,
            dtype='int8', padding='post', truncating='post', value=0
        )

        if self.verbose_ > 1:
            tprint('Flipping...')

        # Reverse order of each post sequence, because you want it to read right to left
        X_post = np.flip(X_post, 1)
        X = [X_pre, X_post]

        if self.verbose_ > 1:
            tprint('Done splitting and padding.')
        return X