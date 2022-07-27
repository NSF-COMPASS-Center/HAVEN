#!/usr/src/env python

import types
# Python libs
from typing import List

# ML libs
import tensorflow as tf
from keras.models import load_model

import TransferModel.Analysis.Visualization as Visualization
import TransferModel.DataUtils.DataProcessor as DataProcessor
# Imports from language model
from LanguageModel.mutation import *
# Our modules
from TransferModel.Analysis import Evaluation
from TransferModel.DataUtils import Ingestion
from TransferModel.DataUtils.Vocabularies import AAs
from TransferModel.Models.Utils import get_target_model


class HepHost():
    def __init__(self, model_name: str, checkpoint: str = None, datasets=None, targetNames: List[str] = ["Human"],
                 targetKey: str = "Host", transferCheckpoint: str = None, namespace: str = "hep", seed: int = 1,
                 dim: int = 512, batch_size: int = 64, inf_batch_size: int = 128, n_epochs: int = 11, n_hidden: int = 2,
                 embedding_dim=20, train: bool = False, embed: bool = False, semantics: bool = False,
                 combfit: bool = False, reinfection: bool = False, train_split: float = 0.8, test: bool = False):
        self.args = types.SimpleNamespace()
        self.args.model_name = model_name
        self.args.seed = seed
        self.args.namespace = namespace
        self.args.embedding_dim = embedding_dim
        self.args.dim = dim
        self.args.batch_size = batch_size
        self.args.inf_batch_size = inf_batch_size
        self.args.checkpoint = checkpoint
        self.args.n_epochs = n_epochs
        self.args.n_hidden = n_hidden
        self.args.train = train
        self.args.train_split = train_split
        self.args.test = test
        self.args.embed = embed
        self.args.semantics = semantics
        self.args.combfit = combfit
        self.args.reinfection = reinfection
        self.args.datasets = datasets
        self.args.targetNames = targetNames
        self.args.targetKey = targetKey
        self.args.transferCheckpoint = transferCheckpoint

        assert datasets

        # Set seeds
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        tf.random.set_seed(self.args.seed)

    def start(self):
        # Initialization
        vocabulary = DataProcessor.initilizeVocab(AAs)
        vocab_size = len(vocabulary) + 2  # For padding characters
        yVocab = DataProcessor.initilizeVocab(self.args.targetNames)
        yVocab['not-in-dictionary'] = 0

        # Load dataframe
        df = Ingestion.loadFastaFiles(self.args.datasets)
        seq_len = int(df.seq.map(len).max()) + 2
        df = DataProcessor.featurize_df(df, vocabulary, yVocab, self.args.targetKey)

        model = None
        # Load in for transfer learning and if we won't overwrite it later
        if self.args.transferCheckpoint and not self.args.checkpoint:
            print("DEBUG: Loaded transfer checkpoint!")
            model = load_model(self.args.transferCheckpoint)

        # Make host model
        hostModel = get_target_model(self.args, model, seq_len,  vocab_size, len(yVocab))

        # Regular checkpoint of weights
        if self.args.checkpoint:
            print("DEBUG: Loading checkpoint model!")
            hostModel.model_ = load_model(self.args.checkpoint)
            hostModel.model_.summary()

        date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        if self.args.train:
            print("DEBUG: Training start")
            hostModel.model_.summary()
            train_df, test_df = DataProcessor.split_df(df, self.args.train_split)

            h = hostModel.fit(train_df['X'], train_df['y'], test_df['X'],
                              test_df['y'], date, yVocab)

        if self.args.test:
            print("DEBUG: Testing start")
            train_df, test_df = DataProcessor.split_df(df, self.args.train_split)
            print(train_df, test_df)
            y_pred = hostModel.predict(test_df['X'], sparse=True)
            testAurocs = Evaluation.report_auroc(test_df['y'], y_pred, labelVocab=yVocab,
                                                 filename=f"hep_bilstm_host_AUROC_{date}")
            yVocabInverse = {y:x for x, y in yVocab.items()}
            print({yVocabInverse[i]: auroc for i, auroc in enumerate(testAurocs)})
            print(Evaluation.report_accuracy_per_class(test_df['y'], y_pred, yDict=yVocab))
            print(Evaluation.report_accuracy(test_df['y'], y_pred))


'''
        if self.args.embed:
            if self.args.checkpoint is None and not self.args.train:
                raise ValueError('Model must be trained or loaded '
                                 'from checkpoint.')
            no_embed = {'hmm'}
            if self.args.model_name in no_embed:
                raise ValueError('Embeddings not available for models: {}'
                                 .format(', '.join(no_embed)))
            self.analyze_embedding(self.args, model, seqs, vocabulary)
'''
