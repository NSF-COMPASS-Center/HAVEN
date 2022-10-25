#!/usr/src/env python

import types
# Python libs
from typing import List

# ML libs
import tensorflow as tf
from keras.models import load_model

import TransferModel.DataUtils.DataProcessor as DataProcessor
# Imports from language model
import TransferModel.Models.Utils
from LanguageModel.mutation import *
# Our modules
from TransferModel.DataUtils import Ingestion
from TransferModel.DataUtils.Vocabularies import AAs
from TransferModel.Models.Utils import get_target_model


class HepHost():
    def __init__(self, model_name: str, checkpoint: str = None, datasets=None, targetNames: List[str] = ["Human"],
                 targetKey: str = "Host", transferCheckpoint: str = None, namespace: str = "hep", seed: int = 1,
                 dim: int = 512, batch_size: int = 64, inf_batch_size: int = 128, n_epochs: int = 11, n_hidden: int = 2,
                 embedding_dim=20, embedTargets: List[str] = None, embedding_cache: bool = False, train: bool = False,
                 embed: bool = False, semantics: bool = False,
                 combfit: bool = False, reinfection: bool = False, train_split: float = 0.8, test: bool = False, visualize_dataset: bool = True, outputDir: str="output", output_model_name: str="bilstmHost"):
        self.args = types.SimpleNamespace()
        self.args.model_name = model_name
        self.args.seed = seed
        self.args.outputDir = outputDir
        self.args.namespace = namespace
        self.args.targetDir = f"{outputDir}/target/{namespace}_{model_name}_{dim}"
        mkdir_p(self.args.targetDir)
        self.args.figDir = f"{outputDir}/figures/{namespace}_{model_name}_{dim}"
        mkdir_p(self.args.figDir)
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
        self.args.embedding_dim = embedding_dim
        self.args.embedding_cache = embedding_cache
        self.args.semantics = semantics
        self.args.combfit = combfit
        self.args.reinfection = reinfection
        self.args.datasets = datasets
        self.args.targetNames = targetNames
        self.args.targetKey = targetKey
        self.args.transferCheckpoint = transferCheckpoint
        self.args.embedTargets = embedTargets
        self.args.visualize_dataset = visualize_dataset
        self.args.output_model_name = output_model_name

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


        if self.args.visualize_dataset:
            print("DEBUG: visualizing dataset")
            for col in self.args.embedTargets:
                df[col].value_counts().plot(kind='bar')
                plt.show()

        seq_len = int(df.seq.map(len).max()) + 2
        df = DataProcessor.featurize_df(df, vocabulary, yVocab, self.args.targetKey)

        model = None
        # Load in for transfer learning and if we won't overwrite it later
        if self.args.transferCheckpoint and not self.args.checkpoint:
            print("DEBUG: Loaded transfer checkpoint!")
            model = load_model(self.args.transferCheckpoint)
            model.summary()

        # Make host model
        hostModel = get_target_model(self.args, model, seq_len, vocab_size, len(yVocab))

        # Regular checkpoint of weights
        if self.args.checkpoint:
            print("DEBUG: Loading checkpoint model!")
            hostModel.model_ = load_model(self.args.checkpoint)
            hostModel.seq_len_ = hostModel.model_.layers[0].input_shape[0][1] + 1
            hostModel.model_.summary()

        #if self.args.train or self.args.test or self.args.embed:
        train_df, test_df = DataProcessor.split_df(df, self.args.train_split, self.args.seed)
        Ingestion.pandasToFastaFile(train_df, f'hep-manual-{self.args.seed}-{self.args.train_split}.fasta')


        date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        if self.args.train:
            print("DEBUG: Training start")
            trainHistory = TransferModel.Models.Utils.fit_model(hostModel, train_df, test_df, yVocab, date)

        if self.args.test:
            print("DEBUG: Testing start")
            '''
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(train_df)
            print(test_df)
            '''
            TransferModel.Models.Utils.test_model(self.args, hostModel, test_df, yVocab, date)

        if self.args.embed:
            print("DEBUG: Testing embedding")
            print(test_df)
            TransferModel.Models.Utils.analyze_embedding(self.args, hostModel, test_df)
