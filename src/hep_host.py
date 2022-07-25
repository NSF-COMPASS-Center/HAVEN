#!/usr/src/env python

import types
# Python libs
from typing import List

# ML libs
import tensorflow as tf
from keras.models import load_model

import TransferModel.Analysis.Visualization as Visualization
import TransferModel.DataUtils.Preprocessor as Preprocessor
# Imports from language model
from LanguageModel.mutation import *
# Our modules
from TransferModel.Analysis.Evaluation import report_auroc_host
from TransferModel.DataUtils import Ingestion
from TransferModel.DataUtils.Vocabularies import AAs
from TransferModel.Models.Utils import batch_train_host, get_model_host


class HepHost():
    def __init__(self, model_name: str, checkpoint: str = None, datasets=None,
                 targetNames: List[str] = ["Human"], targetKey: str = "Host",
                 transferCheckpoint: str = None,
                 namespace: str = "hep", seed: int = 1, dim: int = 512, batch_size: int = 500,
                 n_epochs: int = 11, train: bool = False, test: bool = False, embed: bool = False,
                 semantics: bool = False, combfit: bool = False, reinfection: bool = False, train_split: float = 0.8):
        self.args = types.SimpleNamespace()
        self.args.model_name = model_name
        self.args.seed = seed
        self.args.namespace = namespace
        self.args.dim = dim
        self.args.batch_size = batch_size
        self.args.checkpoint = checkpoint
        self.args.n_epochs = n_epochs
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

    def interpret_clusters(self, adata):
        clusters = sorted(set(adata.obs['louvain']))
        for cluster in clusters:
            tprint('Cluster {}'.format(cluster))
            adata_cluster = adata[adata.obs['louvain'] == cluster]
            for var in ['host', 'country', 'strain']:
                tprint('\t{}:'.format(var))
                counts = Counter(adata_cluster.obs[var])
                for val, count in counts.most_common():
                    tprint('\t\t{}: {}'.format(val, count))
            tprint('')

    def plot_umap(self, adata, categories, namespace='hep'):
        for category in categories:
            sc.pl.umap(adata, color=category,
                       save='_{}_{}.png'.format(namespace, category))

    def analyze_embedding(self, model, seqs, vocabulary):
        seqs = embed_seqs(self.args, model, seqs, vocabulary, use_cache=True)

        X, obs = [], {}
        obs['n_seq'] = []
        obs['seq'] = []
        for seq in seqs:
            meta = seqs[seq][0]
            X.append(meta['embedding'].mean(0))
            for key in meta:
                if key == 'embedding':
                    continue
                if key not in obs:
                    obs[key] = []
                obs[key].append(Counter([
                    meta[key] for meta in seqs[seq]
                ]).most_common(1)[0][0])
            obs['n_seq'].append(len(seqs[seq]))
            obs['seq'].append(str(seq))
        X = np.array(X)

        adata = AnnData(X)
        for key in obs:
            adata.obs[key] = obs[key]

        sc.pp.neighbors(adata, n_neighbors=20, use_rep='X')
        sc.tl.louvain(adata, resolution=1.)
        sc.tl.umap(adata, min_dist=1.)

        sc.set_figure_params(dpi_save=500)
        self.plot_umap(adata, ['host', 'group', 'continent', 'louvain'])

        self.interpret_clusters(adata)

        adata_cov2 = adata[(adata.obs['louvain'] == '0') |
                           (adata.obs['louvain'] == '2')]
        self.plot_umap(adata_cov2, ['host', 'group', 'country'],
                       namespace='cov7')

    def start(self):
        # Initialization
        vocabulary = {aa: idx + 1 for idx, aa in enumerate(sorted(AAs))}
        vocab_size = len(vocabulary) + 2  # For padding characters
        yVocab = {host: idx for idx, host in enumerate(sorted(self.args.targetNames), start=1)}
        yVocab['not-in-dictionary'] = 0

        # Load dataframe
        df = Ingestion.loadFastaFiles(self.args.datasets)
        seq_len = int(df.seq.map(len).max()) + 2
        df = Preprocessor.featurize_df(df, seq_len, vocabulary, yVocab, self.args.targetKey)

        # Load in for transfer learning
        if self.args.transferCheckpoint:
            model = load_model(self.args.transferCheckpoint)
        else:
            model = get_model(self.args, seq_len, vocab_size, inference_batch_size=self.args.batch_size).model_

        # Make host model
        hostModel = get_model_host(self.args, model, seq_len - 1, vocab_size, inference_batch_size=self.args.batch_size)

        # Regular checkpoint of weights
        if self.args.checkpoint:
            hostModel.model_.load_weights(self.args.checkpoint)
            hostModel.model_.summary()

        date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        if self.args.train:
            hostModel.model_.summary()
            train_df, test_df = Preprocessor.split_df(df, self.args.train_split)

            h = hostModel.fit(train_df['seq_processed'], train_df['target'], test_df['seq_processed'], test_df['target'], date, yVocab)

        if self.args.test:
            pass

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
