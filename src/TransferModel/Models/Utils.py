import os

import numpy as np
from anndata import AnnData

from TransferModel.Analysis import Evaluation
from TransferModel.Analysis import Visualization
from TransferModel.Models.BiLSTM_Model import BiLSTMTargetModel

import scanpy as sc
import pathlib


def err_model(name):
    raise ValueError('Model {} not supported'.format(name))


def get_target_model(args, parentModel, seq_len, vocab_size, target_size):
    if 'bilstm' in args.model_name:
        model = BiLSTMTargetModel(
            seq_len,
            parentModel,
            vocab_size,
            model_name=args.model_name,
            target_size=target_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.dim,
            n_hidden=args.n_hidden,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            inference_batch_size=args.inf_batch_size,
            figDir=args.figDir,
            cache_dir=args.targetDir,
            seed=args.seed,
            verbose=True,
        )
    else:
        err_model(args.model_name)

    return model


def fit_model(model, train_df, test_df, yVocab, date):
    model.model_.summary()

    h = model.fit(train_df['X'], train_df['y'], test_df['X'],
                  test_df['y'], date, yVocab)

    return h


def print_per_class(metrics, yVocabInverse, metricName):
    print(f"{metricName} per class: ", {yVocabInverse[i]: m for i, m in enumerate(metrics)})


def test_model(args, model, test_df, yVocab, date):
    y_pred = model.predict(test_df['X'], sparse=True)
    testAurocs = Evaluation.report_auroc(test_df['y'], y_pred, labelVocab=yVocab,
                                         filename=f"{args.figDir}/{args.model_name}_AUROC_{date}")


    yVocabInverse = {y: x for x, y in yVocab.items()}
    print_per_class(testAurocs, yVocabInverse, "AUROC")

    testAuroc = Evaluation.report_auroc(test_df['y'], y_pred, labelVocab=yVocab, average='macro')
    print(f"Macro auroc: {testAuroc}")
    testAuroc = Evaluation.report_auroc(test_df['y'], y_pred, labelVocab=yVocab, average='micro')
    print(f"Micro auroc: {testAuroc}")


    res, matrix = Evaluation.report_accuracy_per_class(test_df['y'], y_pred, yDict=yVocab)
    print_per_class(res, yVocabInverse, "Accuracy")
    modelFreq, _ = Evaluation.report_class_distribution(None, None, None, 0, matrix)
    print_per_class(modelFreq, yVocabInverse, "Frequency of prediction by model")
    dfFreq, _ = Evaluation.report_class_distribution(None, None, None, 1, matrix)
    print_per_class(dfFreq, yVocabInverse, "Frequency by dataset")
    print("Overall accuracy: ", Evaluation.report_accuracy(test_df['y'], y_pred))


def analyze_embedding(args, model, test_df):
    embeddings = embed_sequences(args, test_df["X"], model, args.embedding_cache)

    adata = AnnData(embeddings, dtype=np.float16)

    for c in args.embedTargets:
        adata.obs[c] = test_df[c].to_numpy()

    sc.pp.neighbors(adata, n_neighbors=200, use_rep='X')
    sc.tl.louvain(adata, resolution=1.)
    Visualization.plot_umap(args, adata, args.figDir)
    Visualization.plot_umap3d(args, adata, args.figDir)
    Evaluation.report_cluster_purity(adata)


def embed_sequences(args, X, model, useCache=False):
    embed_fname = ('{}/embedding/{}_{}.npy'
                   .format(args.targetDir, args.model_name, args.dim))
    if useCache:
        embed_dir = ('{}/embedding'.format(args.targetDir))
        pathlib.Path(embed_dir).mkdir(parents=True, exist_ok=True)
        if os.path.exists(embed_fname):
            return np.load(embed_fname, allow_pickle=True)

    if not (useCache and os.path.exists(embed_fname)):
        embeddings = model.transform(X)
        if useCache:
            np.save(embed_fname, embeddings)
        return embeddings



