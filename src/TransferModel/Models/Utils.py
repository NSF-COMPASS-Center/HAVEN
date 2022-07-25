import math
import os
import random

from LanguageModel.utils import tprint
from TransferModel.Analysis.Evaluation import report_performance_host
from TransferModel.DataUtils.Preprocessor import featurize_seqs_host


def batch_train_host(args, model, train_df, val_df, vocabulary, labelVocab, batch_size=500,
                     verbose=True):
    assert args.train

    # Control epochs here.
    n_epochs = args.n_epochs
    args.n_epochs = 1
    model.n_epochs_ = 1

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    train_auc = []
    test_auc = []

    n_batches = math.ceil(len(train_df) / float(batch_size))
    if verbose:
        tprint('Training seq batch size: {}, N batches: {}'
               .format(batch_size, n_batches))

    for epoch in range(n_epochs):
        if verbose:
            tprint('True epoch {}/{}'.format(epoch + 1, n_epochs))

        # Shuffle training sequences in each epoch
        perm_seqs = [str(s) for s in train_df.keys()]
        random.shuffle(perm_seqs)

        for batchi in range(n_batches):
            start = batchi * batch_size
            end = (batchi + 1) * batch_size
            seqs_batch = {seq: train_df[seq] for seq in perm_seqs[start:end]}
            model = fit_model_host(model, seqs_batch, vocabulary, labelVocab)
            del seqs_batch

        if args.test and val_df:
            trainCE, testCE, trainAcc, testAcc, trainAUROC, testAUROC = report_performance_host(args.model_name, model,
                                                                                                vocabulary, labelVocab,
                                                                                                train_df, val_df)
            train_loss.append(trainCE)
            test_loss.append(testCE)
            train_acc.append(trainAcc)
            test_acc.append(testAcc)
            train_auc.append(trainAUROC)
            test_auc.append(testAUROC)

        fname_prefix = ('target/{0}/checkpoints/{1}/{1}_{2}'
                        .format(args.namespace, model.model_name_, args.dim))

        if epoch == 0:
            os.rename('{}-01.hdf5'.format(fname_prefix),
                      '{}-00.hdf5'.format(fname_prefix))
        else:
            os.rename('{}-01.hdf5'.format(fname_prefix),
                      '{}-{:02d}.hdf5'.format(fname_prefix, epoch + 1))
        model.gpu_gc()

    os.rename('{}-00.hdf5'.format(fname_prefix),
              '{}-01.hdf5'.format(fname_prefix))

    return train_loss, test_loss, train_acc, test_acc, train_auc, test_auc


def err_model(name):
    raise ValueError('Model {} not supported'.format(name))


def get_model_host(args, parentModel, seq_len, vocab_size,
                   inference_batch_size=200):
    if args.model_name == 'bilstm':
        from TransferModel.Models.BiLSTM_Model import BiLSTMHostModel
        model = BiLSTMHostModel(
            seq_len,
            parentModel,
            vocab_size,
            target_size=len(args.targetNames),
            embedding_dim=20,
            hidden_dim=args.dim,
            n_hidden=2,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            inference_batch_size=inference_batch_size,
            cache_dir='target/{}'.format(args.namespace),
            seed=args.seed,
            verbose=True,
        )
    else:
        err_model(args.model_name)

    return model


def fit_model_host(model, seqs, vocabulary, labelVocab):
    X, lengths = featurize_seqs_host(seqs, vocabulary)
    y = seqs['target']
    model.fit(X, lengths, y)
    return model