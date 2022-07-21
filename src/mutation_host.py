from LanguageModel.utils import *
from LanguageModel.mutation import err_model
from TransferModel.Analysis.Evaluation import report_auroc_host_


def get_model_host(args, parentModel, seq_len, vocab_size,
                   inference_batch_size=200):
    if args.model_name == 'bilstm':
        from TransferModel.Models.host_model import BiLSTMHostModel
        model = BiLSTMHostModel(
            seq_len,
            parentModel,
            vocab_size,
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


def featurize_seqs_host(seqs, vocabulary):
    # First two in vocabulary are paddings
    start_int = len(vocabulary) + 1

    # TODO: extensive comment
    # tldr since their model is 1 less, we chop off end_int.
    end_int = len(vocabulary) + 2

    sorted_seqs = sorted(seqs.keys())

    X = np.concatenate([
        np.array([start_int] + [
            vocabulary[word] for word in seq
        ]) for seq in sorted_seqs
    ]).reshape(-1, 1)

    # Check that length of each word is valid (all have + 2 padding and same length as original)
    lens = np.array([len(seq) + 1 for seq in sorted_seqs])
    assert (sum(lens) == X.shape[0])

    return X, lens


def featurize_hosts(seqs, vocabulary):
    assert (vocabulary)
    sorted_seqs = sorted(seqs.keys())
    Y = []
    for key in sorted_seqs:
        # Check if key is in
        if seqs[key][0]['host'] in vocabulary:
            Y.append(vocabulary[seqs[key][0]['host']])
        else:
            Y.append(0)
    Y = np.array(Y, dtype=int)
    return Y


def fit_model_host(model, seqs, vocabulary, labelVocab):
    X, lengths = featurize_seqs_host(seqs, vocabulary)
    y = featurize_hosts(seqs, labelVocab)
    model.fit(X, lengths, y)
    return model


def cross_entropy(logprob, n_samples):
    return -logprob / n_samples


def report_performance_host(model_name, model, vocabulary, labelVocab, train_seqs, test_seqs):
    # Expects featurized X, y and lengths, returns
    X_train, lengths_train = featurize_seqs_host(train_seqs, vocabulary)
    y_train = featurize_hosts(train_seqs, labelVocab)
    logprob, trainAcc = model.score(X_train, lengths_train, y_train)
    trainCE = cross_entropy(logprob, len(lengths_train))
    tprint('Model {}, train cross entropy: {}'
           .format(model_name, trainCE))

    trainAUROC = report_auroc_host_(model, X_train, y_train, lengths_train, labelVocab)

    tprint('Model {}, train AUROC: {}'
           .format(model_name, trainAUROC))

    X_test, lengths_test = featurize_seqs_host(test_seqs, vocabulary)
    y_test = featurize_hosts(test_seqs, labelVocab)
    logprob, testAcc = model.score(X_test, lengths_test, y_test)
    testCE = cross_entropy(logprob, len(lengths_test))
    tprint('Model {}, test cross entropy: {}'
           .format(model_name, testCE))

    testAUROC = report_auroc_host_(model, X_test, y_test, lengths_test, labelVocab)

    tprint('Model {}, test AUROC: {}'
           .format(model_name, testAUROC))

    return (trainCE, testCE, trainAcc, testAcc, trainAUROC, testAUROC)


def batch_train_host(args, model, train_seqs, val_seqs, vocabulary, labelVocab, batch_size=500,
                     verbose=True):
    assert (args.train)

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

    n_batches = math.ceil(len(train_seqs) / float(batch_size))
    if verbose:
        tprint('Training seq batch size: {}, N batches: {}'
               .format(batch_size, n_batches))

    for epoch in range(n_epochs):
        if verbose:
            tprint('True epoch {}/{}'.format(epoch + 1, n_epochs))

            # Shuffle training sequences in each epoch
        perm_seqs = [str(s) for s in train_seqs.keys()]
        random.shuffle(perm_seqs)

        for batchi in range(n_batches):
            start = batchi * batch_size
            end = (batchi + 1) * batch_size
            seqs_batch = {seq: train_seqs[seq] for seq in perm_seqs[start:end]}
            model = fit_model_host(model, seqs_batch, vocabulary, labelVocab)
            del seqs_batch

        if args.test and val_seqs:
            trainCE, testCE, trainAcc, testAcc, trainAUROC, testAUROC = report_performance_host(args.model_name, model,
                                                                                                vocabulary, labelVocab,
                                                                                                train_seqs, val_seqs)
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
