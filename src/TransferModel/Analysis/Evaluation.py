from LanguageModel.utils import tprint
from TransferModel.Analysis import Visualization as Visualization
from TransferModel.DataUtils.Preprocessor import featurize_seqs_host
from sklearn.metrics import roc_curve, auc, roc_auc_score
import tensorflow as tf



def auroc_metric(y_true, y_pred):
    return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)

def report_auroc_host_(model, X_test, y_test, labelVocab, filename=None, average="macro"):
    y_pred = model.predict(X_test)
    y_pred = y_pred.argmax(axis=-1)
    return Visualization.plot_auroc(y_test, y_pred, labelVocab, filename, average)


def report_auroc_host(model, vocab, labelVocab, test_seqs, filename=None, average="macro"):
    X_test = test_seqs['seq_processed']
    y_test = test_seqs['target']
    return report_auroc_host_(model, X_test, y_test, labelVocab, filename, average)


def report_performance_host(model_name, model, vocabulary, labelVocab, train_seqs, test_seqs):
    # Expects featurized X, y and lengths, returns
    X_train, lengths_train = featurize_seqs_host(train_seqs, vocabulary)
    y_train = train_seqs['target']
    logprob, trainAcc = model.score(X_train, lengths_train, y_train)
    trainCE = cross_entropy(logprob, len(lengths_train))
    tprint('Model {}, train cross entropy: {}'
           .format(model_name, trainCE))

    trainAUROC = report_auroc_host_(model, X_train, y_train, lengths_train, labelVocab)

    tprint('Model {}, train AUROC: {}'
           .format(model_name, trainAUROC))

    X_test, lengths_test = featurize_seqs_host(test_seqs, vocabulary)
    y_test = test_seqs['target']
    logprob, testAcc = model.score(X_test, lengths_test, y_test)
    testCE = cross_entropy(logprob, len(lengths_test))
    tprint('Model {}, test cross entropy: {}'
           .format(model_name, testCE))

    testAUROC = report_auroc_host_(model, X_test, y_test, lengths_test, labelVocab)

    tprint('Model {}, test AUROC: {}'
           .format(model_name, testAUROC))

    return (trainCE, testCE, trainAcc, testAcc, trainAUROC, testAUROC)


def cross_entropy(logprob, n_samples):
    return -logprob / n_samples