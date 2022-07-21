from TransferModel.Analysis import Visualization as Visualization
from mutation_host import featurize_seqs_host, featurize_hosts


def report_auroc_host_(model, X_test, y_test, lengths_test, labelVocab, filename=None, average="macro"):
    y_pred = model.predict(X_test, lengths_test)
    y_pred = y_pred.argmax(axis=-1)
    return Visualization.plot_auroc(y_test, y_pred, labelVocab, filename, average)


def report_auroc_host(model, vocab, labelVocab, test_seqs, filename=None, average="macro"):
    X_test, lengths_test = featurize_seqs_host(test_seqs, vocab)
    y_test = featurize_hosts(test_seqs, labelVocab)
    return report_auroc_host_(model, X_test, y_test, lengths_test, labelVocab, filename, average)