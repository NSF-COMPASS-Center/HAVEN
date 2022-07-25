from collections import defaultdict

from keras.callbacks import Callback
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

from TransferModel.Analysis import Visualization
from TransferModel.DataUtils import Preprocessor


class AUROCCallback(Callback):
    def __init__(self, x, y, valX, valY, batch_size, date, yVocab):
        self.x = x
        self.x_val = valX
        self.batch_size = batch_size
        self.date = date
        self.history = defaultdict(list)
        self.yVocab = yVocab
        self.y = Preprocessor.toOneHot(y, yVocab)
        self.y_val = Preprocessor.toOneHot(valY, yVocab)


    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict(self.x, batch_size=self.batch_size)
        roc_train = roc_auc_score(self.y, y_pred_train, multi_class="ovo", average=None)

        for k, v in self.yVocab.items():
            self.history[f'train_{k}'].append(roc_train[v])

        y_pred_val = self.model.predict(self.x_val, batch_size=self.batch_size)
        roc_val = roc_auc_score(self.y_val, y_pred_val, multi_class="ovo", average=None)

        for k, v in self.yVocab.items():
            self.history[f'test_{k}'].append(roc_val[v])

        if len(self.history[list(self.history.keys())[0]]) > 1:
            print("Test")
            Visualization.plot_metrics(self.history, f"hep_bilstm_host_AUROC_{self.date}", metricName='AUROC')

        return
