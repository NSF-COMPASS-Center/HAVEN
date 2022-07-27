from collections import defaultdict

from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

from TransferModel.Analysis import Visualization
from TransferModel.DataUtils import DataProcessor


class AUROCCallback(Callback):
    def __init__(self, x, y, valX, valY, batch_size, date, yVocab, average=None, multi_class="ovo"):
        super().__init__()
        self.x = x
        self.x_val = valX
        self.batch_size = batch_size
        self.date = date
        self.history = defaultdict(list)
        self.yVocab = yVocab
        self.y = DataProcessor.toOneHot(y, yVocab)
        self.y_val = DataProcessor.toOneHot(valY, yVocab)
        self.average = average
        self.multi_class = multi_class

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict(self.x, batch_size=self.batch_size)
        y_pred_val = self.model.predict(self.x_val, batch_size=self.batch_size)

        roc_train = roc_auc_score(self.y, y_pred_train, multi_class=self.multi_class, average=self.average)
        roc_val = roc_auc_score(self.y_val, y_pred_val, multi_class=self.multi_class, average=self.average)

        # If average is set, then each class is returned; otherwise, it's just one value (see roc_auc_score docs)
        if not self.average:
            for k, v in self.yVocab.items():
                self.history[f'train_{k}'].append(roc_train[v])
                self.history[f'test_{k}'].append(roc_val[v])
        else:
            self.history[f'train'].append(roc_train)
            self.history[f'test'].append(roc_val)

        if len(self.history[list(self.history.keys())[0]]) > 1:
            Visualization.plot_metrics(self.history, f"hep_bilstm_host_AUROC_{self.date}",
                                       metricName=f'AUROC (avg={self.average}, multi_class={self.multi_class})')
