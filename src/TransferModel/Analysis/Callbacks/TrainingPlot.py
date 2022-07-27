from collections import defaultdict

from keras.callbacks import Callback

from TransferModel.Analysis import Visualization


# Adapted from https://medium.com/@kapilvarshney/how-to-plot-the-model-training-in-keras-using-custom-callback-function-and-using-tensorboard-41e4ce3cb401
class TrainingPlot(Callback):

    def __init__(self, date):
        super().__init__()
        self.date = date
        self.logs = defaultdict(list)
        self.accLogs = defaultdict(list)
        self.lossLogs = defaultdict(list)

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.accLogs["Training Accuracy"].append(logs.get('accuracy'))
        self.accLogs["Validation Accuracy"].append(logs.get('val_accuracy'))
        self.lossLogs["Training Loss"].append(logs.get('loss'))
        self.lossLogs["Validation Loss"].append(logs.get('val_loss'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.lossLogs['Training Loss']) > 1:
            Visualization.plot_metrics(self.accLogs, f"hep_bilstm_host_ACC_{self.date}", metricName="Accuracy")
            Visualization.plot_metrics(self.lossLogs, f"hep_bilstm_host_CE_{self.date}", metricName="Cross entropy")
