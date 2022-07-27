from collections import defaultdict

from keras.callbacks import Callback

from TransferModel.Analysis import Visualization


# Adapted from https://medium.com/@kapilvarshney/how-to-plot-the-model-training-in-keras-using-custom-callback-function-and-using-tensorboard-41e4ce3cb401
class TrainingPlot(Callback):

    def __init__(self, date):
        super().__init__()
        self.date = date
        self.logs = defaultdict(list)

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs["Training Loss"].append(logs.get('loss'))
        self.logs["Training Accuracy"].append(logs.get('accuracy'))
        self.logs["Validation Loss"].append(logs.get('val_loss'))
        self.logs["Validation Accuracy"].append(logs.get('val_accuracy'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            Visualization.plot_metrics(self.log, f"hep_bilstm_host_training_{self.date}", metricName="Accuracy and "
                                                                                                     "Cross Entropy "
                                                                                                     "Loss")
