from keras.callbacks import Callback

from TransferModel.Analysis import Visualization


# Adapted from https://medium.com/@kapilvarshney/how-to-plot-the-model-training-in-keras-using-custom-callback-function-and-using-tensorboard-41e4ce3cb401
class TrainingPlot(Callback):

    def __init__(self, date):
        self.date = date
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            Visualization.plot_metric(self.losses, self.val_losses, f"hep_bilstm_host_CE_{self.date}",
                                      metricName="Cross entropy", optimalFlag="min")
            Visualization.plot_metric(self.acc, self.val_acc, f"hep_bilstm_host_ACC_{self.date}",
                                      metricName="Accuracy", optimalFlag="max")
