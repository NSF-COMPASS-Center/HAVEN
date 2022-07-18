#!/usr/bin/env python
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DataUtils:
    @staticmethod
    def split_seqs(seqs, percentTrain=.8, seed=1):
        x_train, x_test, y_train, y_test = train_test_split(list(seqs.keys()), list(seqs.values()), train_size=percentTrain, random_state=seed)

        train_seqs = {k:v for k,v in zip(x_train, y_train)}
        test_seqs = {k:v for k,v in zip(x_test, y_test)}

        return train_seqs, test_seqs



    @staticmethod
    def plot_metric(trainCE, testCE, filename, metricName="Cross Entropy"):
        with plt.style.context("seaborn"):
            fig = plt.figure(1, [16, 9])
            epochs = range(1,len(trainCE)+1)
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            if trainCE: 
                plt.plot(epochs, trainCE, label=f"Training {metricName}")
            if testCE:
                plt.plot(epochs, testCE, label=f"Testing {metricName}")
            

        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel(f"{metricName}", fontsize=20)
        plt.title('Training history',fontsize=20)
        plt.legend()

        # Print image
        name = f'figures/{filename}.png'
        plt.savefig(name, bbox_inches='tight', dpi=300)
        plt.clf()



