import numpy as np
import torch
import os
from merlion.evaluate.anomaly import ScoreType

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, idx, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path : save model path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.idx = idx
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_affiliation = None
        self.best_rpa_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, score, affiliation, rpa_score, model):

        if self.best_score is None:
            self.best_score = score
            self.best_affiliation = affiliation
            self.best_rpa_score = rpa_score
            self.save_checkpoint(score, model)
        elif score.correct_num < self.best_score.correct_num + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_affiliation = affiliation
            self.best_rpa_score = rpa_score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when score decrease.'''
        if self.verbose:
            print(f'score decreased ({self.best_score.correct_num:.6f} --> {score.correct_num:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, str(self.idx).zfill(2) + '_best_network.pth')
        torch.save(model.state_dict(), path)
