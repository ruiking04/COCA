import numpy as np
import torch
import torch.nn as nn
from typing import Sequence
from torch.utils.data import DataLoader
from merlion.utils import UnivariateTimeSeries, TimeSeries
from merlion.models.base import NormalizingConfig
from merlion.models.anomaly.base import DetectorBase, DetectorConfig
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.misc import ProgressBar, initializer
from merlion.models.anomaly.utils import InputData, batch_detect

from .DetectorConfig import OCSVMConf
from sklearn.svm import OneClassSVM

class OCSVM(DetectorBase):
    """
    The LSTM-encoder-decoder-based multivariate time series anomaly detector.
    The time series representation is modeled by an encoder-decoder network where
    both encoder and decoder are LSTMs. The distribution of the reconstruction error
    is estimated for anomaly detection.
    """

    config_class = OCSVMConf

    def __init__(self, config: OCSVMConf):
        super().__init__(config)
        self.kernel = config.kernel
        self.nu = config.nu
        self.degree = config.degree
        self.gamma = config.gamma
        self.sequence_length = config.sequence_len
        self.clf = None

    def _build_model(self):
        clf = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            degree=self.degree,
            gamma=self.gamma
        )
        return clf

    def _train(self, X):
        """
        :param X: The input time series, a numpy array.
        """
        self.clf = self._build_model()
        self.clf.fit(X)

    def _detect(self, X):
        """
        :param X: The input time series, a numpy array.
        """
        return self.clf.decision_function(X)

    def _get_sequence_len(self):
        return self.sequence_length

    def train(
        self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None, post_rule_train_config=None
    ) -> TimeSeries:
        """
        Train a multivariate time series anomaly detector.

        :param train_data: A `TimeSeries` of metric values to train the model.
        :param anomaly_labels: A `TimeSeries` indicating which timestamps are
            anomalous. Optional.
        :param train_config: Additional training configs, if needed. Only
            required for some models.
        :param post_rule_train_config: The config to use for training the
            model's post-rule. The model's default post-rule train config is
            used if none is supplied here.

        :return: A `TimeSeries` of the model's anomaly scores on the training
            data.
        """
        train_data = self.train_pre_process(train_data, require_even_sampling=False, require_univariate=False)

        train_df = train_data.align().to_pd()
        self._train(train_df.values)
        scores = batch_detect(self, train_df.values)

        train_scores = TimeSeries({"anom_score": UnivariateTimeSeries(train_data.time_stamps, scores)})
        self.train_post_rule(
            anomaly_scores=train_scores, anomaly_labels=anomaly_labels, post_rule_train_config=post_rule_train_config
        )
        return train_scores

    def get_anomaly_score(self, time_series: TimeSeries, time_series_prev: TimeSeries = None) -> TimeSeries:
        """
        :param time_series: The `TimeSeries` we wish to predict anomaly scores for.
        :param time_series_prev: A `TimeSeries` immediately preceding ``time_series``.
        :return: A univariate `TimeSeries` of anomaly scores
        """
        time_series, time_series_prev = self.transform_time_series(time_series, time_series_prev)
        ts = time_series_prev + time_series if time_series_prev is not None else time_series
        scores = batch_detect(self, ts.align().to_pd().values)
        timestamps = time_series.time_stamps
        return TimeSeries({"anom_score": UnivariateTimeSeries(timestamps, scores[-len(timestamps) :])})