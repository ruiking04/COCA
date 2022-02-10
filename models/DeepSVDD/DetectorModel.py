#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The LSTM-encoder-decoder-based anomaly detector for multivariate time series
"""
import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from merlion.utils import UnivariateTimeSeries, TimeSeries
from merlion.models.anomaly.base import DetectorBase
from merlion.models.anomaly.utils import InputData, batch_detect

from models.DeepSVDD import DeepSVDDConf
from .Network import DeepSVDDModule


class DeepSVDD(DetectorBase):
    """
    The LSTM-encoder-decoder-based multivariate time series anomaly detector.
    The time series representation is modeled by an encoder-decoder network where
    both encoder and decoder are LSTMs. The distribution of the reconstruction error
    is estimated for anomaly detection.
    """

    config_class = DeepSVDDConf

    def __init__(self, config: DeepSVDDConf):
        super().__init__(config)
        self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def _build_model(self):
        return DeepSVDDModule(self.config, self.config.objective, self.config.nu)

    def _train(self, X):
        """
        :param X: The input time series, a numpy array.
        """
        train_x = InputData(X, k=self.config.sequence_length)
        self.config.input_channels = X.shape[1]
        self.config.final_out_channels = X.shape[1]
        self.model = self._build_model()

        if not os.path.exists(self.config.xp_path):
            os.mkdir(self.config.xp_path)

        # Set seed
        if self.config.seed != -1:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

        # Initialize DeepSVDD model and set neural network \phi
        self.model.set_network(self.config.net_name)
        # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
        if self.config.load_model and os.path.exists(self.config.load_model):
            self.model.load_model(model_path=self.config.load_model, load_ae=True)

        if self.config.pretrain:
            # Pretrain model on dataset (via autoencoder)
            self.model.pretrain(train_x,
                               optimizer_name=self.config.ae_optimizer_name,
                               lr=self.config.ae_lr,
                               n_epochs=self.config.ae_n_epochs,
                               lr_milestones=self.config.ae_lr_milestone,
                               batch_size=self.config.ae_batch_size,
                               weight_decay=self.config.ae_weight_decay,
                               device=self.config.device,
                               n_jobs_dataloader=self.config.n_jobs_dataloader)

        # Train model on dataset
        self.model.train(train_x,
                        optimizer_name=self.config.optimizer_name,
                        lr=self.config.lr,
                        n_epochs=self.config.n_epochs,
                        lr_milestones=self.config.lr_milestone,
                        batch_size=self.config.batch_size,
                        weight_decay=self.config.weight_decay,
                        device=self.config.device,
                        n_jobs_dataloader=self.config.n_jobs_dataloader)

    def _detect(self, X):
        """
        :param X: The input time series, a numpy array.
        """
        # data_loader = DataLoader(
        #     dataset=InputData(X, k=self.sequence_length), batch_size=self.batch_size, shuffle=False
        # )
        self.model.net.eval()
        scores = self.model.test(InputData(X, k=self.config.sequence_length), device=self.config.device)
        scores = np.concatenate(scores)
        lattice = np.full((self.config.sequence_length, X.shape[0]), np.nan)
        for i, score in enumerate(scores):
            score = score.reshape(-1, self.config.sequence_length)
            score = np.random.rand(*score.shape)
            lattice[i % self.config.sequence_length, i: i + self.config.sequence_length] = score
        scores = np.nanmax(lattice, axis=0)
        return scores

    def _get_sequence_len(self):
        return self.config.sequence_length

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