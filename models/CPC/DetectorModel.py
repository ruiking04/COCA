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

import torch.optim as optim
from utils import ScheduledOptim
from .network.model.model import CDCK2
from .network.validation_v1 import validation
from .network.training_v1 import train, snapshot
from .DetectorConfig import CPCConf


class CPC(DetectorBase):
    """
    The CPC-based multivariate time series anomaly detector.
    """

    config_class = CPCConf

    def __init__(self, config: CPCConf):
        super().__init__(config)
        self.config = config
        self.num_epochs = config.epochs
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def _build_model(self, dim):
        return CDCK2(self.config.timestep, self.config.batch_size, self.config.sequence_length, dim)

    def _train(self, X):
        """
        :param X: The input time series, a numpy array.
        """
        use_cuda = torch.cuda.is_available()
        self.data_dim = X.shape[1]
        self.model = self._build_model(self.data_dim).to(self.device)

        ## Loading the dataset
        params = {
            'num_workers': 0,
            'pin_memory': False} if use_cuda else {}

        training_set = InputData(X, self.config.sequence_length)
        train_loader = DataLoader(training_set, batch_size=self.config.batch_size, shuffle=True,
                                  **params)  # set shuffle to True
        validation_set = InputData(X, self.config.sequence_length)
        validation_loader = DataLoader(validation_set, batch_size=self.config.batch_size, shuffle=False, **params)

        # nanxin optimizer
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4, amsgrad=True),
            self.config.n_warmup_steps)

        # Start training
        for epoch in range(1, self.config.epochs + 1):

            # Train and validate
            train(self.config, self.model, self.device, train_loader, optimizer, epoch, self.config.batch_size)

            # Save
            if epoch % 5 == 0:
                snapshot(self.config.logging_dir, 'cpc', {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                })
            if epoch % 10 == 0:
                optimizer.increase_delta()

    def _detect(self, X):
        """
        :param X: The input time series, a numpy array.
        """
        self.model.eval()
        use_cuda = torch.cuda.is_available()
        data_loader = DataLoader(
            dataset=InputData(X, k=self.sequence_length), batch_size=self.batch_size, shuffle=False
        )
        scores = []
        for data in data_loader:
            inputs = data.float().transpose(1, 2).to(self.device)
            hidden = self.model.init_hidden(len(inputs), use_gpu=use_cuda)
            _, _, _, nce = self.model(inputs, hidden, return_nce=True)
            scores.append(nce)
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, X.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, i: i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)
        return scores

    def _get_sequence_len(self):
        return self.sequence_length

    def train(
            self, train_data: TimeSeries, anomaly_labels: TimeSeries = None, train_config=None,
            post_rule_train_config=None
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
        return TimeSeries({"anom_score": UnivariateTimeSeries(timestamps, scores[-len(timestamps):])})


class LSTMEDModule(nn.Module):
    """
    The LSTM-encoder-decoder module. Both the encoder and decoder are LSTMs.

    :meta private:
    """

    def __init__(self, n_features, hidden_size, n_layers, dropout, device):
        """
        :param n_features: The input feature dimension
        :param hidden_size: The LSTM hidden size
        :param n_layers: The number of LSTM layers
        :param dropout: The dropout rate
        :param device: CUDA or CPU
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.encoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[0],
            bias=True,
            dropout=self.dropout[0],
        )
        self.decoder = nn.LSTM(
            self.n_features,
            self.hidden_size,
            batch_first=True,
            num_layers=self.n_layers[1],
            bias=True,
            dropout=self.dropout[1],
        )
        self.output_layer = nn.Linear(self.hidden_size, self.n_features)

    def init_hidden_state(self, batch_size):
        h = torch.zeros((self.n_layers[0], batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.n_layers[0], batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x, return_latent=False):
        # Encoder
        enc_hidden = self.init_hidden_state(x.shape[0])
        _, enc_hidden = self.encoder(x.float(), enc_hidden)
        # Decoder
        dec_hidden = enc_hidden
        output = torch.zeros(x.shape).to(self.device)
        for i in reversed(range(x.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(x[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        return (output, enc_hidden[1][-1]) if return_latent else output
