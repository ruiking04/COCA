import torch
import torch.nn as nn
from merlion.models.anomaly.autoencoder import MLP, AutoEncoder
from models.DeepSVDD.base import BaseNet

class Merlion_MLP(BaseNet):
    def __init__(self, config):
        super(Merlion_MLP, self).__init__()
        self.rep_dim = config.hidden_size
        self.layers = torch.nn.Sequential(*getNN(
            input_size=config.input_size,
            output_size=config.hidden_size,
            layer_sizes=config.layer_sizes,
            activation=config.activation,
            dropout_prob=config.dropout_prob
        ))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.layers(x)
    
class Merlion_MLP_Autoencoder(BaseNet):
    def __init__(self, config):
        super(Merlion_MLP_Autoencoder, self).__init__()
        self.rep_dim = config.input_size
        encoder = getNN(
            input_size=config.input_size,
            output_size=config.hidden_size,
            layer_sizes=config.layer_sizes,
            activation=config.activation,
            dropout_prob=config.dropout_prob
        )
        decoder = getNN(
            input_size=config.hidden_size,
            output_size=config.input_size,
            layer_sizes=config.layer_sizes[::-1],
            activation=nn.Identity,
            dropout_prob=config.dropout_prob
        )
        self.layers = torch.nn.Sequential(*(encoder + decoder))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        y = self.layers(x)
        return torch.norm(x - y, dim=1)

class LSTMEncoder(nn.Module):
    def __init__(self, configs, device):
        super(LSTMEncoder, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.hidden_size = configs.hidden_size
        self.device = device
        self.num_layers = configs.n_layers
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=self.kernel_size, stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, self.final_out_channels, kernel_size=self.kernel_size, stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.encoder = nn.LSTM(
            self.final_out_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=True,
            dropout=self.dropout,
        )

    def init_hidden_state(self, batch_size):
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)

        # 1D CNN feature extraction
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # Encoder
        x = x.permute(0, 2, 1)

        hidden = torch.zeros((x.shape[0], x.shape[1], self.hidden_size), dtype=torch.double).to(self.device)
        enc_hidden = None
        for i in reversed(range(x.shape[1])):
            _, enc_hidden = self.encoder(x[:, i].unsqueeze(1), enc_hidden)
            hidden[:, i, :] = enc_hidden[0][-1]
        return hidden.reshape((-1, self.hidden_size))

class LSTMAutoEncoder(nn.Module):
    def __init__(self, configs, device):
        super(LSTMAutoEncoder, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.hidden_size = configs.hidden_size
        self.device = device
        self.num_layers = configs.n_layers
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=self.kernel_size, stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, self.final_out_channels, kernel_size=self.kernel_size, stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.encoder = nn.LSTM(
            self.final_out_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=True,
            dropout=self.dropout,
        )
        self.decoder = nn.LSTM(
            self.final_out_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=True,
            dropout=self.dropout,
        )

        self.output_layer = nn.Linear(self.hidden_size, self.final_out_channels)

    def init_hidden_state(self, batch_size):
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)

        # 1D CNN feature extraction
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # Encoder
        x = x.permute(0, 2, 1)
        _, enc_hidden = self.encoder(x)
        # Decoder
        dec_hidden = enc_hidden
        output = torch.zeros(x.shape).to(self.device)

        # feature_dec = torch.zeros(feature.shape).to(self.device)
        for i in reversed(range(x.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(x[:, i].unsqueeze(1), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        return output

def getNN(input_size, output_size, layer_sizes, activation=nn.ReLU, dropout_prob=0.0):
    layers, layer_sizes = [], [input_size] + list(layer_sizes)
    for i in range(1, len(layer_sizes)):
        layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        layers.append(activation())
        layers.append(nn.Dropout(p=dropout_prob))
    layers.append(nn.Linear(layer_sizes[-1], output_size))
    layers.append(activation())
    return layers