from torch import nn
import torch


class base_Model(nn.Module):
    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.hidden_size = configs.hidden_size
        self.window_size = configs.window_size
        self.device = device
        self.num_layers = configs.num_layers
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.encoder = nn.LSTM(
            self.final_out_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout,
        )
        self.decoder = nn.LSTM(
            self.final_out_channels,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout,
        )

        self.output_layer = nn.Linear(self.hidden_size, self.final_out_channels)

    def init_hidden_state(self, batch_size):
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x_in):
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        # 1D CNN feature extraction
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # Encoder
        x = x.permute(0, 2, 1)
        enc_hidden = self.init_hidden_state(x.shape[0])
        _, enc_hidden = self.encoder(x, enc_hidden)
        # Decoder
        dec_hidden = enc_hidden
        output = torch.zeros(x.shape).to(self.device)
        # feature_dec = torch.zeros(feature.shape).to(self.device)
        for i in reversed(range(x.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(x[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        x = x.reshape(-1, self.final_out_channels)
        output = output.reshape(-1, self.final_out_channels)
        return x, output

