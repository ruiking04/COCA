from torch import nn
import torch


class base_Model(nn.Module):
    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.features_len = configs.features_len
        self.project_channels = configs.project_channels
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
            nn.Conv1d(32, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
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
        self.project = nn.Linear(self.final_out_channels * self.features_len, self.project_channels, bias=False)
        self.projection_head = nn.Sequential(
            nn.Linear(self.final_out_channels * self.features_len, self.final_out_channels * self.features_len // 2),
            nn.BatchNorm1d(self.final_out_channels * self.features_len // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_out_channels * self.features_len // 2, self.project_channels),
        )

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
        # Encoder
        hidden = x.permute(0, 2, 1)
        _, enc_hidden = self.encoder(hidden)
        # Decoder
        dec_hidden = enc_hidden
        output = torch.zeros(hidden.shape).to(self.device)
        for i in reversed(range(hidden.shape[1])):
            output[:, i, :] = self.output_layer(dec_hidden[0][0, :])
            if self.training:
                _, dec_hidden = self.decoder(hidden[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)
        hidden = hidden.reshape(hidden.size(0), -1)
        output = output.reshape(output.size(0), -1)
        project = self.projection_head(hidden)
        rec_project = self.projection_head(output)

        return project, rec_project

