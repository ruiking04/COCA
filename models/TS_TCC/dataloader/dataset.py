import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform
from sklearn.model_selection import train_test_split
from utils import subsequences
from merlion.transform.normalize import MeanVarNormalize


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, valid_loader, test_loader


def data_generator1(time_series, time_series_label, configs, training_mode):
    time_series = time_series.to_numpy()
    time_series_label = time_series_label.to_numpy()
    time_series = subsequences(time_series, configs.window_size, configs.time_step)
    time_series_label = subsequences(time_series_label, configs.window_size, configs.time_step)
    # x_train, x_test, y_train, y_test = train_test_split(time_series, time_series_label, test_size=0.2, shuffle=False)
    x_train, x_test, y_train, y_test = train_test_split(time_series, time_series_label, test_size=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    x_train.swapaxes(1, 2)
    x_val.swapaxes(1, 2)
    x_test.swapaxes(1, 2)
    x_train = torch.from_numpy(x_train)
    x_val = torch.from_numpy(x_val)
    x_test = torch.from_numpy(x_test)
    # x_train = torch.permute(torch.from_numpy(x_train), (0, 2, 1))
    # x_val = torch.permute(torch.from_numpy(x_val), (0, 2, 1))
    # x_test = torch.permute(torch.from_numpy(x_test), (0, 2, 1))
    # print(x_train.size())
    # print(x_val.size())
    # print(x_test.size())

    train_dat_dict = dict()
    train_dat_dict["samples"] = x_train
    train_dat_dict["labels"] = torch.from_numpy(y_train)

    val_dat_dict = dict()
    val_dat_dict["samples"] = x_val
    val_dat_dict["labels"] = torch.from_numpy(y_val)

    test_dat_dict = dict()
    test_dat_dict["samples"] = x_test
    test_dat_dict["labels"] = torch.from_numpy(y_test)

    train_dataset = Load_Dataset(train_dat_dict, configs, training_mode)
    valid_dataset = Load_Dataset(val_dat_dict, configs, training_mode)
    test_dataset = Load_Dataset(test_dat_dict, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, valid_loader, test_loader


def data_generator2(time_series, time_series_label, configs, training_mode):
    time_series = time_series.to_numpy()
    time_series_label = time_series_label.to_numpy()
    time_series = subsequences(time_series, configs.window_size, configs.time_step)
    time_series_label = subsequences(time_series_label, configs.window_size, configs.time_step)
    # x_train, x_test, y_train, y_test = train_test_split(time_series, time_series_label, test_size=0.5, shuffle=False)
    x_train, x_test, y_train, y_test = train_test_split(time_series, time_series_label, test_size=0.5, random_state=42)
    x_train.swapaxes(1, 2)
    x_test.swapaxes(1, 2)
    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    # x_train = torch.permute(torch.from_numpy(x_train), (0, 2, 1))
    # x_test = torch.permute(torch.from_numpy(x_test), (0, 2, 1))

    y_window_test = np.zeros(y_test.shape[0])
    anomaly_window_num = 0
    # for i, item in enumerate(y_test[:]):
    #     if sum(item[:]) >= 1:
    #         anomaly_window_num += 1
    #         y_window_test[i] = -1
    #     else:
    #         y_window_test[i] = 1
    for i, item in enumerate(y_test[:]):
        if item[configs.window_size - 1] == 1:
        # if item[0] == 1:
            anomaly_window_num += 1
            y_window_test[i] = -1
        else:
            y_window_test[i] = 1
    # print('anomaly window number:')
    # print(anomaly_window_num)
    # print(x_train.size())
    # print(x_val.size())
    # print(x_test.size())

    train_dat_dict = dict()
    train_dat_dict["samples"] = x_train
    train_dat_dict["labels"] = torch.from_numpy(y_train)

    test_dat_dict = dict()
    test_dat_dict["samples"] = x_test
    test_dat_dict["labels"] = y_window_test

    train_dataset = Load_Dataset(train_dat_dict, configs, training_mode)
    test_dataset = Load_Dataset(test_dat_dict, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, test_loader


def data_generator3(train_data, test_data, train_labels, test_labels, configs):
    train_time_series_ts = train_data
    test_time_series_ts = test_data

    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts + test_time_series_ts)
    # mmn = MinMaxNormalize()
    # mmn.train(time_series_ts)
    bias, scale = mvn.bias, mvn.scale
    train_time_series = train_time_series_ts.to_pd().to_numpy()
    train_time_series = (train_time_series-bias)/scale
    test_time_series = test_time_series_ts.to_pd().to_numpy()
    test_time_series = (test_time_series-bias)/scale

    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    train_x = subsequences(train_time_series, configs.window_size, configs.time_step)
    test_x = subsequences(test_time_series, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    anomaly_window_num_train = 0
    anomaly_window_num_test = 0
    for i, item in enumerate(train_y[:]):
        #if item[0] == 1:
        if sum(item[:]) >= 1:
            anomaly_window_num_train += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    train_residual_label = item[1:]
    for i, item in enumerate(test_y[:]):
        # if item[0] == 1:
        if sum(item[:]) >= 1:
            anomaly_window_num_test += 1
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    test_residual_label = item[1:]
    # print('anomaly window number:')
    # print(anomaly_window_num_train)
    # print(anomaly_window_num_test)
    train_x = train_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y_window

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y_window

    train_dataset = Load_Dataset(train_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, test_loader, train_residual_label, test_residual_label