import torch

import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from models.TS_TCC.TS_utils import _logger, set_requires_grad
from models.TS_TCC.dataloader import data_generator1, data_generator2
from models.TS_TCC.trainer.trainer import Trainer, model_evaluate
from models.TS_TCC.network.TC import TC
from models.TS_TCC.TS_utils import _calc_metrics, copy_Files
from models.TS_TCC.network.model import base_Model
from ts_datasets.ts_datasets.anomaly import NAB
from ts_datasets.ts_datasets.anomaly import IOpsCompetition
from ts_datasets.ts_datasets.anomaly import SMD
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from models.TS_TCC.trainer.confusion_matrix import ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support
from merlion.evaluate.anomaly import TSADMetric
from merlion.utils import TimeSeries
from utils import label_convert


# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp2', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=2, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='self_supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear, anomaly_detection')
parser.add_argument('--selected_dataset', default='SMD', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD, NAB, IOpsCompetition, SMD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()



device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description
selected_dataset = args.selected_dataset

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

log_dir = experiment_log_dir
# Load datasets
if selected_dataset == 'NAB':
    dt = NAB()
elif selected_dataset == 'IOpsCompetition':
    dt = IOpsCompetition()
else:
    dt = SMD()

if training_mode == "anomaly_detection" in training_mode:
    scores_rpa, scores_pw, scores_pa = [], [], []
    mean_p = 0
    mean_r = 0
    mean_f1 = 0
    adjust_mean_p = 0
    adjust_mean_r = 0
    adjust_mean_f1 = 0
    # Aggregate statistics from full dataset
    l1 = []
    all_test_predict = np.array(l1)
    all_test_target = np.array(l1)

    labels = ['normal', 'anomaly']
    confusion = ConfusionMatrix(num_classes=2, labels=labels)
    for idx in tqdm(range(len(dt))):
        experiment_log_dir = os.path.join(log_dir, selected_dataset, '_' + str(idx))
        time_series, meta_data = dt[idx]
        time_series_label = meta_data.anomaly
        # Load Model
        model = base_Model(configs).to(device)
        temporal_contr_model = TC(configs, device).to(device)
        # load data
        train_dl, test_dl = data_generator2(time_series, time_series_label, configs, training_mode)
        logger.debug("Data loaded ...")
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                         selected_dataset, '_' + str(idx), "saved_models"))
        chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
        pretrained_dict = chkpoint["model_state_dict"]
        model.load_state_dict(pretrained_dict)
        clf = OneClassSVM(gamma='auto', nu=0.01)

        for batch_idx, (data, target, _, _) in enumerate(train_dl):
            data, target = data.float().to(device), target.long().to(device)
            predictions, features = model(data)
            train_features_flat = features.reshape(features.shape[0], -1)
            train_features_flat = train_features_flat.detach().numpy()
            clf = clf.fit(train_features_flat)
            train_anomaly_predict = clf.predict(train_features_flat)
            # print(data)
            # print(target)
            # print('train anomaly predict:')
            # print(train_anomaly_predict)
            # print('train anomaly target:')
            # print(target.detach().numpy())

        for batch_idx, (data, target, _, _) in enumerate(test_dl):
            data, target = data.float().to(device), target.long().to(device)
            predictions, features = model(data)
            test_features_flat = features.reshape(features.shape[0], -1)
            test_features_flat = test_features_flat.detach().numpy()
            test_anomaly_score = clf.score_samples(test_features_flat)
            test_anomaly_predict = clf.predict(test_features_flat)

            confusion.update(test_anomaly_predict, target)
            # use sklearn computing precision, recall, f1 score
            p_r_f1 = precision_recall_fscore_support(target, test_anomaly_predict, labels=[-1, 1])
            mean_p += p_r_f1[0][0]
            mean_r += p_r_f1[1][0]
            mean_f1 += p_r_f1[2][0]

            target = target.numpy()
            target = label_convert(target)
            test_anomaly_predict = label_convert(test_anomaly_predict)
            all_test_target = np.append(all_test_target, target)
            all_test_predict = np.append(all_test_predict, test_anomaly_predict)


            target = pd.DataFrame(target)
            test_anomaly_predict = pd.DataFrame(test_anomaly_predict)
            adjust_p = TSADMetric.Precision.value(ground_truth=TimeSeries.from_pd(target),
                                                  predict=TimeSeries.from_pd(test_anomaly_predict))
            adjust_r = TSADMetric.Recall.value(ground_truth=TimeSeries.from_pd(target),
                                               predict=TimeSeries.from_pd(test_anomaly_predict))
            adjust_f1 = TSADMetric.F1.value(ground_truth=TimeSeries.from_pd(target),
                                            predict=TimeSeries.from_pd(test_anomaly_predict))
            adjust_mean_p += adjust_p
            adjust_mean_r += adjust_r
            adjust_mean_f1 += adjust_f1
            # print(adjust_p)
            # print(adjust_r)
            # print(adjust_f1)
            # print(data)
            # print(target)
            # print('test anomaly predict:')
            # print(test_anomaly_predict)
            # print(test_anomaly_score)
            # print('test anomaly target:')
            # print(target.detach().numpy())

    # according to all time-series confusion matrix computes adjust precision, recall and F1 score
    logger.debug(f"all time-series Revised point-adjusted metrics——precision, recall and F1 score: ")
    # print('all time-series Revised point-adjusted metrics——precision, recall and F1 score:')
    all_test_target = pd.DataFrame(all_test_target)
    all_test_predict = pd.DataFrame(all_test_predict)
    logger.debug(f"Target length is : {len(all_test_target)}")

    all_p = TSADMetric.Precision.value(ground_truth=TimeSeries.from_pd(all_test_target),
                                          predict=TimeSeries.from_pd(all_test_predict))
    all_r = TSADMetric.Recall.value(ground_truth=TimeSeries.from_pd(all_test_target),
                                       predict=TimeSeries.from_pd(all_test_predict))
    all_f1 = TSADMetric.F1.value(ground_truth=TimeSeries.from_pd(all_test_target),
                                    predict=TimeSeries.from_pd(all_test_predict))
    logger.debug(f"Precision : {all_p}")
    logger.debug(f"Recall : {all_r}")
    logger.debug(f"F1 : {all_f1}")
    logger.debug("")

    logger.debug(f"all time-series Point-adjusted metrics——precision, recall and F1 score:")
    all_adjust_p = TSADMetric.PointAdjustedPrecision.value(ground_truth=TimeSeries.from_pd(all_test_target),
                                             predict=TimeSeries.from_pd(all_test_predict))
    all_adjust_r = TSADMetric.PointAdjustedRecall.value(ground_truth=TimeSeries.from_pd(all_test_target),
                                           predict=TimeSeries.from_pd(all_test_predict))
    all_adjust_f1 = TSADMetric.PointAdjustedF1.value(ground_truth=TimeSeries.from_pd(all_test_target),
                                        predict=TimeSeries.from_pd(all_test_predict))
    logger.debug(f"Point-adjusted Precision : {all_adjust_p}")
    logger.debug(f"Point-adjusted Recall : {all_adjust_r}")
    logger.debug(f"Point-adjusted F1 : {all_adjust_f1}")

    # print('all time-series precision, recall and F1 score:')
    # confusion.summary()
    # print()
    #
    # # no weight to computes adjust all time-series precision, recall and F1 score
    # adjust_mean_p = adjust_mean_p / len(dt)
    # adjust_mean_r = adjust_mean_r / len(dt)
    # adjust_mean_f1 = adjust_mean_f1 / len(dt)
    # print('no weight all time-series adjust precision, recall and F1 score:')
    # print(adjust_mean_p)
    # print(adjust_mean_r)
    # print(adjust_mean_f1)
    # print()
    #
    # # no weight to computes all time-series precision, recall and F1 score
    # mean_p = mean_p / len(dt)
    # mean_r = mean_r / len(dt)
    # mean_f1 = mean_f1 / len(dt)
    # print('no weight all time-series precision, recall, F1 score and confusion matrix:')
    # print(mean_p)
    # print(mean_r)
    # print(mean_f1)


else:
    for idx in tqdm(range(len(dt))):
        experiment_log_dir = os.path.join(log_dir, selected_dataset, '_' + str(idx))
        time_series, meta_data = dt[idx]
        time_series_label = meta_data.anomaly
        # # Get training split
        # train = time_series[meta_data.trainval]
        # train_data = TimeSeries.from_pd(train)
        # train_labels = TimeSeries.from_pd(meta_data[meta_data.trainval].anomaly)
        #
        # # Get testing split
        # test = time_series[~meta_data.trainval]
        # test_data = TimeSeries.from_pd(test)
        # test_labels = TimeSeries.from_pd(meta_data[~meta_data.trainval].anomaly)

        # Load Model
        model = base_Model(configs).to(device)
        temporal_contr_model = TC(configs, device).to(device)
        train_dl, valid_dl, test_dl = data_generator1(time_series, time_series_label, configs, training_mode)
        logger.debug("Data loaded ...")
        if training_mode == "fine_tune":
            # load saved model of this experiment
            load_from = os.path.join(
                os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                             selected_dataset, '_'+str(idx), "saved_models"))
            # print(load_from)
            # print(experiment_log_dir)
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
            pretrained_dict = chkpoint["model_state_dict"]
            model_dict = model.state_dict()
            del_list = ['logits']
            pretrained_dict_copy = pretrained_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del pretrained_dict[i]
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        if training_mode == "train_linear" or "tl" in training_mode:
            load_from = os.path.join(
                os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                             selected_dataset, '_' + str(idx), "saved_models"))
            chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
            pretrained_dict = chkpoint["model_state_dict"]
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # delete these parameters (Ex: the linear layer at the end)
            del_list = ['logits']
            pretrained_dict_copy = pretrained_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del pretrained_dict[i]

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

        if training_mode == "random_init":
            model_dict = model.state_dict()

            # delete all the parameters except for logits
            del_list = ['logits']
            pretrained_dict_copy = model_dict.copy()
            for i in pretrained_dict_copy.keys():
                for j in del_list:
                    if j in i:
                        del model_dict[i]
            set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.

        model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                           weight_decay=3e-4)
        temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                                    betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

        if training_mode == "self_supervised":  # to do it only once
            copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

        # Trainer
        Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl,
                device,
                logger, configs, experiment_log_dir, training_mode)

        if training_mode != "self_supervised":
            # Testing
            outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
            total_loss, total_acc, pred_labels, true_labels = outs
            _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)
logger.debug(f"Training time is : {datetime.now()-start_time}")
