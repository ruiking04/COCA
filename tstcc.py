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
from ts_datasets.ts_datasets.anomaly import NAB, IOpsCompetition, SMAP, SMD, UCR
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from models.TS_TCC.trainer.confusion_matrix import ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support
from merlion.evaluate.anomaly import TSADMetric, accumulate_tsad_score, TSADScoreAccumulator as ScoreAcc, ScoreType
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
parser.add_argument('--training_mode', default='anomaly_detection', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear, anomaly_detection')
parser.add_argument('--selected_dataset', default='SMAP', type=str,
                    help='Dataset of choice: NAB, IOpsCompetition,  SMAP, UCR, SMD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cpu', type=str,
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


exec(f'from conf.TSTCC_{data_type}_Configs import Config as Configs')
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
elif selected_dataset == 'SMAP':
    dt = SMAP()
elif selected_dataset == 'UCR':
    dt = UCR()
else:
    dt = SMD()

if training_mode == "anomaly_detection" in training_mode:
    all_test_score = []

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
        clf = OneClassSVM(gamma='auto', nu=configs.nu)

        # train one-class
        for batch_idx, (data, target, _, _) in enumerate(train_dl):
            data, target = data.float().to(device), target.long().to(device)
            predictions, features = model(data)
            train_features_flat = features.reshape(features.shape[0], -1)
            train_features_flat = train_features_flat.detach().cpu().numpy()
            clf = clf.fit(train_features_flat)
            train_anomaly_predict = clf.predict(train_features_flat)

        all_test_target, all_test_predict = [], []
        for batch_idx, (data, target, _, _) in enumerate(test_dl):
            data, target = data.float().to(device), target.long().to(device)
            predictions, features = model(data)
            test_features_flat = features.reshape(features.shape[0], -1)
            test_features_flat = test_features_flat.detach().cpu().numpy()
            test_anomaly_score = clf.score_samples(test_features_flat)
            test_anomaly_predict = clf.predict(test_features_flat)

            target = target.cpu().numpy()
            target = label_convert(target)
            test_anomaly_predict = label_convert(test_anomaly_predict)
            # all_test_target = np.append(all_test_target, target)
            # all_test_predict = np.append(all_test_predict, test_anomaly_predict)

            # target = pd.DataFrame(target)
            # test_anomaly_predict = pd.DataFrame(test_anomaly_predict)
            all_test_target.extend(target)
            all_test_predict.extend(test_anomaly_predict)
        all_test_target = TimeSeries.from_pd(pd.DataFrame(all_test_target))
        all_test_predict = TimeSeries.from_pd(pd.DataFrame(all_test_predict))
        test_score = accumulate_tsad_score(ground_truth=all_test_target, predict=all_test_predict)
        all_test_score.append(test_score)
        test_f1 = test_score.f1(ScoreType.RevisedPointAdjusted)
        test_precision = test_score.precision(ScoreType.RevisedPointAdjusted)
        test_recall = test_score.recall(ScoreType.RevisedPointAdjusted)

        logger.debug(f'\ntest F1     : {test_f1:2.4f}  | \ttest precision     : {test_precision:2.4f}  | \ttest recall     : {test_recall:2.4f}\n')

    total_test_score = sum(all_test_score, ScoreAcc())
    print("Revised-point-adjusted metrics")
    print("test")
    print(f"F1 score:  {total_test_score.f1(ScoreType.RevisedPointAdjusted):.5f}")
    print(f"Precision: {total_test_score.precision(ScoreType.RevisedPointAdjusted):.5f}")
    print(f"Recall:    {total_test_score.recall(ScoreType.RevisedPointAdjusted):.5f}")
    print()

    print("Point-adjusted metrics")
    print("test")
    print(f"F1 score:  {total_test_score.f1(ScoreType.PointAdjusted):.5f}")
    print(f"Precision: {total_test_score.precision(ScoreType.PointAdjusted):.5f}")
    print(f"Recall:    {total_test_score.recall(ScoreType.PointAdjusted):.5f}")
    print()

    print("NAB Scores")
    print("test")
    print(f"NAB Score (balanced):       {total_test_score.nab_score():.5f}")
    print(f"NAB Score (high precision): {total_test_score.nab_score(fp_weight=0.22):.5f}")
    print(f"NAB Score (high recall):    {total_test_score.nab_score(fn_weight=2.0):.5f}")
    print()

else:
    for idx in tqdm(range(len(dt))):
        print(idx + "time series")
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
