import torch

import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from models.TS_TCC.TS_utils import _logger, set_requires_grad
from models.TS_TCC.dataloader import data_generator1, data_generator4
from models.TS_TCC.trainer.trainer import Trainer, model_evaluate
from models.TS_TCC.trainer.oc_train import Trainer as oc_train
from models.TS_TCC.network.TC import TC
from models.TS_TCC.TS_utils import _calc_metrics, copy_Files
from models.TS_TCC.network.model import base_Model
from ts_datasets.ts_datasets.anomaly import NAB, IOpsCompetition, SMAP, UCR
from tqdm import tqdm
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType
from merlion.utils import TimeSeries
from models.reasonable_metric import reasonable_accumulator
from utils import print_object

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp2', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=7, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='anomaly_detection', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear, anomaly_detection')
parser.add_argument('--selected_dataset', default='UCR', type=str,
                    help='Dataset of choice: NAB, IOpsCompetition,  SMAP, UCR')
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


exec(f'from conf.ts_tcc.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
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
else:
    dt = UCR()

if training_mode == "anomaly_detection" in training_mode:
    all_anomaly_num, all_test_score, all_test_scores_reasonable = [], [], []
    all_test_aff_score, all_test_aff_precision, all_test_aff_recall = [], [], []
    detect_list = np.zeros(len(dt))
    for idx in tqdm(range(len(dt))):
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        experiment_log_dir = os.path.join(log_dir, selected_dataset, '_' + str(idx))
        time_series, meta_data = dt[idx]
        train_data = TimeSeries.from_pd(time_series[meta_data.trainval])
        test_data = TimeSeries.from_pd(time_series[~meta_data.trainval])
        train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
        test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
        # Load Model
        model = base_Model(configs).to(device)
        temporal_contr_model = TC(configs, device).to(device)
        # load data
        logger.debug("Data loaded ...")
        train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator4(train_data, test_data,
                                                                             train_labels, test_labels,
                                                                             configs, training_mode)
        # load pre-train model
        load_from = os.path.join(
            os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}",
                         selected_dataset, '_' + str(idx), "saved_models"))
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

        # train one-class classification
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                           weight_decay=1e-4)
        # Trainer
        test_score_origin, test_aff, test_score, score_reasonable, predict = oc_train(model, optimizer, train_dl,
                                                                                     val_dl, test_dl, device, logger,
                                                                                     configs)
        all_anomaly_num.append(test_anomaly_window_num)
        all_test_scores_reasonable.append(score_reasonable)
        all_test_aff_precision.append(test_aff["precision"])
        all_test_aff_recall.append(test_aff["recall"])
        all_test_aff_score.append(test_aff)
        all_test_score.append(test_score)
    all_anomaly_num = np.array(all_anomaly_num)
    sum_anomaly_num = np.sum(all_anomaly_num)
    all_test_aff_precision = np.array(all_test_aff_precision)
    all_test_aff_precision = all_test_aff_precision * all_anomaly_num / sum_anomaly_num
    test_aff_precision = np.nansum(all_test_aff_precision)
    all_test_aff_recall = np.array(all_test_aff_recall)
    all_test_aff_recall = all_test_aff_recall * all_anomaly_num / sum_anomaly_num
    test_aff_recall = np.nansum(all_test_aff_recall)
    test_aff_f1 = 2 * (test_aff_precision * test_aff_recall) / (test_aff_precision + test_aff_recall)

    total_test_score = sum(all_test_score, ScoreAcc())
    total_test_scores_reasonable = sum(all_test_scores_reasonable, reasonable_accumulator())
    print(total_test_scores_reasonable.get_all_metrics())
    print('>' * 32)
    if configs.dataset == 'UCR':
        print("UCR metrics:\n",
              f"accuracy: {total_test_scores_reasonable.get_all_metrics()}\n")
    print("affiliation metrics:\n",
          f"Precision: {test_aff_precision:.5f}\n",
          f"Recall:    {test_aff_recall:.5f}\n"
          f"f1:        {test_aff_f1:.5f}\n"
          "Revised-point-adjusted metrics:\n",
          f"F1 score:  {total_test_score.f1(ScoreType.RevisedPointAdjusted):.5f}\n",
          f"Precision: {total_test_score.precision(ScoreType.RevisedPointAdjusted):.5f}\n",
          f"Recall:    {total_test_score.recall(ScoreType.RevisedPointAdjusted):.5f}\n"
          "Point-adjusted metrics:\n",
          f"F1 score:  {total_test_score.f1(ScoreType.PointAdjusted):.5f}\n",
          f"Precision: {total_test_score.precision(ScoreType.PointAdjusted):.5f}\n",
          f"Recall:    {total_test_score.recall(ScoreType.PointAdjusted):.5f}\n"
          "NAB Scores:\n",
          f"NAB Score (balanced):       {total_test_score.nab_score():.5f}\n",
          f"NAB Score (high precision): {total_test_score.nab_score(fp_weight=0.22):.5f}\n",
          f"NAB Score (high recall):    {total_test_score.nab_score(fn_weight=2.0):.5f}\n"
          "seed:", SEED, "\n"
                         "config setup:\n"
          )
    print_object(configs)
    print_object(configs.augmentation)

else:
    for idx in tqdm(range(len(dt))):
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        experiment_log_dir = os.path.join(log_dir, selected_dataset, '_' + str(idx))
        time_series, meta_data = dt[idx]
        time_series_label = meta_data.anomaly

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
                device, logger, configs, experiment_log_dir, training_mode)

        if training_mode != "self_supervised":
            # Testing
            outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
            total_loss, total_acc, pred_labels, true_labels = outs
            _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)
logger.debug(f"Training time is : {datetime.now()-start_time}")
