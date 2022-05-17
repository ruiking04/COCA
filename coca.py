import torch

import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from models.TS_TCC.TS_utils import _logger, set_requires_grad
from models.OC_CL.cl_dataloader import data_generator1, data_generator2, data_generator3
from models.OC_CL.cl_trainer.trainer import Trainer, model_evaluate
from models.OC_CL.cl_network.model import base_Model
from ts_datasets.anomaly import NAB, IOpsCompetition, SMAP, SMD, UCR
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support
from merlion.evaluate.anomaly import TSADMetric, TSADScoreAccumulator as ScoreAcc, ScoreType
from merlion.utils import TimeSeries
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
parser.add_argument('--seed', default=2, type=int,
                    help='seed value')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay (L2 penalty) hyperparameter for COCA objective')
parser.add_argument('--selected_dataset', default='UCR', type=str,
                    help='Dataset of choice: NAB, IOpsCompetition, NAB, SMAP, UCR')
parser.add_argument('--logs_save_dir', default='NAB', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'OC-CL'
run_description = args.run_description
selected_dataset = args.selected_dataset
weight_decay = args.weight_decay

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from conf.COCA_{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
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

# Aggregate statistics from full dataset
all_train_score, all_test_score = [], []
for idx in tqdm(range(len(dt))):
# for idx in tqdm(range(1)):
    logger.debug(str(idx)+"time series")
    experiment_log_dir = os.path.join(log_dir, selected_dataset, '_' + str(idx))
    time_series, meta_data = dt[idx]
    train_data = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_data = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])

    # Load Model
    model = base_Model(configs, device).to(device)
    logger.debug("Data loaded ...")
    train_dl, test_dl, train_residual_label, test_residual_label = data_generator3(train_data, test_data,
                                                                                   train_labels, test_labels,
                                                                                   configs)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=weight_decay)

    # Trainer
    train_score, test_score = Trainer(model, model_optimizer, train_dl, test_dl,
                                      train_residual_label, test_residual_label,
                                      device, logger, configs, experiment_log_dir)
    # all_train_score.append(train_score)
    all_test_score.append(test_score)

# total_train_score = sum(all_train_score, ScoreAcc())
total_test_score = sum(all_test_score, ScoreAcc())

print("Revised-point-adjusted metrics")
# print("train")
# print(f"F1 score:  {total_train_score.f1(ScoreType.RevisedPointAdjusted):.5f}")
# print(f"Precision: {total_train_score.precision(ScoreType.RevisedPointAdjusted):.5f}")
# print(f"Recall:    {total_train_score.recall(ScoreType.RevisedPointAdjusted):.5f}")
print("test")
print(f"F1 score:  {total_test_score.f1(ScoreType.RevisedPointAdjusted):.5f}")
print(f"Precision: {total_test_score.precision(ScoreType.RevisedPointAdjusted):.5f}")
print(f"Recall:    {total_test_score.recall(ScoreType.RevisedPointAdjusted):.5f}")
print()

print("Point-adjusted metrics")
# print("train")
# print(f"F1 score:  {total_train_score.f1(ScoreType.PointAdjusted):.5f}")
# print(f"Precision: {total_train_score.precision(ScoreType.PointAdjusted):.5f}")
# print(f"Recall:    {total_train_score.recall(ScoreType.PointAdjusted):.5f}")
print("test")
print(f"F1 score:  {total_test_score.f1(ScoreType.PointAdjusted):.5f}")
print(f"Precision: {total_test_score.precision(ScoreType.PointAdjusted):.5f}")
print(f"Recall:    {total_test_score.recall(ScoreType.PointAdjusted):.5f}")
print()

print("NAB Scores")
# print("train")
# print(f"NAB Score (balanced):       {total_train_score.nab_score():.5f}")
# print(f"NAB Score (high precision): {total_train_score.nab_score(fp_weight=0.22):.5f}")
# print(f"NAB Score (high recall):    {total_train_score.nab_score(fn_weight=2.0):.5f}")
print("test")
print(f"NAB Score (balanced):       {total_test_score.nab_score():.5f}")
print(f"NAB Score (high precision): {total_test_score.nab_score(fp_weight=0.22):.5f}")
print(f"NAB Score (high recall):    {total_test_score.nab_score(fn_weight=2.0):.5f}")
print()

print('config setup:')
print_object(configs)
print_object(configs.augmentation)
logger.debug(f"Training time is : {datetime.now()-start_time}")
