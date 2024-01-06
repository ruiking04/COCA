import torch

import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from dataloader import data_generator1
from models.reasonable_metric import reasonable_accumulator
from ts_datasets.ts_datasets.anomaly import NAB, IOpsCompetition, SMAP, SMD, UCR
from tqdm import tqdm
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType
from merlion.utils import TimeSeries
from utils import print_object
import matplotlib.pyplot as plt


# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--visualization', default=True, type=bool,
                    help='Visualize')
parser.add_argument('--seed', default=2, type=int,
                    help='seed value')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay (L2 penalty) hyperparameter for COCA objective')
parser.add_argument('--selected_dataset', default='UCR', type=str,
                    help='Dataset of choice: NAB, IOpsCompetition, SMAP, UCR')
parser.add_argument('--selected_model', default='COCA_view', type=str,
                    help='Model of choice: COCA, COCA_no_aug, COCA_no_cl, COCA_no_oc, COCA_no_var, COCA_view,')
parser.add_argument('--device', default='cpu', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = args.selected_model
run_description = args.run_description
selected_dataset = args.selected_dataset
weight_decay = args.weight_decay
visualization = args.visualization

if method == 'COCA_no_aug':
    from models.COCA.coca_trainer.trainer_no_aug import Trainer
    from models.COCA.coca_network.model_no_aug import base_Model
elif method == 'COCA_no_cl':
    from models.COCA.coca_trainer.trainer_no_cl import Trainer
    from models.COCA.coca_network.model_no_cl import base_Model
elif method == 'COCA_no_oc':
    from models.COCA.coca_trainer.trainer_no_oc import Trainer
    from models.COCA.coca_network.model_no_oc import base_Model
elif method == 'COCA_no_var':
    from models.COCA.coca_trainer.trainer_no_var import Trainer
    from models.COCA.coca_network.model_no_var import base_Model
elif method == 'COCA_view':
    from models.COCA.coca_trainer.trainer_view import Trainer
    from models.COCA.coca_network.model_view import base_Model
else:
    from models.COCA.coca_trainer.trainer import Trainer
    from models.COCA.coca_network.model import base_Model

exec(f'from conf.coca.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed

print("=" * 45)
print(f'Dataset: {data_type}')
print(f'Method:  {method}')
print("=" * 45)

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
all_anomaly_num, all_test_score, all_test_scores_reasonable = [], [], []
all_test_aff_score, all_test_aff_precision, all_test_aff_recall = [], [], []

model_num = len(dt)
detect_list = np.zeros(model_num)
for idx in tqdm(range(model_num)):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    time_series, meta_data = dt[idx]
    train_data = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_data = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])

    print('>' * 32, len(train_data))
    print('>' * 32, len(test_data))

    # Load Model
    model = base_Model(configs, device).to(device)
    print("Data loaded ...")
    train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator1(train_data, test_data, train_labels, test_labels, configs)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=weight_decay)

    # Trainer
    test_score_origin, test_aff, test_score, score_reasonable, predict = Trainer(model, model_optimizer, train_dl,
                                                                                 val_dl, test_dl, device,
                                                                                 configs, idx)

    all_anomaly_num.append(test_anomaly_window_num)
    all_test_scores_reasonable.append(score_reasonable)
    all_test_aff_precision.append(test_aff["precision"])
    all_test_aff_recall.append(test_aff["recall"])
    all_test_aff_score.append(test_aff)
    all_test_score.append(test_score)

    # visualization
    if visualization:
        print('*'*32)
        fig = plt.figure(facecolor="w", figsize=(10, 6))
        ax = fig.add_subplot(111)
        # train_data_plot = time_series[meta_data.trainval]
        # train_labels_plot = meta_data.anomaly[meta_data.trainval]
        test_data_plot = time_series[~meta_data.trainval]
        test_labels_plot = meta_data.anomaly[~meta_data.trainval]
        # plot time-series value
        t_data, y_data = test_data_plot.index, test_data_plot.values
        t = np.arange(0, len(y_data), 1)
        g = len(y_data.shape)
        if g > 1:
            y_data = y_data[:, 0]
        ax.plot(t, y_data, linewidth=1)
        ax.set_ylabel('value', fontsize=10)

        # plot ground-truth anomaly
        t_label, y_label = test_labels_plot.index, test_labels_plot.values
        splits = np.where(y_label[1:] != y_label[:-1])[0] + 1
        splits = np.concatenate(([0], splits, [len(y_label) - 1]))
        for k in range(len(splits) - 1):
            if y_label[splits[k]]:  # If splits[k] is anomalous
                ax.axvspan(t[splits[k]], t[splits[k + 1]], color="#e07070", alpha=0.5)
        # plot predict anomaly score
        predict = np.tile(predict.reshape(-1, 1), configs.time_step).flatten()
        t_pred = np.arange(0, len(predict), 1)
        ax2 = ax.twinx()
        ax2.set_ylabel('anomaly', fontsize=10)
        ax2.plot(t_pred, predict, linewidth=1, color='r')
        time_series_name = test_data_plot.columns[0]
        plt.title(time_series_name + '_' + str(idx))
        plt.show()

        fig_origin = plt.figure(facecolor="w", figsize=(20, 12))
        ax_origin = fig_origin.add_subplot(111)
        test_score_origin = np.array(test_score_origin).reshape(-1, 1)
        test_score_origin = np.tile(test_score_origin, configs.time_step).flatten()
        ax_origin.plot(test_score_origin, linewidth=1)
        plt.tight_layout()
        plt.show()
        score_item = test_score.f1(ScoreType.RevisedPointAdjusted)
        if score_item > 0:
            detect_list[idx] = score_item


# visualization
if visualization:
    fig_all = plt.figure(facecolor="w", figsize=(20, 12))
    ax_all = fig_all.add_subplot(111)
    ax_all.plot(detect_list, linewidth=1)
    plt.show()
    np.savetxt("detect_list_coca.csv", detect_list, delimiter=",")

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
ucr_accuracy = total_test_scores_reasonable.get_all_metrics()

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
str_conf = print_object(configs)
train_time = datetime.now()-start_time
print(f"Training time is : {train_time}")

path = "./results"
if not os.path.exists(path):
    os.makedirs(path)
summary = os.path.join("results", f"{method}_{selected_dataset}_summary.csv")
if os.path.exists(summary):
    df = pd.read_csv(summary, index_col=0)
else:
    df = pd.DataFrame()
model_name = method + f"{df.shape[1]}"

df.loc["Hyper-parameter", model_name] = str_conf
df.loc["seed", model_name] = SEED
df.loc["Train Time", model_name] = train_time
df.loc["UCR Accuracy", model_name] = round(ucr_accuracy["accuracy"], 5)
df.loc["Affiliation Precision", model_name] = round(test_aff_precision, 5)
df.loc["Affiliation Recall", model_name] = round(test_aff_recall, 5)
df.loc["Affiliation F1", model_name] = round(test_aff_f1, 5)
df.loc["RPA Precision", model_name] = round(total_test_score.precision(ScoreType.RevisedPointAdjusted), 5)
df.loc["RPA Recall", model_name] = round(total_test_score.recall(ScoreType.RevisedPointAdjusted), 5)
df.loc["RPA F1", model_name] = round(total_test_score.f1(ScoreType.RevisedPointAdjusted), 5)
df.loc["PA Precision", model_name] = round(total_test_score.precision(ScoreType.PointAdjusted), 5)
df.loc["PA Recall", model_name] = round(total_test_score.recall(ScoreType.PointAdjusted), 5)
df.loc["PA F1", model_name] = round(total_test_score.f1(ScoreType.PointAdjusted), 5)
df.loc["Point-wise Precision", model_name] = round(total_test_score.precision(ScoreType.Pointwise), 5)
df.loc["Point-wise Recall", model_name] = round(total_test_score.recall(ScoreType.Pointwise), 5)
df.loc["Point-wise F1", model_name] = round(total_test_score.f1(ScoreType.Pointwise), 5)
df.loc["NAB Score (balanced)", model_name] = round(total_test_score.nab_score(), 5)
df.loc["NAB Score (high precision)", model_name] = round(total_test_score.nab_score(fp_weight=0.22), 5)
df.loc["NAB Score (high recall)", model_name] = round(total_test_score.nab_score(fn_weight=2.0), 5)


df.to_csv(summary, index=True)
