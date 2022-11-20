import copy
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from merlion.evaluate.anomaly import (
    accumulate_tsad_score,
    TSADScoreAccumulator as ScoreAcc
)
from merlion.models.anomaly.base import DetectorBase
from merlion.models.ensemble.anomaly import DetectorEnsemble
from merlion.evaluate.anomaly import TSADMetric, ScoreType
from merlion.models.factory import ModelFactory
from merlion.utils import TimeSeries
from merlion.utils.resample import to_pd_datetime

from ts_datasets.ts_datasets.anomaly import *

MERLION_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_JSON = os.path.join(MERLION_ROOT, "..", "conf", "benchmark_anomaly.json")

def evaluate_predictions(
    model_names,
    dataset,
    all_model_preds,
    metric: TSADMetric,
    pointwise_metric: TSADMetric,
    point_adj_metric: TSADMetric,
    tune_on_test=False,
    unsupervised=False,
    debug=False,
):
    scores_rpa, scores_pw, scores_pa = [], [], []
    use_ucr_eval = isinstance(dataset, UCR) and (unsupervised or not tune_on_test)
    for i, (true, md) in enumerate(tqdm(dataset)):
        # Get time series for the train & test splits of the ground truth
        idx = ~md.trainval if tune_on_test else md.trainval
        true_train = df_to_merlion(true[idx], md[idx], get_ground_truth=True)
        true_test = df_to_merlion(true[~md.trainval], md[~md.trainval], get_ground_truth=True)

        for acc_id, (simple_threshold, opt_metric, scores) in enumerate(
            [
                (use_ucr_eval and not tune_on_test, metric, scores_rpa),
                (True, pointwise_metric, scores_pw),
                (True, point_adj_metric, scores_pa),
            ]
        ):
            if acc_id > 0 and use_ucr_eval:
                scores_pw = scores_rpa
                scores_pa = scores_rpa
                continue
            # For each model, load its raw anomaly scores for the i'th time series
            # as a UnivariateTimeSeries, and collect all the models' scores as a
            # TimeSeries. Do this for both the train and test splits.
            if i >= min(len(p) for p in all_model_preds):
                break
            pred = [model_preds[i] for model_preds in all_model_preds]
            pred_train = [p[~p["trainval"]] if tune_on_test else p[p["trainval"]] for p in pred]
            pred_train = [TimeSeries.from_pd(p["y"]) for p in pred_train]
            pred_test = [p[~p["trainval"]] for p in pred]
            pred_test = [TimeSeries.from_pd(p["y"]) for p in pred_test]

            # Train each model's post rule on the train split
            models = []
            for name, train, og_pred in zip(model_names, pred_train, pred):
                m, prtc = get_model(
                    model_name=name,
                    dataset=dataset,
                    metric=opt_metric,
                    tune_on_test=tune_on_test,
                    unsupervised=unsupervised,
                )
                m.config.enable_threshold = len(model_names) == 1
                if simple_threshold:
                    m.threshold = m.threshold.to_simple_threshold()
                if tune_on_test and not unsupervised:
                    m.calibrator.train(TimeSeries.from_pd(og_pred["y"][og_pred["trainval"]]))
                m.train_post_rule(anomaly_scores=train, anomaly_labels=true_train, post_rule_train_config=prtc)
                models.append(m)

            # Get the lead & lag time for the dataset
            early, delay = dataset.max_lead_sec, dataset.max_lag_sec
            if early is None:
                leads = [getattr(m.threshold, "suppress_secs", delay) for m in models]
                leads = [dt for dt in leads if dt is not None]
                early = None if len(leads) == 0 else max(leads)

            # No further training if we only have 1 model
            if len(models) == 1:
                model = models[0]
                pred_test_raw = pred_test[0]

            # If we have multiple models, train an ensemble model
            else:
                threshold = dataset_to_threshold(dataset, tune_on_test)
                ensemble_threshold_train_config = dict(
                    metric=opt_metric if tune_on_test else None,
                    max_early_sec=early,
                    max_delay_sec=delay,
                    unsup_quantile=None,
                )

                # Train the ensemble and its post-rule on the current time series
                model = DetectorEnsemble(models=models)
                use_m = [len(p) > 1 for p in zip(models, pred_train)]
                pred_train = [m.post_rule(p) for m, p, use in zip(models, pred_train, use_m) if use]
                pred_test = [m.post_rule(p) for m, p, use in zip(models, pred_test, use_m) if use]
                pred_train = model.train_combiner(pred_train, true_train)
                if simple_threshold:
                    model.threshold = model.threshold.to_simple_threshold()
                model.threshold.alm_threshold = threshold
                model.train_post_rule(pred_train, true_train, ensemble_threshold_train_config)
                pred_test_raw = model.combiner(pred_test, true_test)

            # For UCR dataset, the evaluation just checks whether the point with the highest
            # anomaly score is anomalous or not.
            if acc_id == 0 and use_ucr_eval and not unsupervised:
                df = pred_test_raw.to_pd()
                df[np.abs(df) < df.max()] = 0
                pred_test = TimeSeries.from_pd(df)
            else:
                pred_test = model.post_rule(pred_test_raw)

            # Compute the individual components comprising various scores.
            score = accumulate_tsad_score(true_test, pred_test, max_early_sec=early, max_delay_sec=delay)

            # Make sure all time series have exactly one detection for UCR dataset (either 1 TP, or 1 FN & 1 FP).
            if acc_id == 0 and use_ucr_eval:
                n_anom = score.num_tp_anom + score.num_fn_anom
                if n_anom == 0:
                    score.num_tp_anom, score.num_fn_anom, score.num_fp = 0, 0, 0
                elif score.num_tp_anom > 0:
                    score.num_tp_anom, score.num_fn_anom, score.num_fp = 1, 0, 0
                else:
                    score.num_tp_anom, score.num_fn_anom, score.num_fp = 0, 1, 1
            scores.append(score)

    # Aggregate statistics from full dataset
    score_rpa = sum(scores_rpa, ScoreAcc())
    score_pw = sum(scores_pw, ScoreAcc())
    score_pa = sum(scores_pa, ScoreAcc())

    # Determine if it's better to have all negatives for each time series if
    # using the test data in a supervised way.
    if tune_on_test and not unsupervised:
        # Convert true positives to false negatives, and remove all false positives.
        # Keep the updated version if it improves F1 score.
        for s in sorted(scores_rpa, key=lambda x: x.num_fp, reverse=True):
            stype = ScoreType.RevisedPointAdjusted
            sprime = copy.deepcopy(score_rpa)
            sprime.num_tp_anom -= s.num_tp_anom
            sprime.num_fn_anom += s.num_tp_anom
            sprime.num_fp -= s.num_fp
            sprime.tp_score -= s.tp_score
            sprime.fp_score -= s.fp_score
            if score_rpa.f1(stype) < sprime.f1(stype):
                # Update anomaly durations
                for duration, delay in zip(s.tp_anom_durations, s.tp_detection_delays):
                    sprime.tp_anom_durations.remove(duration)
                    sprime.tp_detection_delays.remove(delay)
                score_rpa = sprime

        # Repeat for pointwise scores
        for s in sorted(scores_pw, key=lambda x: x.num_fp, reverse=True):
            stype = ScoreType.Pointwise
            sprime = copy.deepcopy(score_pw)
            sprime.num_tp_pointwise -= s.num_tp_pointwise
            sprime.num_fn_pointwise += s.num_tp_pointwise
            sprime.num_fp -= s.num_fp
            if score_pw.f1(stype) < sprime.f1(stype):
                score_pw = sprime

        # Repeat for point-adjusted scores
        for s in sorted(scores_pa, key=lambda x: x.num_fp, reverse=True):
            stype = ScoreType.PointAdjusted
            sprime = copy.deepcopy(score_pa)
            sprime.num_tp_point_adj -= s.num_tp_point_adj
            sprime.num_fn_point_adj += s.num_tp_point_adj
            sprime.num_fp -= s.num_fp
            if score_pa.f1(stype) < sprime.f1(stype):
                score_pa = sprime

    # Compute MTTD & report F1, precision, and recall
    mttd = score_rpa.mean_time_to_detect()
    if mttd < pd.to_timedelta(0):
        mttd = f"-{-mttd}"
    print()
    print("Revised point-adjusted metrics")
    print(f"F1 score:  {score_rpa.f1(ScoreType.RevisedPointAdjusted):.4f}")
    print(f"Precision: {score_rpa.precision(ScoreType.RevisedPointAdjusted):.4f}")
    print(f"Recall:    {score_rpa.recall(ScoreType.RevisedPointAdjusted):.4f}")
    print()
    print(f"Mean Time To Detect Anomalies:  {mttd}")
    print(f"Mean Detected Anomaly Duration: {score_rpa.mean_detected_anomaly_duration()}")
    print(f"Mean Anomaly Duration:          {score_rpa.mean_anomaly_duration()}")
    print()
    if debug:
        print("Pointwise metrics")
        print(f"F1 score:  {score_pw.f1(ScoreType.Pointwise):.4f}")
        print(f"Precision: {score_pw.precision(ScoreType.Pointwise):.4f}")
        print(f"Recall:    {score_pw.recall(ScoreType.Pointwise):.4f}")
        print()
        print("Point-adjusted metrics")
        print(f"F1 score:  {score_pa.f1(ScoreType.PointAdjusted):.4f}")
        print(f"Precision: {score_pa.precision(ScoreType.PointAdjusted):.4f}")
        print(f"Recall:    {score_pa.recall(ScoreType.PointAdjusted):.4f}")
        print()
        print("NAB Scores")
        print(f"NAB Score (balanced):       {score_rpa.nab_score():.4f}")
        print(f"NAB Score (high precision): {score_rpa.nab_score(fp_weight=0.22):.4f}")
        print(f"NAB Score (high recall):    {score_rpa.nab_score(fn_weight=2.0):.4f}")
        print()

    return score_rpa, score_pw, score_pa

def df_to_merlion(df: pd.DataFrame, md: pd.DataFrame, get_ground_truth=False, transform=None) -> TimeSeries:
    """Converts a pandas dataframe time series to the Merlion format."""
    if get_ground_truth:
        if False and "changepoint" in md.keys():
            series = md["anomaly"] | md["changepoint"]
        else:
            series = md["anomaly"]
    else:
        series = df
    time_series = TimeSeries.from_pd(series)
    if transform is not None:
        time_series = transform(time_series)
    return time_series

def dataset_to_threshold(dataset: TSADBaseDataset, tune_on_test=False):
    if isinstance(dataset, IOpsCompetition):
        return 2.25
    elif isinstance(dataset, NAB):
        return 3.5
    elif isinstance(dataset, Synthetic):
        return 2
    elif isinstance(dataset, MSL):
        return 3.0
    elif isinstance(dataset, SMAP):
        return 3.5
    elif isinstance(dataset, SMD):
        return 3 if not tune_on_test else 2.5
    elif hasattr(dataset, "default_threshold"):
        return dataset.default_threshold
    return 3

def get_model(
    model_name: str, dataset: TSADBaseDataset, metric: TSADMetric, tune_on_test=False, unsupervised=False
) -> Tuple[DetectorBase, dict]:
    with open(CONFIG_JSON, "r") as f:
        config_dict = json.load(f)

    if model_name not in config_dict:
        raise NotImplementedError(
            f"Benchmarking not implemented for model {model_name}. Valid model names are {list(config_dict.keys())}"
        )

    while "alias" in config_dict[model_name]:
        model_name = config_dict[model_name]["alias"]

    # Load the model with default kwargs, but override with dataset-specific
    # kwargs where relevant
    model_configs = config_dict[model_name]["config"]
    model_type = config_dict[model_name].get("model_type", model_name)
    model_kwargs = model_configs["default"]
    model_kwargs.update(model_configs.get(type(dataset).__name__, {}))
    model = ModelFactory.create(name=model_type, **model_kwargs)

    # The post-rule train configs are fully specified for each dataset (where
    # relevant), with a default option if there is no dataset-specific option.
    post_rule_train_configs = config_dict[model_name].get("post_rule_train_config", {})
    d = post_rule_train_configs.get("default", {})
    d.update(post_rule_train_configs.get(type(dataset).__name__, {}))
    if len(d) == 0:
        d = copy.copy(model._default_post_rule_train_config)
    d["metric"] = None if unsupervised else metric
    d.update({"max_early_sec": dataset.max_lead_sec, "max_delay_sec": dataset.max_lag_sec})

    t = dataset_to_threshold(dataset, tune_on_test)
    model.threshold.alm_threshold = t
    d["unsup_quantile"] = None
    return model, d

def resolve_model_name(model_name: str):
    with open(CONFIG_JSON, "r") as f:
        config_dict = json.load(f)

    if model_name not in config_dict:
        raise NotImplementedError(
            f"Benchmarking not implemented for model {model_name}. Valid model names are {list(config_dict.keys())}"
        )

    while "alias" in config_dict[model_name]:
        assert model_name != config_dict[model_name]["alias"], "Alias name cannot be the same as the model name"
        model_name = config_dict[model_name]["alias"]

    return model_name

def read_model_predictions(dataset: TSADBaseDataset, model_dir: str):
    """
    Returns a list of lists all_preds, where all_preds[i] is the model's raw
    anomaly scores for time series i in the dataset.
    """
    csv = os.path.join("results", "anomaly", model_dir, f"pred_{dataset_to_name(dataset)}.csv.gz")
    preds = pd.read_csv(csv, dtype={"trainval": bool, "idx": int})
    preds["timestamp"] = to_pd_datetime(preds["timestamp"])
    return [preds[preds["idx"] == i].set_index("timestamp") for i in sorted(preds["idx"].unique())]

def dataset_to_name(dataset: TSADBaseDataset):
    if dataset.subset is not None:
        return f"{type(dataset).__name__}_{dataset.subset}"
    return type(dataset).__name__

class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr