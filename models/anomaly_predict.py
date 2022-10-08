from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
import numpy as np
import pandas as pd
from merlion.evaluate.anomaly import accumulate_tsad_score, ScoreType
from merlion.utils import TimeSeries
from models.reasonable_metric import tsad_reasonable


def ad_predict(target, scores, mode, nu):
    if_aff = np.count_nonzero(target)
    if if_aff != 0:
        events_gt = convert_vector_to_events(target)
    target = TimeSeries.from_pd(pd.DataFrame(target))
    scores = np.array(scores)
    # standardization
    mean = np.mean(scores)
    std = np.std(scores)
    if std != 0:
        scores = (scores - mean)/std

    # For UCR dataset, there is only one anomaly period in the test set.
    if mode == 'one-anomaly':
        mount = 0
        threshold = np.max(scores, axis=0)
        max_number = np.sum(scores == threshold)
        predict = np.zeros(len(scores))
        if max_number <= 10:
            for index, r2 in enumerate(scores):
                if r2.item() >= threshold:
                    predict[index] = 1
                    mount += 1

        if if_aff != 0:
            events_pred = convert_vector_to_events(predict)
            Trange = (0, len(predict))
            affiliation_max = pr_from_events(events_pred, events_gt, Trange)
        else:
            affiliation_max = dict()
            affiliation_max["precision"] = 0
            affiliation_max["recall"] = 0

        predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
        score_max = accumulate_tsad_score(ground_truth=target, predict=predict_ts)

    # Fixed threshold
    elif mode == 'fix':
        detect_nu = 100 * (1 - nu)
        threshold = np.percentile(scores, detect_nu)
        mount = 0
        predict = np.zeros(len(scores))
        for index, r2 in enumerate(scores):
            if r2.item() > threshold:
                predict[index] = 1
                mount += 1
        if if_aff != 0:
            events_pred = convert_vector_to_events(predict)
            Trange = (0, len(predict))
            affiliation_max = pr_from_events(events_pred, events_gt, Trange)
        else:
            affiliation_max = dict()
            affiliation_max["precision"] = 0
            affiliation_max["recall"] = 0
        predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
        score_max = accumulate_tsad_score(ground_truth=target, predict=predict_ts)

    # Floating threshold
    else:
        nu_list = np.arange(1, 301) / 1e3
        f1_list, score_list, f1_list2, affiliation_list = [], [], [], []
        for detect_nu in nu_list:
            threshold = np.percentile(scores, 100-detect_nu)
            mount = 0
            predict = np.zeros(len(scores))
            for index, r2 in enumerate(scores):
                if r2.item() > threshold:
                    predict[index] = 1
                    mount += 1
            if if_aff != 0:
                events_pred = convert_vector_to_events(predict)
                Trange = (0, len(predict))
                dic = pr_from_events(events_pred, events_gt, Trange)
                affiliation_f1 = 2 * (dic["precision"] * dic["recall"]) / (dic["precision"] + dic["recall"])
                f1_list2.append(affiliation_f1)
            else:
                dic = dict()
                dic["precision"] = 0
                dic["recall"] = 0
                f1_list2.append(0)
            affiliation_list.append(dic)
            predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
            score = accumulate_tsad_score(ground_truth=target, predict=predict_ts)
            f1 = score.f1(ScoreType.RevisedPointAdjusted)
            f1_list.append(f1)
            score_list.append(score)
        index_max1 = np.argmax(f1_list2, axis=0)
        affiliation_max = affiliation_list[index_max1]
        nu_max1 = nu_list[index_max1]
        print("Best affiliation quantile:", nu_max1)

        index_max2 = np.argmax(f1_list, axis=0)
        score_max = score_list[index_max2]
        nu_max2 = nu_list[index_max2]
        print('Best anomaly quantile:', nu_max2)

        threshold = np.percentile(scores, 100 - nu_max1)
        mount = 0
        predict = np.zeros(len(scores))
        for index, r2 in enumerate(scores):
            if r2.item() > threshold:
                predict[index] = 1
                mount += 1
    return affiliation_max, score_max, predict
