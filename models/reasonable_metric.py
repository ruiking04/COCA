from merlion.evaluate.anomaly import *

class reasonable_accumulator:
    def __init__(
        self,
        cnt=0,
        correct_num=0
    ):
        self.cnt = cnt
        self.correct_num = correct_num

    def __add__(self, acc):
        kwargs = {
            'cnt': self.cnt + acc.cnt,
            'correct_num': self.correct_num + acc.correct_num
        }
        return reasonable_accumulator(**kwargs)

    def get_all_metrics(self):
        return {
            'accuracy': self.correct_num / self.cnt
        }

def tsad_reasonable(
        ground_truth,
        predict,
        time_step
):
    """
    Computes the components required to compute multiple different types of
    performance metrics for time series anomaly detection.

    Anomaly detection metrics for  UCR Time Series Anomaly Detection datasets by
    `Lu, Yue, et al. "Matrix Profile XXIV: Scaling Time Series Anomaly Detection to Trillions of Datapoints and
    Ultra-fast Arriving Data Streams." Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and
    Data Mining. 2022. <https://www.cs.ucr.edu/~eamonn/DAMP_long_version.pdf>`.
    """
    if isinstance(ground_truth, TimeSeries) and isinstance(predict, TimeSeries):
        assert (
                ground_truth.dim == 1 and predict.dim == 1
        ), "Can only evaluate anomaly scores when ground truth and prediction are single-variable time series."
        ground_truth = ground_truth.univariates[ground_truth.names[0]]
        ys = list(map(int, ground_truth.np_values.astype(bool)))

        predict = predict.univariates[predict.names[0]]
        ys_pred = list(map(int, predict.np_values.astype(bool)))
    elif isinstance(ground_truth, np.ndarray) and isinstance(predict, np.ndarray):
        ys = list(map(int, ground_truth.astype(bool)))
        ys_pred = list(map(int, predict.astype(bool)))

    begin, end = -1, -1
    P = -100
    for idx, val in enumerate(ys):
        if idx and ys[idx - 1] == 0 and ys[idx] == 1: begin = idx
        if idx and ys[idx - 1] == 1 and ys[idx] == 0: end = idx; break
    for idx, val in enumerate(ys_pred):
        if val > 0: P = idx; break
    L = end - begin
    L_limit = int(100 / time_step)
    if min(begin - L, begin - L_limit) <= P <= max(end + L, end + L_limit):
        return reasonable_accumulator(1, 1)
    return reasonable_accumulator(1, 0)

if __name__ == '__main__':
    import random
    import numpy as np
    from merlion.utils import TimeSeries

    UCR_ts_nums = 250
    scores_reasonable = []
    for _ in range(UCR_ts_nums):
        # Generating test data
        true_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        if random.random() < 0.3:
            pred_test = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        else:
            pred_test = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # compute reasonable metric
        score_reasonable = tsad_reasonable(true_test, pred_test, 1)
        scores_reasonable.append(score_reasonable)

    score_affiliation_all = sum(scores_reasonable, reasonable_accumulator())
    print(score_affiliation_all.get_all_metrics())
    print('>' * 32)