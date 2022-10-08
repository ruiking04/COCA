from merlion.evaluate.anomaly import *
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
from IPython import embed


class affiliation_accumulator:
    def __init__(
        self,
        cnt=0,
        precision=0,
        recall=0,
        individual_precision_probabilities=0,
        individual_recall_probabilities=0,
        individual_precision_distances=0,
        individual_recall_distances=0
    ):
        self.cnt = cnt
        self.precision = precision
        self.recall = recall
        self.individual_precision_probabilities = individual_precision_probabilities
        self.individual_recall_probabilities = individual_recall_probabilities
        self.individual_precision_distances = individual_precision_distances
        self.individual_recall_distances = individual_recall_distances

    def __add__(self, acc):
        kwargs = {
            'cnt': self.cnt + acc.cnt,
            'precision': self.precision + acc.precision,
            'recall': self.recall + acc.recall
        }
        return affiliation_accumulator(**kwargs)

    def get_all_metrics(self):
        return {
            'precision': self.precision / self.cnt,
            'recall': self.recall / self.cnt,
            # 'individual_precision_probabilities': self.individual_precision_probabilities / self.cnt,
            # 'individual_recall_probabilities': self.individual_recall_probabilities / self.cnt,
            # 'individual_precision_distances': self.individual_precision_distances / self.cnt,
            # 'individual_recall_distances': self.individual_recall_distances / self.cnt
        }

def tsad_affiliation(
        ground_truth: TimeSeries,
        predict: TimeSeries
):
    """
    Computes the components required to compute multiple different types of
    performance metrics for time series anomaly detection.
    """
    if isinstance(ground_truth, TimeSeries):
        assert (
                ground_truth.dim == 1 and predict.dim == 1
        ), "Can only evaluate anomaly scores when ground truth and prediction are single-variable time series."
        ground_truth = ground_truth.univariates[ground_truth.names[0]]
        ys = list(map(int, ground_truth.np_values.astype(bool)))

        predict = predict.univariates[predict.names[0]]
        ys_pred = list(map(int, predict.np_values.astype(bool)))
    elif isinstance(ground_truth, np.ndarray):
        ys = list(map(int, ground_truth.astype(bool)))
        ys_pred = list(map(int, predict.astype(bool)))

    try:
        events_pred = convert_vector_to_events(ys_pred)
        events_gt = convert_vector_to_events(ys)
        if len(events_gt) == 0: return affiliation_accumulator(cnt=0)
        if len(events_pred) == 0: return affiliation_accumulator(cnt=1)
        result = pr_from_events(events_pred, events_gt, Trange=(0, max([max(x) for x in events_pred] + [max(x) for x in events_gt])))
        result['cnt'] = 1
    except Exception as e:
        print(e)
        embed()

    return affiliation_accumulator(**result)

if __name__ == '__main__':
    import numpy as np
    from merlion.utils import TimeSeries

    KPI_ts_nums = 29
    scores_affiliation = []
    for _ in range(KPI_ts_nums):
        # 生成测试数据
        true_test = np.random.randint(0, 2, 366)
        pred_test = np.random.randint(0, 2, 366)

        # 计算affiliation metric
        score_affiliation = tsad_affiliation(true_test, pred_test)
        scores_affiliation.append(score_affiliation)

    score_affiliation_all = sum(scores_affiliation, affiliation_accumulator())
    print('Affiliation Metrics')
    print('>' * 32)
    print(score_affiliation_all.get_all_metrics())
    print('>' * 32)