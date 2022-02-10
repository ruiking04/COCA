from merlion.models.anomaly.base import DetectorConfig
from merlion.post_process.threshold import Threshold
from merlion.transform.normalize import MeanVarNormalize
from merlion.utils.misc import initializer


class OCSVMConf(DetectorConfig):
    _default_transform = MeanVarNormalize()

    @initializer
    def __init__(self, kernel='rbf', nu=0.005, degree=3, gamma='scale', sequence_len=32, **kwargs):
        super(OCSVMConf, self).__init__(**kwargs)
