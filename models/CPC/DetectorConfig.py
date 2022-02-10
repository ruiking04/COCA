import torch
import torch.nn as nn
from merlion.models.anomaly.base import DetectorConfig
from merlion.transform.normalize import MeanVarNormalize

from merlion.utils.misc import initializer


class CPCConf(DetectorConfig):
    _default_transform = MeanVarNormalize()

    @initializer
    def __init__(
        self,
        logging_dir='./results/cpc',
        epochs=150,
        n_warmup_steps=100,
        batch_size=256,
        sequence_length=16,
        timestep=2,
        masked_frames=0,
        cuda=torch.cuda.is_available(),
        seed=1,
        log_interval=50,
        **kwargs
    ):
        super(CPCConf, self).__init__(**kwargs)