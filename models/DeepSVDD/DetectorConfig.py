import torch.nn as nn
from merlion.models.anomaly.base import DetectorConfig
from merlion.transform.normalize import MeanVarNormalize
from merlion.utils.misc import initializer


class DeepSVDDConf(DetectorConfig):
    _default_transform = MeanVarNormalize()

    @initializer
    def __init__(
        self,
        net_name='merlion',
        xp_path='./results/deepsvdd',
        load_model='./results/deepsvdd/deepsvdd.pkl',
        objective='one-class',
        nu=.1,
        device='cpu',
        seed=-1,
        optimizer_name='adam',
        lr=1e-3,
        n_epochs=300,
        lr_milestone=None,
        batch_size=32,
        weight_decay=1e-3,
        pretrain=True,
        ae_optimizer_name='adam',
        ae_lr=1e-3,
        ae_n_epochs=300,
        ae_lr_milestone=None,
        ae_batch_size=32,
        ae_weight_decay=1e-6,
        n_jobs_dataloader=0,
        normal_class=0,

        input_channels=None,
        final_out_channels=None,
        sequence_length=8,
        n_layers=3,
        dropout=0.1,
        hidden_size=5,
        kernel_size=3,
        stride=1,
        **kwargs
    ):
        super(DeepSVDDConf, self).__init__(**kwargs)
