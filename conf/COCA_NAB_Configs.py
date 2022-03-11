class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'NAB'
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 64
        self.hidden_size = 128
        self.num_layers = 3

        self.dropout = 0.5
        self.features_len = 6
        self.window_size = 32
        self.time_step = 1

        # training configs
        self.num_epoch = 1
        self.freeze_length_epoch = 9
        self.change_center_epoch = 2
        self.center_beta = 0.4
        self.center_eps = 0.1

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = False
        self.batch_size = 512

        # Anomaly Detection parameters
        self.nu = 0.01
        self.detect_nu = 0.0005
        # Methods for determining thresholds ("fix","floating","one-anomaly")
        self.threshold_determine = 'floating'
        # Specify COCA objective ("one-class" or "soft-boundary")
        self.objective = 'soft-boundary'
        # Specify loss objective ("arc1","arc2","mix","no_reconstruction", or "distance")
        self.loss_type = 'distance'

        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.8
        self.jitter_ratio = 0.2

