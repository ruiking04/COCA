class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'UCR'
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 64
        self.hidden_size = 128
        self.num_layers = 3
        self.project_channels = 32

        self.dropout = 0.45
        self.features_len = 18
        self.window_size = 64
        self.time_step = 4

        # training configs
        self.num_epoch = 50
        self.freeze_length_epoch = 10
        self.change_center_epoch = 10
        self.center_eps = 0.1
        self.omega1 = 1
        self.omega2 = 0.1


        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = False
        self.batch_size = 512

        # Anomaly Detection parameters
        self.nu = 0.01
        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.0005
        # Methods for determining thresholds ("fix","floating","one-anomaly")
        self.threshold_determine = 'one-anomaly'
        # Specify COCA objective ("one-class" or "soft-boundary")
        self.objective = 'one-class'
        # Specify loss objective ("arc1","arc2","mix","no_reconstruction", or "distance")
        self.loss_type = 'distance'

        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.scale_ratio = 0.8
        self.jitter_ratio = 0.2

