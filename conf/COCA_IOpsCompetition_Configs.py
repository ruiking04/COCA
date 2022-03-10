class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 4
        self.stride = 2
        self.final_out_channels = 64
        self.hidden_size = 128
        self.num_layers = 3

        self.dropout = 0.5
        # self.features_len = 4
        # self.window_size = 32
        self.features_len = 3
        self.window_size = 20
        self.time_step = 1

        # training configs
        self.num_epoch = 1
        self.freeze_length_epoch = 2
        self.change_center_epoch = 1
        self.center_beta = 0.4
        self.center_eps = 0.1

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-4

        # data parameters
        self.drop_last = False
        self.batch_size = 128

        # Anomaly Detection parameters
        self.nu = 0.01
        self.detect_nu = 0.0015
        # Specify COCA objective ("one-class" or "soft-boundary")
        self.objective = 'soft-boundary'
        # Specify loss objective ("arc1","arc2","mix","no_reconstruction", or "distance")
        self.loss_type = 'distance'

        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.8
        self.jitter_ratio = 0.3

