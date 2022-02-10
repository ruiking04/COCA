class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 25
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 32
        self.hidden_size = 64
        self.num_layers = 3

        self.dropout = 0.5
        self.features_len = 6
        self.window_size = 32
        self.time_step = 1

        # training configs
        self.num_epoch = 10
        self.freeze_length_epoch = 9
        self.change_center_epoch = 1
        self.center_beta = 0.4
        self.center_eps = 0.1

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        # True, False. training maybe report errors
        self.drop_last = False
        self.batch_size = 512

        # Anomaly Detection parameters
        self.nu = 0.01
        self.detect_nu = 0.0003
        # Specify COCA objective ("one-class" or "soft-boundary")
        self.objective = 'soft-boundary'
        # Specify loss objective ("arc1" ,"arc2", or "distance")
        self.loss_type = 'distance'

        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 0.4


