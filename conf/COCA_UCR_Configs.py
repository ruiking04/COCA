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

        self.dropout = 0.5
        self.features_len = 6
        self.window_size = 32
        self.time_step = 1

        # training configs
        # If PA F1 metric is concerned, epoch=5. If RPA F1 metric is concerned, epoch=30
        self.num_epoch = 30
        self.freeze_length_epoch = 2
        self.change_center_epoch = 1
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
        # In the experiment, choose the 'floating' mode, PA F1 can reach more than 70%, while RPA F1 is very low;
        # Referring to Melion's method and choosing 'one-anomaly' mode, RPA F1 will exceed 35%.
        self.threshold_determine = 'one-anomaly'
        # Specify COCA objective ("one-class" or "soft-boundary")
        self.objective = 'one-class'
        # Specify loss objective ("arc1","arc2","mix","no_reconstruction", or "distance")
        self.loss_type = 'distance'

        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.8
        self.jitter_ratio = 0.2

