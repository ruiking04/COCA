class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'UCR'
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 64

        self.num_classes = 2
        self.dropout = 0.45
        self.features_len = 10
        self.window_size = 64
        self.time_step = 4

        # training configs
        self.num_epoch = 40
        # Anomaly Detection parameters
        self.freeze_length_epoch = 10
        self.nu = 0.01
        self.center_eps = 0.1

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        # True, False. training maybe report errors
        self.drop_last = True
        self.batch_size = 128

        # Specify one-class objective ("one-class" or "soft-boundary")
        self.objective = 'one-class'
        # Methods for determining thresholds ("fix","floating","one-anomaly")
        self.threshold_determine = 'one-anomaly'
        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.0005

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.8
        self.jitter_ratio = 0.2
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 2
