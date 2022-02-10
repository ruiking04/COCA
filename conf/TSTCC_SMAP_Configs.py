class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 25
        self.kernel_size = 4
        self.stride = 1
        self.final_out_channels = 32

        self.num_classes = 2
        self.dropout = 0.5
        # Need to adjust
        self.features_len = 6
        self.window_size = 32
        self.time_step = 1

        # training configs
        self.num_epoch = 40
        # Anomaly Detection parameters
        self.nu = 0.001

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        # True, False. training maybe report errors
        self.drop_last = True
        self.batch_size = 512

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 2
