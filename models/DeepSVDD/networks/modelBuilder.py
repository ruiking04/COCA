from .timeSeries import Merlion_MLP, Merlion_MLP_Autoencoder, LSTMEncoder, LSTMAutoEncoder

def build_network(config):
    """Builds the neural network."""

    implemented_networks = ('merlion', 'lstmae')
    assert config.net_name in implemented_networks

    net = None

    if config.net_name == 'merlion':
        net = Merlion_MLP(config)
    elif config.net_name == 'lstmae':
        net = LSTMEncoder(config, config.device)

    return net


def build_autoencoder(config):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('merlion', 'lstmae')
    assert config.net_name in implemented_networks

    ae_net = None

    if config.net_name == 'merlion':
        ae_net = Merlion_MLP_Autoencoder(config)
    elif config.net_name == 'lstmae':
        ae_net = LSTMAutoEncoder(config, config.device)

    return ae_net
