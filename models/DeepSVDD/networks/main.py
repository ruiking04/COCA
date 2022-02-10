from .timeSeries import Merlion_MLP, Merlion_MLP_Autoencoder
from merlion.models.anomaly.autoencoder import AutoEncoderConfig

def build_network(config):
    """Builds the neural network."""

    implemented_networks = ('merlion')
    assert config.net_name in implemented_networks

    net = None

    if config.net_name == 'merlion':
        net = Merlion_MLP(config)

    return net


def build_autoencoder(config):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('merlion')
    assert config.net_name in implemented_networks

    ae_net = None

    if config.net_name == 'merlion':
        ae_net = Merlion_MLP_Autoencoder(config)

    return ae_net
