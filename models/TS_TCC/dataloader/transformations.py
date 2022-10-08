import numpy as np
from tsaug import AddNoise, Convolve, Crop, Drift, Pool, Quantize, Resize, Reverse, TimeWarp

def ts_augmentation(values, augmentations={'TimeWarp', 'Convolve', 'Quantize'}, reduction='none'):
    values = values[np.newaxis, :]
    augmentors = {
        'TimeWarp': TimeWarp(n_speed_change=3, max_speed_ratio=5),
        'Convolve': Convolve(window='flattop', size=32),
        'Quantize': Quantize([10, 50, 90])
    }
    if reduction == 'none':
        results = [augmentors[augmentor].augment(values).squeeze(0) for augmentor in augmentations]
        return np.concatenate(results, axis=0)
    elif reduction == 'sum':
        return sum(augmentors.values(), Reverse() @ 0.03).augment(values).squeeze(0)
