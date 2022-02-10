import numpy as np
import os

# convert label 1 normal and other value abnormal to label 0 normal and 1 abnormal
def label_convert(target):
    for inx, value in enumerate(target):
        if value == 1:
            target[inx] = 0
        else:
            target[inx] = 1
    return target


# according to window size and time step to create subsequence
def subsequences(sequence, window_size, time_step):
    # An array of non-contiguous memory is converted to an array of contiguous memory
    sq = np.ascontiguousarray(sequence)
    a = (sq.shape[0] - window_size + time_step) % time_step
    # label array
    if sq.ndim == 1:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size)
        stride = sq.itemsize * np.array([time_step * 1, 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a]
    # data array
    elif sq.ndim == 2:
        shape = (int((sq.shape[0] - window_size + time_step) / time_step), window_size, sq.shape[1])
        stride = sq.itemsize * np.array([time_step * sq.shape[1], sq.shape[1], 1])
        if a != 0:
            sq = sq[:sq.shape[0] - a, :]
    else:
        print('Array dimension error')
        os.exit()
    # print(sq.strides)
    sq = np.lib.stride_tricks.as_strided(sq, shape=shape, strides=stride)
    return sq


# Prints all property values of the object
def print_object(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))