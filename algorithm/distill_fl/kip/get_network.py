from neural_tangents import stax
import numpy as np
import functools


def FullyConnectedNetwork(
        depth,
        width,
        W_std=np.sqrt(2),
        b_std=0.1,
        num_classes=10,
        parameterization='ntk',
        activation='relu'):
    """Returns neural_tangents.stax fully connected network."""
    activation_fn = stax.Relu()
    dense = functools.partial(
        stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

    layers = [stax.Flatten()]
    for _ in range(depth):
        layers += [dense(width), activation_fn]
    layers += [stax.Dense(num_classes, W_std=W_std, b_std=b_std,
                          parameterization=parameterization)]

    return stax.serial(*layers)


def FullyConvolutionalNetwork(
        depth,
        width,
        W_std=np.sqrt(2),
        b_std=0.1,
        num_classes=10,
        parameterization='ntk',
        activation='relu'):
    """Returns neural_tangents.stax fully convolutional network."""
    activation_fn = stax.Relu()
    conv = functools.partial(
        stax.Conv,
        W_std=W_std,
        b_std=b_std,
        padding='SAME',
        parameterization=parameterization)

    for _ in range(depth):
        layers += [conv(width, (3, 3)), activation_fn]
    layers += [stax.Flatten(), stax.Dense(num_classes, W_std=W_std, b_std=b_std,
                                          parameterization=parameterization)]

    return stax.serial(*layers)


def MyrtleNetwork(
        depth,
        width,
        W_std=np.sqrt(2),
        b_std=0.1,
        num_classes=10,
        parameterization='ntk',
        activation='relu'):
    """Returns neural_tangents.stax Myrtle network."""
    layer_factor = {5: [1, 1, 1], 7: [1, 2, 2], 10: [2, 3, 3]}
    if depth not in layer_factor.keys():
        raise NotImplementedError(
            'Myrtle network withd depth %d is not implemented!' % depth)
    activation_fn = stax.Relu()
    layers = []
    conv = functools.partial(
        stax.Conv,
        W_std=W_std,
        b_std=b_std,
        padding='SAME',
        parameterization=parameterization)
    layers += [conv(width, (3, 3)), activation_fn]

    # generate blocks of convolutions followed by average pooling for each
    # layer of layer_factor except the last
    for block_depth in layer_factor[depth][:-1]:
        for _ in range(block_depth):
            layers += [conv(width, (3, 3)), activation_fn]
        layers += [stax.AvgPool((2, 2), strides=(2, 2))]

    # generate final blocks of convolution followed by global average pooling
    for _ in range(layer_factor[depth][-1]):
        layers += [conv(width, (3, 3)), activation_fn]
    layers += [stax.GlobalAvgPool()]

    layers += [
        stax.Dense(num_classes, W_std, b_std,
                   parameterization=parameterization)
    ]

    return stax.serial(*layers)
