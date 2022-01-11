# -*- coding: utf-8 -*-
"""Parameter initialisers.

This module defines classes to initialise the parameters in a layer.
"""

import numpy as np
from mlp import DEFAULT_SEED


class ConstantInit(object):
    """Constant parameter initialiser."""

    def __init__(self, value):
        """Construct a constant parameter initialiser.

        Args:
            value: Value to initialise parameter to.
        """
        self.value = value

    def __call__(self, shape):
        return np.ones(shape=shape) * self.value


class UniformInit(object):
    """Random uniform parameter initialiser."""

    def __init__(self, low, high, rng=None):
        """Construct a random uniform parameter initialiser.

        Args:
            low: Lower bound of interval to sample from.
            high: Upper bound of interval to sample from.
            rng (RandomState): Seeded random number generator.
        """
        self.low = low
        self.high = high
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        return self.rng.uniform(low=self.low, high=self.high, size=shape)


class NormalInit(object):
    """Random normal parameter initialiser."""

    def __init__(self, mean, std, rng=None):
        """Construct a random uniform parameter initialiser.

        Args:
            mean: Mean of distribution to sample from.
            std: Standard deviation of distribution to sample from.
            rng (RandomState): Seeded random number generator.
        """
        self.mean = mean
        self.std = std
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        return self.rng.normal(loc=self.mean, scale=self.std, size=shape)


class SqrtInit(object):
    """Random normal parameter initialiser with variance equal to inverse of input layer dimension."""

    def __init__(self, rng=None):
        """Construct a random uniform parameter initialiser.

        Args:
            rng (RandomState): Seeded random number generator.
        """
        self.mean = 0
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        return self.rng.normal(loc=self.mean, scale=1/np.sqrt(shape[1]), size=shape)


class XavierInit(object):
    """Xavier layer parameter initialiser."""

    def __init__(self, rng=None):
        """Construct a random uniform parameter initialiser.

        Args:
            rng (RandomState): Seeded random number generator.
        """

        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        return self.rng.uniform(low=-np.sqrt(6 / (shape[0] + shape[1])), high=np.sqrt(6 / (shape[0] + shape[1])), size=shape)


class XavierInitNormal(object):
    """Xavier layer parameter initialiser - Normal distribution variation."""

    def __init__(self, rng=None):
        """Construct a random uniform parameter initialiser.

        Args:
            rng (RandomState): Seeded random number generator.
        """
        self.mean = 0
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        return self.rng.normal(loc=self.mean, scale=np.sqrt(2 / (shape[0] + shape[1])), size=shape)


class KaimingInit(object):
    """Kaiming layer parameter initialiser."""

    def __init__(self, rng=None):
        """Construct a random uniform parameter initialiser.

        Args:
            rng (RandomState): Seeded random number generator.
        """
        self.mean = 0
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng

    def __call__(self, shape):
        return self.rng.normal(loc=self.mean, scale=np.sqrt(2 / shape[1]), size=shape)
