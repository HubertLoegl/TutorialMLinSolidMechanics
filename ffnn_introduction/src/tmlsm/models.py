"""Model implementations."""

from typing import Callable, Literal

import equinox as eqx
import jax
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax

InFeatures = int | Literal["scalar"]

class Model(eqx.Module):
    """A custom trainable `equinox.Module`."""

    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, key: PRNGKeyArray):
        nodes = 1
        self.layers = (
            klax.nn.Linear("scalar", nodes, weight_init=he_normal(), key=key),
            klax.nn.Linear(nodes, nodes, weight_init=he_normal(), key=key),
            klax.nn.Linear(nodes, nodes, weight_init=he_normal(), key=key),
            klax.nn.Linear(nodes, nodes, weight_init=he_normal(), key=key),
            klax.nn.Linear(nodes, "scalar", weight_init=he_normal(), key=key),
        )

        #convex layers
        # self.layers = (
        #     klax.nn.Linear("scalar", nodes, weight_init=he_normal(), key=key),
        #     klax.nn.Linear(nodes, nodes, weight_init=he_normal(), weight_wrap=klax.NonNegative, key=key),
        #     # klax.nn.Linear(nodes, nodes, weight_init=he_normal(), weight_wrap=klax.NonNegative, key=key),
        #     # klax.nn.Linear(nodes, nodes, weight_init=he_normal(), weight_wrap=klax.NonNegative, key=key),
        #     klax.nn.Linear(nodes, "scalar", weight_init=he_normal(), weight_wrap=klax.NonNegative, key=key),
        # )

        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x,
        )

    def __call__(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x


def build(*, key: PRNGKeyArray):
    """Build and return a model instance."""
    return Model(key=key)



class FFNN(eqx.Module):
    # Standard-FFNN R^d -> R (nicht konvex)
    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, in_features: InFeatures = 2, width=16, depth=2,
                 activation=jax.nn.softplus, *, key: PRNGKeyArray):
        keys = jax.random.split(key, depth + 1)
        sizes = [in_features] + [width] * depth + ["scalar"]
        self.layers = tuple(
            klax.nn.Linear(sizes[i], sizes[i + 1], weight_init=he_normal(), key=keys[i])
            for i in range(depth + 1)
        )
        self.activations = (activation,) * depth + (lambda z: z,)

    def __call__(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x


class ICNN(eqx.Module):
    # Input-convex NN R^d -> R
    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, in_features: InFeatures = 2, width=16, depth=2,
                 activation=jax.nn.softplus, *, key: PRNGKeyArray):
        keys = jax.random.split(key, depth + 1)
        sizes = [in_features] + [width] * depth + ["scalar"]
        self.layers = tuple(
            klax.nn.Linear(
                sizes[i], sizes[i + 1], weight_init=he_normal(),
                weight_wrap=None if i == 0 else klax.NonNegative,  # ab Layer 2: W >= 0
                key=keys[i],
            )
            for i in range(depth + 1)
        )
        self.activations = (activation,) * depth + (lambda z: z,)

    def __call__(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return x
