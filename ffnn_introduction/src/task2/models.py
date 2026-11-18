import klax
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
from typing import Callable

class FFNN(eqx.Module):
    """A custom trainable `equinox.Module`."""

    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, in_features: int, out_features: int, width: int = 16, depth: int = 2,
                 activation= jax.nn.softplus,*, key: PRNGKeyArray):
        sizes = [in_features] + [width] * depth + [out_features]
        keys = jax.random.split(key, depth + 1)
        self.layers = tuple(
            klax.nn.Linear(sizes[i], sizes[i + 1], weight_init=he_normal(), key=keys[i])
            for i in range(depth + 1)
        )


        self.activations = (activation,) * depth + (lambda x: x,)

    def __call__(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x
