"""Implementations of objective functions."""

import jax
import jax.numpy as jnp
import klax


class MSE(klax.Loss):
    """Reimplementation of `klax.MSE` loss function."""

    def __call__(self, model, batch, batch_axis):
        x, y = batch
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y - y_pred))

class SobolevMSE(klax.Loss):
    w_f: float
    w_g: float

    def __init__(self, w_f=1.0, w_g=1.0):
        self.w_f, self.w_g = w_f, w_g

    def __call__(self, model, batch, batch_axis):
        x, (y, dy) = batch
        y_p, dy_p = jax.vmap(model)(x)
        return (self.w_f * jnp.mean(jnp.square(y - y_p))
                + self.w_g * jnp.mean(jnp.square(dy - dy_p)))
