"""
Optimized VOGN Optimizer — Variational Online Gauss-Newton
"""
import acopf
import jax
import jax.numpy as jnp
import optax
from optax import GradientTransformation
from typing import Any, NamedTuple, Union, Callable


def _as_schedule(lr) -> Callable:
    return lr if callable(lr) else lambda _: lr


class VOGNState(NamedTuple):
    prec: Any      
    step: jnp.ndarray


def vogn(
    learning_rate: Union[float, Callable],
    prior_precision: float = 1.0,
    eps: float = 1e-8,
) -> GradientTransformation:
    
    lr_schedule = _as_schedule(learning_rate)
    delta = prior_precision

    def init_fn(params: Any) -> VOGNState:
        prec = jax.tree_util.tree_map(
            lambda p: jnp.full_like(p, delta, dtype=p.dtype),
            params,
        )
        return VOGNState(prec=prec, step=jnp.zeros([], jnp.int32))

    def update_fn(updates: Any, state: VOGNState, params: Any = None):
        # Fail fast: VOGN mathematically requires the current parameter values.
        # Defaulting to zeros (as in the original code) will severely corrupt 
        # the std updates: (1 / sqrt(prec) - 0).
        if params is None:
            raise ValueError("VOGN requires 'params' to compute updates. Ensure NumPyro passes them.")

        step = state.step
        beta = lr_schedule(step)

        # ------------------------------------------------------------------
        # Leaf-level update logic applied directly via tree_map_with_path
        # ------------------------------------------------------------------
        def _vogn_step(path, g, p, old_prec):
            # Convert JAX KeyPath to string to check NumPyro naming conventions
            key_str = jax.tree_util.keystr(path)
            is_std = "_std" in key_str

            # Precision update (Eq. 32)
            # jnp.square(g) is marginally faster than g ** 2 in XLA
            new_prec = (1.0 - beta) * old_prec + beta * (delta + jnp.square(g))
            new_prec = jnp.maximum(new_prec, eps)

            # Standard Python `if` is safe here because `is_std` is a static 
            # boolean based on the parameter name, not a dynamic JAX array.
            if is_std:
                update_leaf = (1.0 / jnp.sqrt(new_prec)) - p
            else:
                # mean update: beta * sigma_{t+1} * (g_mu - delta * mu_t)
                update_leaf = beta * (1.0 / new_prec) * (g - delta * p)

            return update_leaf, new_prec

        # Map the logic across all trees simultaneously.
        # Returns a tree where every leaf is a tuple: (update_leaf, new_prec_leaf)
        results = jax.tree_util.tree_map_with_path(
            _vogn_step, updates, params, state.prec
        )

        # Unzip the resulting tree into two separate PyTrees 
        new_updates = jax.tree_util.tree_map(lambda x: x[0], results)
        new_prec_tree = jax.tree_util.tree_map(lambda x: x[1], results)

        return new_updates, VOGNState(prec=new_prec_tree, step=step + 1)

    return GradientTransformation(init_fn, update_fn)


def clipped_vogn(
    learning_rate: Union[float, Callable],
    clip_norm: float = 10.0,
    prior_precision: float = 1.0,
    eps: float = 1e-8,
) -> GradientTransformation:
    return optax.chain(
        optax.clip(clip_norm),
        vogn(learning_rate, prior_precision=prior_precision, eps=eps),
    )