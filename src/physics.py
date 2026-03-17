from classes import OPFData
from typing import Union
import logging 
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
import jax.numpy as jnp
from collections import OrderedDict
from acopf import assess_feasibility
from bnncommon import *
from optax import adam, chain, clip, nadam
from numpyro.infer import Predictive, SVI, Trace_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO
from numpyro import handlers
from jax import random
from jax import jit
import jax
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from acopf import *
from supervisedmodel import *
from stopping import *

def unsupervised_model(
    X_norm, X, 
    opf_data: Union[None, OPFData] = None, 
    vi_parameters = None,
    λ_eq: float = 100.0,
    λ_ineq: float = 10.0,
    λ_cost: float = 0.1
):
    """
    Unsupervised physics-informed BNN for AC-OPF using numpyro.factor.
    Drop-in replacement for the original unsupervised_model.
    
    Implements exactly the objective from equation 20 in your paper:
    ELBO + λ_eq * E[eq_violation²] + λ_ineq * E[ineq_violation²] + λ_cost * E[cost]
    """
    params = get_model_params(opf_data)
    num_data_points, num_inputs = X_norm.shape
    num_layers = params['num_layers']
    num_nodes_per_layer = params['num_nodes_per_hidden_layer']
    
    # Exactly the same multi-head BNN architecture as your original model
    def create_block(block_name: str): 
        z = X_norm
        std_multiplier = params['weight_prior_std_multiplier']
        input_dim_to_layer = num_inputs
        for i in range(num_layers):
            w_shape = (input_dim_to_layer, num_nodes_per_layer)
            w = numpyro.sample(f'{block_name}_w{i+1}', normal(w_shape, std_multiplier))
            b_shape = num_nodes_per_layer
            b = numpyro.deterministic(f'{block_name}_b{i+1}', jnp.zeros(b_shape))
            z = jax.nn.relu(jnp.matmul(z, w) + b)
            input_dim_to_layer = num_nodes_per_layer
        w_out_shape = (num_nodes_per_layer, params['output_block_dim'][block_name])
        w_out = numpyro.sample(f'{block_name}_w_out', normal(w_out_shape, std_multiplier))
        z_out = jnp.matmul(z, w_out)
        return z_out
    
    z = OrderedDict([ (name, create_block(name)) for name in params['output_block_dim'].keys() ])
    z_e = jnp.concatenate(list(z.values()), axis=-1)
    z_e = z_e * opf_data.Y_std + opf_data.Y_mean

    # Calculate all physics terms using your existing acopf functions
    eq_residual = get_equality_constraint_violations(X, z_e, opf_data)
    ineq_residual = get_inequality_constraint_violations(z_e, opf_data)
    cost = get_objective_value(z_e, opf_data)

    # Per-sample penalties (sum of squares for constraints)
    eq_penalty = (eq_residual ** 2).sum(axis=1)
    ineq_penalty = (ineq_residual ** 2).sum(axis=1)
    cost_penalty = cost

    # This is the key part: add all penalties directly to the log joint probability
    # numpyro.factor adds the value to the log likelihood, so we use the negative
    # because we want to minimize the penalty, which is equivalent to maximizing
    # the log probability.
    with numpyro.plate('data', size=num_data_points):
        numpyro.factor('eq_penalty', -λ_eq * eq_penalty)
        numpyro.factor('ineq_penalty', -λ_ineq * ineq_penalty)
        numpyro.factor('cost_penalty', -λ_cost * cost_penalty)

    # That's it! No more hacky Normal sample.