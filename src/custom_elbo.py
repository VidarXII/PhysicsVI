from supervisedmodel import *
from unsupervisedmodel import * 
from bnncommon import *
from acopf import *
from numpyro.infer import Predictive, SVI, Trace_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO

class OPF_ELBO(TraceMeanField_ELBO):
    """
    Custom ELBO for AC-OPF that adds weighted penalties for feasibility and cost.
    
    This ELBO is 100% backwards compatible with any model. If the model exposes
    the deterministic terms `eq_penalty`, `ineq_penalty` or `cost_penalty`, they
    will be added to the loss with the corresponding weights. If they are not
    present, this ELBO behaves exactly like the standard TraceMeanField_ELBO.
    
    This allows using the exact same ELBO for supervised, unsupervised and hybrid
    training.
    """
    def __init__(
        self, 
        num_particles: int = 1,
        lambda_eq: float = 0.0,
        lambda_ineq: float = 0.0,
        lambda_cost: float = 0.0,
        vectorize_particles: bool = True
    ):
        super().__init__(num_particles=num_particles, vectorize_particles=vectorize_particles)
        self.lambda_eq = lambda_eq
        self.lambda_ineq = lambda_ineq
        self.lambda_cost = lambda_cost

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        result = self.loss_with_mutable_state(
        rng_key, param_map, model, guide, *args, **kwargs)
        # 1. FIXED UNPACKING: Expecting 2 values instead of 3
        elbo = result["loss"]
        model_trace = result["mutable_state"]

        additional_loss = 0.0

        # Add penalties if they exist in the model trace
        if 'eq_penalty' in model_trace:
            additional_loss += self.lambda_eq * model_trace['eq_penalty']['value'].mean()
        
        if 'ineq_penalty' in model_trace:
            additional_loss += self.lambda_ineq * model_trace['ineq_penalty']['value'].mean()
        
        if 'cost_penalty' in model_trace:
            additional_loss += self.lambda_cost * model_trace['cost_penalty']['value'].mean()

        total_loss = elbo + additional_loss

        return total_loss