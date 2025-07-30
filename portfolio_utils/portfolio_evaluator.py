# portfolio_utils/portfolio_evaluator.py

import numpy as np
import pandas as pd
from datetime import datetime

def _repair_weights(weights, epsilon, delta, k, n):
    selected_indices = np.argsort(weights)[-k:]
    selected_epsilon = epsilon[selected_indices]
    selected_delta = delta[selected_indices]
    if np.sum(selected_epsilon) > 1.0:
        return None, selected_indices, True
    final_k_weights = selected_epsilon.copy()
    free_assets_map = list(range(k))
    while True:
        free_capital = 1.0 - np.sum(final_k_weights)
        if free_capital < 1e-9:
            break
        s_i_free = weights[selected_indices[free_assets_map]]
        sum_s_i_free = np.sum(s_i_free)
        if sum_s_i_free > 1e-9:
            distribution = free_capital * (s_i_free / sum_s_i_free)
            final_k_weights[free_assets_map] += distribution
        violating_assets_map = [idx for idx in free_assets_map if final_k_weights[idx] > selected_delta[idx]]
        if not violating_assets_map:
            break
        else:
            for idx in violating_assets_map:
                final_k_weights[idx] = selected_delta[idx]
                free_assets_map.remove(idx)
    final_weights = np.zeros(n)
    final_weights[selected_indices] = final_k_weights
    return final_weights, selected_indices, False

def _calc_metrics(final_weights, returns, cov, lambda_, risk_free_rate):
    # (Esta função permanece inalterada)
    expected_return = np.dot(final_weights, returns)
    variance = np.dot(final_weights, np.dot(cov, final_weights))
    objective = lambda_ * variance - (1 - lambda_) * expected_return
    risk = np.sqrt(variance)
    sharpe = (expected_return - risk_free_rate) / risk if risk > 0 else -1e6
    return expected_return, variance, objective, risk, sharpe

def portfolio_evaluation(weights, instance_data, logger=None, lambda_=0.5, k=None, risk_free_rate=0.03, epsilon=None, delta=None):
    n = instance_data["n_assets"]
    returns = instance_data["returns"]
    cov = instance_data["cov_matrix"]
    weights = np.array(weights)
    if epsilon is None:
        epsilon = instance_data.get("epsilon", np.zeros(n))
    if delta is None:
        delta = instance_data.get("delta", np.ones(n))
    is_constrained = k is not None and k < n
    if not is_constrained:
        k = n
    
    final_weights, selected_indices, infeasible = _repair_weights(weights, epsilon, delta, k, n)
    if infeasible:
        return 1e7, {"error": "Portfólio inviável: soma dos pisos > 1"}
        
    expected_return, variance, objective, risk, sharpe = _calc_metrics(final_weights, returns, cov, lambda_, risk_free_rate)
    final_objective = objective
    
    execution_log = {
        "weights": final_weights.copy().tolist(),
        "selected_assets": np.nonzero(final_weights > 0)[0].tolist(),
        "expected_return": float(expected_return),
        "risk": float(risk),
        "sharpe": float(sharpe),
        "variance": float(variance),
        "objective": float(final_objective),
        "timestamp": datetime.now().isoformat()
    }
    
    if logger:
        logger.log(execution_log)
    
    return objective, execution_log

def configure_problem(instance_data, k=None, risk_free_rate=0.03, lambda_= 0.5):
    n = instance_data["n_assets"]
    lower_bounds = [0.00] * n
    upper_bounds = [1.00] * n
    epsilon = instance_data.get("epsilon", np.zeros(n))
    delta = instance_data.get("delta", np.ones(n))
    
    return {
        "function": lambda weights: portfolio_evaluation(weights, instance_data, lambda_=lambda_, k=k, risk_free_rate=risk_free_rate, epsilon=epsilon, delta=delta)[0],
        "is_constrained": True,
        "boundaries": (lower_bounds, upper_bounds),
    }