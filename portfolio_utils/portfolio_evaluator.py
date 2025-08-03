# portfolio_utils/portfolio_evaluator.py

import numpy as np
import pandas as pd
from datetime import datetime

def _repair_weights(weights, epsilon, delta, k, n):
    # Validações de entrada
    weights = np.array(weights)
    epsilon = np.array(epsilon)
    delta = np.array(delta)
    
    # Verificar se os arrays têm o tamanho correto
    if len(weights) != n:
        raise ValueError(f"Array weights deve ter tamanho {n}, mas tem tamanho {len(weights)}")
    if len(epsilon) != n:
        raise ValueError(f"Array epsilon deve ter tamanho {n}, mas tem tamanho {len(epsilon)}")
    if len(delta) != n:
        raise ValueError(f"Array delta deve ter tamanho {n}, mas tem tamanho {len(delta)}")
    
    # Verificar se k é válido
    if k > n:
        raise ValueError(f"k ({k}) não pode ser maior que n_assets ({n})")
    if k <= 0:
        raise ValueError(f"k ({k}) deve ser positivo")
        
    # Verificar se epsilon <= delta para todos os ativos
    if np.any(epsilon > delta):
        raise ValueError("epsilon deve ser menor ou igual a delta para todos os ativos")
    
    selected_indices = np.argsort(weights)[-k:]
    selected_epsilon = epsilon[selected_indices]
    selected_delta = delta[selected_indices]
    if np.sum(selected_epsilon) > 1.0:
        return None, selected_indices, True
    final_k_weights = selected_epsilon.copy()
    free_assets_map = list(range(k))
    
    # Proteção contra loop infinito
    max_iterations = 1000
    iteration_count = 0
    
    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            raise RuntimeError(f"Loop infinito detectado na função _repair_weights após {max_iterations} iterações")
            
        free_capital = 1.0 - np.sum(final_k_weights)
        if free_capital < 1e-9:
            break
        
        # Verificar se ainda há ativos livres
        if not free_assets_map:
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
    elif isinstance(epsilon, (int, float)):
        epsilon = np.full(n, epsilon)  # Converte escalar para array
    else:
        epsilon = np.array(epsilon)
    if delta is None:
        delta = instance_data.get("delta", np.ones(n))
    elif isinstance(delta, (int, float)):
        delta = np.full(n, delta)
    else:
        delta = np.array(delta)
    
    if delta is None:
        delta = instance_data.get("delta", np.ones(n))
    
    # Converter para arrays numpy e validar tamanhos
    epsilon = np.array(epsilon)
    delta = np.array(delta)
    
    if len(epsilon) != n:
        return 1e7, {"error": f"Array epsilon deve ter tamanho {n}, mas tem tamanho {len(epsilon)}"}
    if len(delta) != n:
        return 1e7, {"error": f"Array delta deve ter tamanho {n}, mas tem tamanho {len(delta)}"}
    
    # Validar valores
    if np.any(np.isnan(epsilon)) or np.any(np.isnan(delta)) or np.any(np.isnan(weights)):
        return 1e7, {"error": "Valores NaN detectados em epsilon, delta ou weights"}
    
    if np.any(np.isinf(epsilon)) or np.any(np.isinf(delta)) or np.any(np.isinf(weights)):
        return 1e7, {"error": "Valores infinitos detectados em epsilon, delta ou weights"}
    
    if np.any(epsilon < 0) or np.any(delta < 0):
        return 1e7, {"error": "epsilon e delta devem ser não-negativos"}
    
    if np.any(epsilon > delta):
        return 1e7, {"error": "epsilon deve ser menor ou igual a delta para todos os ativos"}
    
    is_constrained = k is not None and k < n
    if not is_constrained:
        k = n
    
    # Validar k
    if k > n:
        return 1e7, {"error": f"k ({k}) não pode ser maior que n_assets ({n})"}
    if k <= 0:
        return 1e7, {"error": f"k ({k}) deve ser positivo"}
    
    try:
        final_weights, _, infeasible = _repair_weights(weights, epsilon, delta, k, n)
        if infeasible:
            return 1e7, {"error": "Portfólio inviável: soma dos pisos > 1"}
    except Exception as e:
        return 1e7, {"error": f"Erro em _repair_weights: {str(e)}"}
        
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
    
    # Ignora soluções onde tanto risco quanto retorno são 0 no log
    if logger and not (risk == 0 or expected_return == 0):
        logger.log(execution_log)
    
    return objective, execution_log

def configure_problem(instance_data, k=None, risk_free_rate=0.03, lambda_= 0.5):
    n = instance_data["n_assets"]
    lower_bounds = [0.01] * n
    upper_bounds = [1.00] * n
    epsilon = instance_data.get("epsilon", np.zeros(n))
    delta = instance_data.get("delta", np.ones(n))
    
    # Validações dos parâmetros
    epsilon = np.array(epsilon)
    delta = np.array(delta)
    
    if len(epsilon) != n:
        raise ValueError(f"Array epsilon deve ter tamanho {n}, mas tem tamanho {len(epsilon)}")
    if len(delta) != n:
        raise ValueError(f"Array delta deve ter tamanho {n}, mas tem tamanho {len(delta)}")
    
    if k is not None and k > n:
        raise ValueError(f"k ({k}) não pode ser maior que n_assets ({n})")
    if k is not None and k <= 0:
        raise ValueError(f"k ({k}) deve ser positivo")
    
    return {
        "function": lambda weights: portfolio_evaluation(weights, instance_data, lambda_=lambda_, k=k, risk_free_rate=risk_free_rate, epsilon=epsilon, delta=delta
                                                         )[0],
        "is_constrained": True,
        "boundaries": (lower_bounds, upper_bounds),
    }