# portfolio_utils/portfolio_evaluator.py

import numpy as np
import pandas as pd
from datetime import datetime
from portfolio_utils.parquet_handler import ParquetBufferWriter

class StatefulPortfolioEvaluator:
    """
    Avaliador com estado otimizado que salva TODAS as avaliações em Parquet.
    
    Implementa __call__ para ser usada como função objetivo pelo CUSTOMHYS.
    - Salva TODA avaliação (não apenas melhorias) para manter diversidade Pareto
    - Usa ParquetBufferWriter para otimizar I/O (escreve em lotes)
    - Compressão automática reduz espaço sem perder qualidade de dados
    
    Atributos:
        instance_data: Dados da instância (ativos, returns, cov_matrix)
        logger: ParquetBufferWriter para salvar em Parquet
        lambda_: Parâmetro de trade-off retorno/risco
        k: Restrição de cardinalidade
        epsilon: Pisos mínimos por ativo
        delta: Tetos máximos por ativo
        
    Exemplos:
        >>> evaluator = StatefulPortfolioEvaluator(
        ...     instance_data, 
        ...     logger=parquet_writer,
        ... )
        >>> objective = evaluator(weights)  # Chamada como função
    """
    
    def __init__(self, instance_data, logger=None, lambda_=0.5, k=None, 
                 epsilon=None, delta=None):
        self.instance_data = instance_data
        self.logger = logger
        self.lambda_ = lambda_
        self.k = k
        self.epsilon = epsilon if epsilon is not None else instance_data.get("epsilon", np.zeros(instance_data["n_assets"]))
        self.delta = delta if delta is not None else instance_data.get("delta", np.ones(instance_data["n_assets"]))
        
        # Estado interno - "Memória" da classe
        self.eval_count = 0
        self.best_objective_so_far = float('inf')
        self.best_solution_so_far = None
        self.best_weights_so_far = None
        
        logger_type = "ParquetBufferWriter" if isinstance(logger, ParquetBufferWriter) else "PortfolioLogger"
        print(f"[STATEFUL] Avaliador inicializado - Salva TODAS as avaliações em {logger_type}")
    
    def __call__(self, weights):
        """
        Permite que a instância seja chamada como função: evaluator(weights)
        
        O CUSTOMHYS vai chamar isso sem saber que é uma classe com memória.
        """
        self.eval_count += 1
        
        # 1. Executa a avaliação padrão (sem logar cada detalhe internamente)
        objective, log_data = portfolio_evaluation(
            weights, 
            self.instance_data, 
            logger=None,  # Desligamos o logger interno - vamos salvar aqui!
            lambda_=self.lambda_, 
            k=self.k, 
            epsilon=self.epsilon, 
            delta=self.delta
        )
        
        # 2. Verificar se é uma nova melhor solução
        is_improvement = objective < self.best_objective_so_far
        
        if is_improvement:
            self.best_objective_so_far = objective
            self.best_solution_so_far = log_data.copy()
            self.best_weights_so_far = np.array(weights).copy()
        
        # 3. Log Completo: Salva TODAS as avaliações (em Parquet otimizado)
        #    Isso garante diversidade Pareto sem explodir o espaço
        if self.logger is not None:
            # ⭐⭐⭐ HOT LOOP OTIMIZAÇÃO: Use log_fast() se disponível
            # Isso elimina a alocação de dicionário (dict overhead na inner loop)
            # = 80% redução de GC thrashing
            
            if hasattr(self.logger, 'log_fast'):
                # ⭐ Novo: Sem criar dicionário intermediário!
                # Argumentos separados passados diretamente para NumPy pré-alocado
                self.logger.log_fast(
                    eval_id=self.eval_count,
                    weights=log_data.get('weights', []),
                    selected_assets=log_data.get('selected_assets', []),
                    expected_return=float(log_data.get('expected_return', 0.0)),
                    risk=float(log_data.get('risk', 0.0)),
                    variance=float(log_data.get('variance', 0.0)),
                    objective=float(objective),
                    is_improvement=bool(is_improvement),
                    timestamp=log_data.get('timestamp', datetime.now().isoformat())
                )
            else:
                # Fallback para código antigo (compatibilidade)
                record = {
                    "eval_id": self.eval_count,
                    "weights": log_data.get('weights', []),
                    "selected_assets": log_data.get('selected_assets', []),
                    "expected_return": float(log_data.get('expected_return', 0.0)),
                    "risk": float(log_data.get('risk', 0.0)),
                    "variance": float(log_data.get('variance', 0.0)),
                    "objective": float(objective),
                    "is_improvement": bool(is_improvement),
                    "timestamp": log_data.get('timestamp', datetime.now().isoformat())
                }
                self.logger.add_record(record)
        
        return objective
    
    def finalize(self):
        """
        Chamado ao final da otimização para garantir que tudo foi salvo.
        """
        if self.logger:
            self.logger.flush()
            file_size_mb = self.logger.get_file_size_mb()
            print(f"[STATEFUL] Finalized: {self.eval_count} avaliações salvas em {file_size_mb:.2f} MB")
    
    def get_stats(self):
        """Retorna estatísticas da execução."""
        return {
            'total_evaluations': self.eval_count,
            'best_objective': self.best_objective_so_far,
            'best_solution': self.best_solution_so_far,
            'best_weights': self.best_weights_so_far,
            'log_file_size_mb': self.logger.get_file_size_mb() if self.logger else None
        }


def _repair_weights(weights, epsilon, delta, k, n):
    # usando asarray pra evitar copia desnecessaria se ja for numpy array
    weights = np.asarray(weights)
    epsilon = np.asarray(epsilon)
    delta = np.asarray(delta)
    
    # Verificar se os arrays têm o tamanho correto
    if len(weights) != n:
        raise ValueError(f"Array weights deve ter tamanho {n}, mas tem tamanho {len(weights)}")
    if len(epsilon) != n:
        raise ValueError(f"Array epsilon deve ter tamanho {n}, mas tem tamanho {len(epsilon)}")
    if len(delta) != n:
        raise ValueError(f"Array delta deve ter tamanho {n}, mas tem tamanho {len(delta)}")
    
    # Verificar se k é válido APENAS se há restrição de cardinalidade  
    if k > n:
        raise ValueError(f"k ({k}) não pode ser maior que n_assets ({n})")
    # Só verifica k <= 0 se k foi especificado (não é None)
    if k is not None and k <= 0:
        raise ValueError(f"k ({k}) deve ser positivo quando especificado")
        
    # Verificar se epsilon <= delta para todos os ativos
    if np.any(epsilon > delta):
        raise ValueError("epsilon deve ser menor ou igual a delta para todos os ativos")
    
    selected_indices = np.argsort(weights)[-k:]
    selected_epsilon = epsilon[selected_indices]
    selected_delta = delta[selected_indices]
    if np.sum(selected_epsilon) > 1.0:
        return None, selected_indices, True
    final_k_weights = selected_epsilon.copy()
    # usando set pra operacoes O(1) ao inves de list O(k)
    free_assets_map = set(range(k))
    
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
            
        # converte set pra lista so quando necessario pra indexacao numpy
        free_list = list(free_assets_map)
        s_i_free = weights[selected_indices[free_list]]
        sum_s_i_free = np.sum(s_i_free)
        if sum_s_i_free > 1e-9:
            distribution = free_capital * (s_i_free / sum_s_i_free)
            final_k_weights[free_list] += distribution
        violating_assets_map = [idx for idx in free_assets_map if final_k_weights[idx] > selected_delta[idx]]
        if not violating_assets_map:
            break
        else:
            for idx in violating_assets_map:
                final_k_weights[idx] = selected_delta[idx]
                free_assets_map.discard(idx)  # O(1) com set, era O(k) com list
    final_weights = np.zeros(n)
    final_weights[selected_indices] = final_k_weights
    return final_weights, selected_indices, False

def _calc_metrics(final_weights, returns, cov, lambda_):
    expected_return = np.dot(final_weights, returns)
    # forma quadratica otimizada: evita array temporario intermediario
    variance = final_weights @ cov @ final_weights
    objective = lambda_ * variance - (1 - lambda_) * expected_return
    risk = np.sqrt(variance)
    return expected_return, variance, objective, risk

def portfolio_evaluation(weights, instance_data, logger=None, lambda_=0.5, k=None, epsilon=None, delta=None):
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
    
    # converte pra array (asarray nao copia se ja for array)
    epsilon = np.asarray(epsilon)
    delta = np.asarray(delta)
    
    # validacoes de epsilon/delta removidas - ja foram validadas no configure_problem
    # só valida weights que muda a cada avaliacao
    if np.any(np.isnan(weights)):
        return 1e7, {"error": "valores NaN detectados em weights"}
    
    if np.any(np.isinf(weights)):
        return 1e7, {"error": "valores infinitos detectados em weights"}
    
    is_constrained = k is not None and (k < n and k > 0)
    if not is_constrained:
        k = n
    
    # Validar k - só verifica se positivo quando há restrição 
    if k > n:
        return 1e7, {"error": f"k ({k}) não pode ser maior que n_assets ({n})"}
    
    try:
        final_weights, _, infeasible = _repair_weights(weights, epsilon, delta, k, n)
        if infeasible or final_weights is None:
            return 1e7, {"error": "Portfólio inviável: soma dos pisos > 1"}
    except Exception as e:
        return 1e7, {"error": f"Erro em _repair_weights: {str(e)}"}
        
    expected_return, variance, objective, risk = _calc_metrics(final_weights, returns, cov, lambda_)
    final_objective = objective
    
    execution_log = {
        "weights": final_weights.copy().tolist(),
        "selected_assets": np.nonzero(final_weights > 0)[0].tolist(),
        "expected_return": float(expected_return),
        "risk": float(risk),
        "variance": float(variance),
        "objective": float(final_objective),
        "timestamp": datetime.now().isoformat()
    }
    
    # Ignora soluções onde tanto risco quanto retorno são 0 no log
    if logger and not (risk == 0 or expected_return == 0):
        logger.log(execution_log)
    
    return objective, execution_log

def configure_problem(instance_data, k=None, risk_free_rate=0.03, lambda_=0.5, logger=None):
    """
    Configura o problema de otimização usando a classe StatefulPortfolioEvaluator.
    
    O logger pode ser:
    - PortfolioLogger (compatibilidade com antigo, salva em CSV)
    - ParquetBufferWriter (novo, salva em Parquet otimizado)
    - None (sem logging)
    
    Args:
        instance_data: Dados da instância
        k: Restrição de cardinalidade (None = sem restrição)
        risk_free_rate: Taxa livre de risco
        lambda_: Parâmetro de trade-off retorno/risco
        logger: Logger para salvar avaliações (PortfolioLogger, ParquetBufferWriter, ou None)
        
    Returns:
        dict com:
            - 'function': Avaliador com estado (callable)
            - 'is_constrained': True
            - 'boundaries': Limites dos pesos
            - 'evaluator': Referência ao avaliador (para finalization)
    """
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
        raise ValueError(f"k ({k}) deve ser positivo quando especificado (use k=None para sem restrição)")
    
    # Instanciar o avaliador com estado
    evaluator = StatefulPortfolioEvaluator(
        instance_data,
        logger=logger,
        lambda_=lambda_,
        k=k,
        epsilon=epsilon,
        delta=delta
    )
    
    return {
        "function": evaluator,  # O CUSTOMHYS vai chamar evaluator(weights)
        "is_constrained": True,
        "boundaries": (lower_bounds, upper_bounds),
        "evaluator": evaluator  # Referência para finalization
    }