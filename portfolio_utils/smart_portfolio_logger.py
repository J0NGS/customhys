# portfolio_utils/smart_portfolio_logger.py
import pandas as pd
import os
import numpy as np
from collections import deque

class SmartPortfolioLogger:
    """
    Logger inteligente que filtra soluções por qualidade para reduzir o volume de dados.
    """
    def __init__(self, log_file_path, buffer_size=1000, strategy="quality_filter"):
        self.log_file_path = log_file_path
        self.buffer_size = buffer_size
        self.strategy = strategy
        self.execution_logs = []
        self._header_written = os.path.exists(self.log_file_path)
        
        # Configurações para diferentes estratégias
        self._setup_strategy()
        
    def _setup_strategy(self):
        """Configura a estratégia de filtragem"""
        if self.strategy == "quality_filter":
            self.best_objective = float('inf')
            self.quality_threshold = 0.1  # Salva soluções até 10% piores que a melhor
            
        elif self.strategy == "top_percentile":
            self.top_solutions = deque(maxlen=10000)  # Mantém apenas as 10k melhores
            
        elif self.strategy == "sampling":
            self.sample_rate = 0.001  # Salva 0.1% das avaliações
            self.sample_counter = 0
            
        elif self.strategy == "milestone_based":
            self.evaluation_counter = 0
            self.milestone_intervals = [1, 10, 100, 1000, 10000]  # Intervalos crescentes
            self.current_milestone = 0
            
        elif self.strategy == "pareto_frontier":
            self.pareto_solutions = []
            
    def should_log(self, log_data):
        """Decide se deve salvar o log baseado na estratégia"""
        
        if self.strategy == "quality_filter":
            objective = log_data.get('objective', float('inf'))
            if objective < self.best_objective:
                self.best_objective = objective
                return True
            # Salva se está dentro do threshold de qualidade
            return objective <= self.best_objective * (1 + self.quality_threshold)
            
        elif self.strategy == "top_percentile":
            objective = log_data.get('objective', float('inf'))
            if len(self.top_solutions) < self.top_solutions.maxlen:
                self.top_solutions.append((objective, log_data))
                return True
            else:
                worst_in_top = max(self.top_solutions, key=lambda x: x[0])
                if objective < worst_in_top[0]:
                    self.top_solutions.remove(worst_in_top)
                    self.top_solutions.append((objective, log_data))
                    return True
                return False
                
        elif self.strategy == "sampling":
            self.sample_counter += 1
            return np.random.random() < self.sample_rate
            
        elif self.strategy == "milestone_based":
            self.evaluation_counter += 1
            if self.current_milestone < len(self.milestone_intervals):
                interval = self.milestone_intervals[self.current_milestone]
                if self.evaluation_counter % interval == 0:
                    # Aumenta o intervalo progressivamente
                    if self.evaluation_counter >= interval * 10:
                        self.current_milestone = min(self.current_milestone + 1, 
                                                   len(self.milestone_intervals) - 1)
                    return True
            return False
            
        elif self.strategy == "pareto_frontier":
            return self._is_pareto_optimal(log_data)
            
        return True  # Fallback: salva tudo
        
    def _is_pareto_optimal(self, new_solution):
        """Verifica se a solução é Pareto-ótima"""
        new_return = new_solution.get('expected_return', 0)
        new_risk = new_solution.get('risk', float('inf'))
        
        # Remove soluções dominadas pela nova
        self.pareto_solutions = [sol for sol in self.pareto_solutions 
                               if not (new_return >= sol.get('expected_return', 0) and 
                                      new_risk <= sol.get('risk', float('inf')))]
        
        # Verifica se a nova solução é dominada
        for sol in self.pareto_solutions:
            if (sol.get('expected_return', 0) >= new_return and 
                sol.get('risk', float('inf')) <= new_risk):
                return False
                
        return True

    def log(self, log_data):
        """Adiciona um registro de log ao buffer se passar pelo filtro"""
        if self.should_log(log_data):
            self.execution_logs.append(log_data)
            if len(self.execution_logs) >= self.buffer_size:
                self.flush()

    def flush(self):
        """Escreve o conteúdo do buffer no arquivo CSV e o limpa."""
        if not self.execution_logs:
            return

        df = pd.DataFrame(self.execution_logs)
        
        if self._header_written:
            df.to_csv(self.log_file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.log_file_path, mode='w', header=True, index=False)
            self._header_written = True

        self.execution_logs.clear()

    def close(self):
        """Garante que todos os logs restantes sejam salvos"""
        self.flush()
        
        # Se usando top_percentile, salva as melhores soluções
        if self.strategy == "top_percentile" and self.top_solutions:
            final_logs = [sol[1] for sol in sorted(self.top_solutions, key=lambda x: x[0])]
            df = pd.DataFrame(final_logs)
            df.to_csv(self.log_file_path.replace('.csv', '_top_solutions.csv'), 
                     mode='w', header=True, index=False)
        
    def get_stats(self):
        """Retorna estatísticas do logger"""
        return {
            "strategy": self.strategy,
            "logs_saved": len(self.execution_logs),
            "best_objective": getattr(self, 'best_objective', None),
            "evaluation_counter": getattr(self, 'evaluation_counter', 0)
        }
