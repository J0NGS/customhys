# portfolio_utils/simple_sampling_logger.py

import pandas as pd
import os
import random

class SamplingLogger:
    """
    Logger simples que salva apenas uma fração das avaliações
    """
    def __init__(self, log_file_path, sample_rate=0.001, buffer_size=1000):
        """
        Args:
            sample_rate: Fração das avaliações a salvar (0.001 = 0.1%)
        """
        self.log_file_path = log_file_path
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.execution_logs = []
        self._header_written = os.path.exists(self.log_file_path)
        self.best_objective = float('inf')
        self.total_evaluations = 0

    def log(self, log_data):
        """Salva o log apenas se passou no filtro de amostragem ou é uma melhoria"""
        self.total_evaluations += 1
        objective = log_data.get('objective', float('inf'))
        
        # Sempre salva se é uma melhoria
        is_improvement = objective < self.best_objective
        if is_improvement:
            self.best_objective = objective
            
        # Ou salva com base na taxa de amostragem
        should_sample = random.random() < self.sample_rate
        
        if is_improvement or should_sample:
            self.execution_logs.append(log_data)
            if len(self.execution_logs) >= self.buffer_size:
                self.flush()

    def flush(self):
        """Escreve o buffer no arquivo"""
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
        """Finaliza o logger e mostra estatísticas"""
        self.flush()
        print(f"📊 Logger Statistics:")
        print(f"   Total evaluations: {self.total_evaluations:,}")
        print(f"   Logs saved: {len(self.execution_logs):,}")
        print(f"   Reduction rate: {(1 - len(self.execution_logs)/max(self.total_evaluations, 1))*100:.1f}%")
        print(f"   Best objective: {self.best_objective:.6f}")

# Exemplo de uso:
# from portfolio_utils.simple_sampling_logger import SamplingLogger
# logger = SamplingLogger(log_file_path, sample_rate=0.001)  # 0.1% das avaliações
