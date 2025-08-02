# portfolio_utils/milestone_logger.py

import pandas as pd
import os

class MilestoneLogger:
    """
    Logger baseado em marcos com intervalos progressivos
    Salva com frequência decrescente: 1, 10, 100, 1000, 10000...
    """
    def __init__(self, log_file_path, buffer_size=1000, initial_interval=1):
        self.log_file_path = log_file_path
        self.buffer_size = buffer_size
        self.execution_logs = []
        self._header_written = os.path.exists(self.log_file_path)
        
        # Configuração dos marcos
        self.evaluation_counter = 0
        self.milestone_intervals = [1, 10, 100, 1000, 10000, 100000]
        self.current_milestone_index = 0
        self.current_interval = initial_interval
        self.next_log_at = initial_interval
        
        # Tracking de melhorias
        self.best_objective = float('inf')
        self.improvements_saved = 0
        self.milestone_logs_saved = 0

    def _should_log_milestone(self):
        """Verifica se deve salvar baseado no marco atual"""
        return self.evaluation_counter == self.next_log_at

    def _update_milestone(self):
        """Atualiza o próximo marco quando apropriado"""
        if self._should_log_milestone():
            # Calcula próximo marco
            self.next_log_at += self.current_interval
            
            # Verifica se deve aumentar o intervalo
            milestone_threshold = self.current_interval * 10
            if (self.evaluation_counter >= milestone_threshold and 
                self.current_milestone_index < len(self.milestone_intervals) - 1):
                
                self.current_milestone_index += 1
                self.current_interval = self.milestone_intervals[self.current_milestone_index]
                
                # Ajusta próximo marco para o novo intervalo
                next_multiple = ((self.evaluation_counter // self.current_interval) + 1) * self.current_interval
                self.next_log_at = next_multiple

    def log(self, log_data):
        """Salva o log se for uma melhoria ou um marco"""
        self.evaluation_counter += 1
        objective = log_data.get('objective', float('inf'))
        
        # Sempre salva melhorias
        is_improvement = objective < self.best_objective
        if is_improvement:
            self.best_objective = objective
            self.improvements_saved += 1
            
        # Verifica se é um marco
        is_milestone = self._should_log_milestone()
        if is_milestone:
            self.milestone_logs_saved += 1
            
        # Salva se é melhoria ou marco
        if is_improvement or is_milestone:
            # Adiciona informações de contexto
            log_data_with_context = log_data.copy()
            log_data_with_context['evaluation_number'] = self.evaluation_counter
            log_data_with_context['is_improvement'] = is_improvement
            log_data_with_context['is_milestone'] = is_milestone
            log_data_with_context['current_interval'] = self.current_interval
            
            self.execution_logs.append(log_data_with_context)
            
            if len(self.execution_logs) >= self.buffer_size:
                self.flush()
                
        # Atualiza marcos após verificação
        if is_milestone:
            self._update_milestone()

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
        total_logs = self.improvements_saved + self.milestone_logs_saved
        reduction_rate = (1 - total_logs / max(self.evaluation_counter, 1)) * 100
        
        print("📊 Milestone Logger Statistics:")
        print(f"   Total evaluations: {self.evaluation_counter:,}")
        print(f"   Improvements saved: {self.improvements_saved:,}")
        print(f"   Milestone logs saved: {self.milestone_logs_saved:,}")
        print(f"   Total logs saved: {total_logs:,}")
        print(f"   Reduction rate: {reduction_rate:.1f}%")
        print(f"   Best objective: {self.best_objective:.6f}")
        print(f"   Final interval: {self.current_interval:,}")
        print(f"   Next milestone at: {self.next_log_at:,}")

    def get_stats(self):
        """Retorna estatísticas atuais"""
        return {
            "total_evaluations": self.evaluation_counter,
            "improvements_saved": self.improvements_saved,
            "milestone_logs_saved": self.milestone_logs_saved,
            "current_interval": self.current_interval,
            "next_milestone": self.next_log_at,
            "best_objective": self.best_objective
        }

# Exemplo de uso:
# from portfolio_utils.milestone_logger import MilestoneLogger
# logger = MilestoneLogger(log_file_path, initial_interval=1)
