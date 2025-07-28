# Em portfolio_utils/portfolio_logger.py
import pandas as pd
import os

class PortfolioLogger:
    """
    Gerencia o log de execuções de forma eficiente, salvando os dados
    em disco em lotes (buffers) para economizar memória RAM.
    """
    def __init__(self, log_file_path, buffer_size=1000):
        self.log_file_path = log_file_path
        self.buffer_size = buffer_size
        self.execution_logs = []
        # Verifica se o arquivo já existe para decidir se escreve o cabeçalho
        self._header_written = os.path.exists(self.log_file_path)

    def log(self, log_data):
        """Adiciona um registro de log ao buffer e o esvazia se estiver cheio."""
        self.execution_logs.append(log_data)
        if len(self.execution_logs) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Escreve o conteúdo do buffer no arquivo CSV e o limpa."""
        if not self.execution_logs:
            return  # Não faz nada se o buffer estiver vazio

        df = pd.DataFrame(self.execution_logs)
        
        if self._header_written:
            # Se o arquivo já existe, adiciona sem o cabeçalho
            df.to_csv(self.log_file_path, mode='a', header=False, index=False)
        else:
            # Se é a primeira vez, cria o arquivo com o cabeçalho
            df.to_csv(self.log_file_path, mode='w', header=True, index=False)
            self._header_written = True # Marca que o cabeçalho já foi escrito

        self.execution_logs.clear() # Limpa o buffer da memória

    def close(self):
        """Garante que todos os logs restantes no buffer sejam salvos antes de fechar."""
        self.flush()