#!/usr/bin/env python3
"""
Exemplo de uso do portfolio_optimizer.py
"""

import subprocess
import sys

def run_example():
    """
    Executa um exemplo usando port1.txt com cardinalidade 3
    """
    print("🔹 Executando exemplo: port1.txt com cardinalidade 3")
    print("🔹 Comando: python portfolio_optimizer.py port1.txt 3 config_hh_default.txt")
    print("-" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, 
            "portfolio_optimizer.py", 
            "port1.txt", 
            "3", 
            "config_hh_default.txt"
        ], capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nCódigo de retorno: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout: execução demorou mais de 5 minutos")
    except Exception as e:
        print(f"❌ Erro ao executar: {e}")

if __name__ == "__main__":
    run_example()
