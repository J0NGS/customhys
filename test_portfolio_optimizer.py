#!/usr/bin/env python3
"""
Scripts de teste para portfolio_optimizer.py
Demonstra diferentes cenários de uso
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description):
    """Executa um comando e exibe o resultado"""
    print(f"\n{'='*60}")
    print(f"🔹 {description}")
    print(f"🔹 Comando: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        end_time = datetime.now()
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        duration = (end_time - start_time).total_seconds()
        print(f"\n⏱️  Tempo de execução: {duration:.2f} segundos")
        print(f"🔍 Código de retorno: {result.returncode}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout: execução demorou mais de 2 minutos")
        return False
    except Exception as e:
        print(f"❌ Erro ao executar: {e}")
        return False

def test_scenarios():
    """Executa diferentes cenários de teste"""
    scenarios = [
        {
            "description": "Teste 1: port1.txt sem restrição de cardinalidade (configuração rápida)",
            "cmd": [sys.executable, "portfolio_optimizer.py", "port1.txt", "0", "config_hh_fast.txt"]
        },
        {
            "description": "Teste 2: port1.txt com 3 ativos (configuração rápida)",
            "cmd": [sys.executable, "portfolio_optimizer.py", "port1.txt", "3", "config_hh_fast.txt"]
        },
        {
            "description": "Teste 3: port2.txt com 5 ativos (configuração rápida)",
            "cmd": [sys.executable, "portfolio_optimizer.py", "port2.txt", "5", "config_hh_fast.txt"]
        }
    ]
    
    print("🚀 Iniciando testes do Portfolio Optimizer")
    print(f"📅 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success_count = 0
    
    for i, scenario in enumerate(scenarios, 1):
        if run_command(scenario["cmd"], scenario["description"]):
            success_count += 1
            print("✅ Teste executado com sucesso!")
        else:
            print("❌ Teste falhou!")
    
    print(f"\n{'='*60}")
    print(f"📊 RESUMO DOS TESTES:")
    print(f"   - Total de testes: {len(scenarios)}")
    print(f"   - Sucessos: {success_count}")
    print(f"   - Falhas: {len(scenarios) - success_count}")
    print(f"   - Taxa de sucesso: {(success_count/len(scenarios)*100):.1f}%")
    
    # Lista os diretórios criados
    print(f"\n📁 Diretórios de resultados criados:")
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("2025"):
            print(f"   - {item}")

def test_help():
    """Testa o help do programa"""
    print("🔹 Testando help do programa...")
    cmd = [sys.executable, "portfolio_optimizer.py", "--help"]
    run_command(cmd, "Teste do Help")

if __name__ == "__main__":
    print("🔧 Portfolio Optimizer - Bateria de Testes")
    
    # Verifica se os arquivos necessários existem
    required_files = ["portfolio_optimizer.py", "config_hh_fast.txt", "port1.txt", "port2.txt"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Arquivos faltando: {missing_files}")
        sys.exit(1)
    
    # Executa o help primeiro
    test_help()
    
    # Executa os cenários de teste
    test_scenarios()
    
    print("\n🎉 Bateria de testes finalizada!")
