#!/usr/bin/env python3
"""
Portfolio Optimization usando CustomHyS
Baseado no notebook portfolio_optimization_OR_LIBRARY.ipynb

Uso:
    python portfolio_optimizer.py <arquivo_instancia> <cardinalidade> <arquivo_config>

Exemplo:
    python portfolio_optimizer.py port1.txt 5 config_hh.txt
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from customhys.hyperheuristic import Hyperheuristic

# Lista global para registrar os resultados de cada avaliação da função
execution_logs = []
BUFFER_SIZE = 1000  # ### MOD: número de logs em memória antes de salvar
LOG_FILE_PATH = None  # ### MOD: caminho do arquivo de log incremental
def read_or_library_instance(filepath):
    """
    Lê um arquivo de instância da OR-Library (ex: port1.txt) e retorna:
    - número de ativos
    - retornos esperados (μ_i)
    - desvios padrão (σ_i)
    - matriz de correlação (ρ_{i,j})
    - matriz de covariância (Σ)
    - dataframe da covariância para visualização
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo {filepath} não encontrado")

    # Abre e lê todas as linhas do arquivo
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Número total de ativos financeiros disponíveis na instância
    n_assets = int(lines[0].strip())

    # Lê os retornos esperados e os desvios padrão (μ_i e σ_i)
    returns = []
    std_devs = []
    for i in range(1, n_assets + 1):
        mu, sigma = map(float, lines[i].strip().split())
        returns.append(mu)
        std_devs.append(sigma)

    # Inicializa a matriz de correlação com 1's na diagonal principal
    corr_matrix = np.eye(n_assets)

    # Lê os pares de ativos e suas respectivas correlações ρ_{i,j}
    for line in lines[n_assets + 1:]:
        if line.strip():  # Ignora linhas vazias
            i, j, rho = line.strip().split()
            i, j = int(i) - 1, int(j) - 1  # Corrige índice (de 1-based para 0-based)
            rho = float(rho)
            corr_matrix[i][j] = rho
            corr_matrix[j][i] = rho  # Simetria da matriz

    # Constrói a matriz de covariância Σ com:
    # Σ_{i,j} = ρ_{i,j} * σ_i * σ_j
    sigma_array = np.array(std_devs)
    cov_matrix = np.outer(sigma_array, sigma_array) * corr_matrix

    # Cria um DataFrame para visualização fácil da matriz de covariância
    df_cov = pd.DataFrame(
        cov_matrix,
        columns=[f"Asset {i+1}" for i in range(n_assets)],
        index=[f"Asset {i+1}" for i in range(n_assets)]
    )

    # Retorna um dicionário com todos os dados necessários
    return {
        "n_assets": n_assets,
        "returns": np.array(returns),         # Vetor de retornos esperados μ_i
        "std_devs": np.array(std_devs),       # Vetor de desvios padrão σ_i
        "corr_matrix": corr_matrix,           # Matriz de correlação ρ_{i,j}
        "cov_matrix": cov_matrix,             # Matriz de covariância Σ
        "df_cov": df_cov                      # DataFrame para visualização
    }

def portfolio_evaluation(weights, instance_data, lambda_=0.5, k=None, risk_free_rate=0.03, epsilon=None, delta=None):
    """
    Função de avaliação do portfólio baseada no modelo de Chang et al. (2000)
    
    Args:
        weights: vetor de pesos do portfólio
        instance_data: dados da instância (retornos, covariância, etc.)
        lambda_: parâmetro de ponderação entre risco e retorno (0 a 1)
        k: restrição de cardinalidade (None = sem restrição)
        risk_free_rate: taxa livre de risco
    
    Returns:
        tuple: (objetivo, execution_log)
    """
    n = instance_data["n_assets"]
    returns = instance_data["returns"]
    cov = instance_data["cov_matrix"]
    weights = np.array(weights)

    # Novos parâmetros: epsilon (piso), delta (teto)
    if epsilon is None:
        epsilon = instance_data.get("epsilon", np.zeros(n))
    if delta is None:
        delta = instance_data.get("delta", np.ones(n))

    # Se k não for especificado, o portfólio não tem restrição de cardinalidade (k = n)
    is_constrained = k is not None and k < n
    if not is_constrained:
        k = n
    # Algoritmo de reparo para restrições de piso/teto e cardinalidade
    selected_indices = np.argsort(weights)[-k:]
    selected_epsilon = epsilon[selected_indices]
    selected_delta = delta[selected_indices]
    if np.sum(selected_epsilon) > 1.0:
        return 1e7, {"error": "Portfólio inviável: soma dos pisos > 1"}

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

    expected_return = np.dot(final_weights, returns)
    variance = np.dot(final_weights, np.dot(cov, final_weights))
    objective = lambda_ * variance - (1 - lambda_) * expected_return
    final_objective = objective
    risk = np.sqrt(variance)
    sharpe = (expected_return - risk_free_rate) / risk if risk > 0 else -1e6
    execution_log = {
        "execution_number": len(execution_logs) + 1,
        "weights": final_weights.copy().tolist(),
        "selected_assets": selected_indices.tolist(),
        "expected_return": float(expected_return),
        "risk": float(risk),
        "sharpe": float(sharpe),
        "variance": float(variance),
        "objective": float(final_objective),
        "timestamp": datetime.now().isoformat()
    }
    # ### MOD: salvar em disco quando atingir BUFFER_SIZE
    execution_logs.append(execution_log)
    if len(execution_logs) >= BUFFER_SIZE and LOG_FILE_PATH is not None:
        df = pd.DataFrame(execution_logs)
        if os.path.exists(LOG_FILE_PATH):
            df.to_csv(LOG_FILE_PATH, mode='a', header=False, index=False)
        else:
            df.to_csv(LOG_FILE_PATH, mode='w', header=True, index=False)
        execution_logs.clear()

    return objective, execution_log

def configure_problem(instance_data, k=None, risk_free_rate=0.03):
    """
    Configura o problema no formato aceito pelo CUSTOMHyS.
    
    Parâmetros:
    - instance_data: dicionário retornado por read_or_library_instance()
    - k: restrição de cardinalidade (número máximo de ativos)
    - risk_free_rate: taxa livre de risco

    Retorna:
    - dicionário com as chaves esperadas por CUSTOMHyS
    """
    n = instance_data["n_assets"]
    lower_bounds = [0.00] * n
    upper_bounds = [1.00] * n
    # Permite passar epsilon/delta via instance_data
    epsilon = instance_data.get("epsilon", np.zeros(n))
    delta = instance_data.get("delta", np.ones(n))
    return {
        "function": lambda weights: portfolio_evaluation(weights, instance_data, lambda_=0.5, k=k, risk_free_rate=risk_free_rate, epsilon=epsilon, delta=delta)[0],
        "is_constrained": True,
        "boundaries": (lower_bounds, upper_bounds),
    }

def load_hh_config(config_file):
    """
    Carrega a configuração da hyper-heurística de um arquivo texto.
    O arquivo deve conter os parâmetros em formato JSON.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Arquivo de configuração {config_file} não encontrado")
    
    with open(config_file, 'r') as f:
        # Remove comentários que começam com #
        lines = []
        for line in f:
            # Remove comentários no final da linha
            line = line.split('#')[0].strip()
            if line:
                lines.append(line)
        
        # Junta as linhas e converte para JSON
        config_text = '\n'.join(lines)
        try:
            config = json.loads(config_text)
            return config
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON do arquivo {config_file}: {e}")
            return None

def create_output_directory(instance_file, initial_scheme):
    """
    Cria um diretório com nome padrão data-hora-nomeDoArquivoTestado-MetodoInicialPopulação
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_name = os.path.splitext(os.path.basename(instance_file))[0]
    dir_name = f"{timestamp}_{instance_name}_{initial_scheme}"
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    return dir_name

def save_logs(output_dir, execution_logs, instance_data, hh_config, result):
    """
    Salva os logs e resultados em arquivos
    """
    # Salva configuração da HH
    with open(os.path.join(output_dir, "hh_config.json"), 'w') as f:
        json.dump(hh_config, f, indent=2)
    
    # Salva dados da instância
    instance_summary = {
        "n_assets": instance_data["n_assets"],
        "returns": instance_data["returns"].tolist(),
        "std_devs": instance_data["std_devs"].tolist(),
        "cov_matrix": instance_data["cov_matrix"].tolist()
    }
    with open(os.path.join(output_dir, "instance_data.json"), 'w') as f:
        json.dump(instance_summary, f, indent=2)
    
    # Salva resultado final da HH
    with open(os.path.join(output_dir, "hh_result.json"), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # Estatísticas resumidas
    if execution_logs:
        best_solution = min(execution_logs, key=lambda x: x.get('objective', float('inf')))
        stats = {
            "total_evaluations": len(execution_logs),
            "best_objective": best_solution.get('objective'),
            "best_sharpe": best_solution.get('sharpe'),
            "best_return": best_solution.get('expected_return'),
            "best_risk": best_solution.get('risk'),
            "best_weights": best_solution.get('weights'),
            "best_selected_assets": best_solution.get('selected_assets')
        }
        with open(os.path.join(output_dir, "summary_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Portfolio Optimization usando CustomHyS')
    parser.add_argument('instance_file', help='Arquivo da instância (ex: port1.txt)')
    parser.add_argument('cardinality', type=int, help='Restrição de cardinalidade (0 = sem restrição)')
    parser.add_argument('config_file', help='Arquivo de configuração da hyper-heurística')
    parser.add_argument('--epsilon', type=str, default=None, help='Arquivo ou lista separada por vírgula com os valores de piso (ex: "0.01,0.01,0.01" ou "epsilon.txt")')
    parser.add_argument('--delta', type=str, default=None, help='Arquivo ou lista separada por vírgula com os valores de teto (ex: "0.2,0.2,0.2" ou "delta.txt")')
    args = parser.parse_args()
    
    print(f"🔍 Carregando instância: {args.instance_file}")
    
    # Carrega os dados da instância
    try:
        instance_data = read_or_library_instance(args.instance_file)
        print(f"✅ Instância carregada com sucesso!")
        print(f"   - Número de ativos: {instance_data['n_assets']}")
        
        # Mostra os primeiros 5 e últimos 5 ativos
        n_assets = instance_data['n_assets']
        print(f"\n📊 Primeiros 5 ativos:")
        for i in range(min(5, n_assets)):
            print(f"   Asset {i+1}: μ={instance_data['returns'][i]:.4f}, σ={instance_data['std_devs'][i]:.4f}")
        
        if n_assets > 5:
            print(f"\n📊 Últimos 5 ativos:")
            for i in range(max(0, n_assets-5), n_assets):
                print(f"   Asset {i+1}: μ={instance_data['returns'][i]:.4f}, σ={instance_data['std_devs'][i]:.4f}")
                
    except Exception as e:
        print(f"❌ Erro ao carregar instância: {e}")
        sys.exit(1)
    
    # Configura restrição de cardinalidade
    k = None if args.cardinality == 0 else args.cardinality
    if k is not None:
        if k > instance_data['n_assets']:
            print(f"⚠️  Cardinalidade {k} maior que o número de ativos {instance_data['n_assets']}. Usando sem restrição.")
            k = None
        else:
            print(f"🎯 Restrição de cardinalidade: {k} ativos")
    else:
        print("🎯 Sem restrição de cardinalidade")
    
    # Carrega configuração da HH
    print(f"\n🔧 Carregando configuração: {args.config_file}")
    try:
        hh_config = load_hh_config(args.config_file)
        if hh_config is None:
            print("❌ Erro ao carregar configuração da hyper-heurística")
            sys.exit(1)
        print("✅ Configuração carregada com sucesso!")
        print(f"   - Iterações: {hh_config.get('num_iterations', 'N/A')}")
        print(f"   - Agentes: {hh_config.get('num_agents', 'N/A')}")
        print(f"   - Solver: {hh_config.get('solver', 'N/A')}")
        print(f"   - Esquema inicial: {hh_config.get('initial_scheme', 'N/A')}")
    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        sys.exit(1)
    
    # Configura o problema
    print(f"\n⚙️  Configurando problema...")
    problem_config = configure_problem(instance_data, k=k, risk_free_rate=0.03)
    print("✅ Problema configurado!")
    
    # Cria diretório de saída
    output_dir = create_output_directory(args.instance_file, hh_config.get('initial_scheme', 'unknown'))
    print(f"📁 Diretório de saída: {output_dir}")
    global LOG_FILE_PATH  # para uso no portfolio_evaluation
    LOG_FILE_PATH = os.path.join(output_dir, "execution_logs.csv")

    
    # Limpa logs antigos
    execution_logs.clear()
    
    # Executa a hyper-heurística
    print(f"\n🚀 Iniciando execução da Hyper-Heurística...")
    try:
        hh = Hyperheuristic(heuristic_space='default.txt', problem=problem_config, parameters=hh_config)
        result = hh.solve()
        print(f"✅ Execução concluída!")
        print(f"   - Total de avaliações: {len(execution_logs)}")
        
        if execution_logs:
            best_solution = min(execution_logs, key=lambda x: x.get('objective', float('inf')))
            print(f"   - Melhor objetivo: {best_solution.get('objective', 'N/A'):.6f}")
            print(f"   - Melhor Sharpe: {best_solution.get('sharpe', 'N/A'):.4f}")
            print(f"   - Retorno: {best_solution.get('expected_return', 'N/A'):.4f}")
            print(f"   - Risco: {best_solution.get('risk', 'N/A'):.4f}")
            print(f"   - Ativos selecionados: {len(best_solution.get('selected_assets', []))}")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        result = {"error": str(e)}
    
    # Salva logs e resultados
    print(f"\n💾 Salvando resultados...")
    try:
        if execution_logs and LOG_FILE_PATH is not None:
            df = pd.DataFrame(execution_logs)
            if os.path.exists(LOG_FILE_PATH):
                df.to_csv(LOG_FILE_PATH, mode='a', header=False, index=False)
            else:
                df.to_csv(LOG_FILE_PATH, mode='w', header=True, index=False)
            execution_logs.clear()
        save_logs(output_dir, execution_logs, instance_data, hh_config, result)
        print(f"✅ Resultados salvos em: {output_dir}")
        print(f"   - execution_logs.csv: Log de todas as avaliações")
        print(f"   - hh_config.json: Configuração da hyper-heurística")
        print(f"   - instance_data.json: Dados da instância")
        print(f"   - hh_result.json: Resultado final da HH")
        print(f"   - summary_stats.json: Estatísticas resumidas")
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}")
    
    print(f"\n🎉 Execução finalizada!")

if __name__ == "__main__":
    main()
