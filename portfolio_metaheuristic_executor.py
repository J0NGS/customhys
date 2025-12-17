#!/usr/bin/env python3
"""
Portfolio Metaheuristic Executor
Executa uma metaheurística montada a partir dos operadores selecionados pelo CustomHyS
"""

import os
import sys
import json
import argparse
import pandas as pd
from functools import partial
from datetime import datetime

from customhys.metaheuristic import Metaheuristic

from portfolio_utils.instance_reader import read_or_library_instance
from portfolio_utils.portfolio_logger import PortfolioLogger
from portfolio_utils.portfolio_evaluator import portfolio_evaluation, configure_problem
from portfolio_utils.config_utils import create_output_directory


def parse_operator_indices(indices_str):
    """
    Parseia a string de índices do hh_result.json e retorna uma lista de inteiros.
    Ex: "[ 98  55  53 120  36 138   1  40  59]" -> [98, 55, 53, 120, 36, 138, 1, 40, 59]
    """
    # Remove colchetes e espaços extras, depois converte para inteiros
    indices_str = indices_str.strip().strip('[]')
    indices = [int(x) for x in indices_str.split()]
    return indices


def load_operators_from_collection(collection_file, indices):
    """
    Carrega os operadores do arquivo de coleção usando os índices fornecidos.
    Retorna uma lista de tuplas (operator_name, params, selector).
    """
    # ler o arquivo
    with open(collection_file, 'r') as f:
        lines = f.read().strip().split('\n')
    
    # converter cada linha em tupla
    all_operators = [eval(line) for line in lines if line.strip()]
    
    # extrair apenas os operadores especificados
    selected_operators = [all_operators[idx] for idx in indices if idx < len(all_operators)]
    
    return selected_operators


def build_metaheuristic_sequence(selected_operators):
    """
    Constroi a sequencia de operadores no formato esperado pelo módulo Metaheuristic.
    Cada operador já vem na forma (perturbador, params, seletor) do arquivo txt.
    """
    heur_sequence = []
    
    for operator_tuple in selected_operators:
        perturbador, params, seletor = operator_tuple
        heur_sequence.append((perturbador, params, seletor))
    
    return heur_sequence


def main():
    parser = argparse.ArgumentParser(
        description='Execute uma Metaheurística com operadores selecionados pelo CustomHyS'
    )
    parser.add_argument(
        'hh_result_file',
        help='arquivo hh_result.json gerado pelo portfolio_optimizer'
    )
    parser.add_argument(
        'instance_file',
        help='arquivo da instância (ex: port1.txt)'
    )
    parser.add_argument(
        'cardinality',
        type=int,
        help='restrição de cardinalidade (0 = sem restrição)'
    )
    parser.add_argument(
        'collection_file',
        help='coleção de operadores (ex: default_portfolio.txt)'
    )
    parser.add_argument(
        'num_iterations',
        type=int,
        help='iterações da metaheurística'
    )
    parser.add_argument(
        '--lambda',
        dest='lambda_param',
        type=float,
        default=0.5,
        help='lambda para trade-off risco-retorno (padrão: 0.5)'
    )
    parser.add_argument(
        '--no-logger',
        action='store_true',
        default=False,
        help='Desabilita a criação do logger'
    )
    
    args = parser.parse_args()
    
    # ===== VALIDAÇÕES INICIAIS =====
    if not os.path.exists(args.hh_result_file):
        print(f"[ERRO] Arquivo hh_result.json não encontrado: {args.hh_result_file}")
        sys.exit(1)
    
    if not os.path.exists(args.instance_file):
        print(f"[ERRO] Arquivo da instância não encontrado: {args.instance_file}")
        sys.exit(1)
    
    if not os.path.exists(args.collection_file):
        print(f"[ERRO] Arquivo de coleção não encontrado: {args.collection_file}")
        sys.exit(1)
    
    # ===== CARREGAR hh_result.json =====
    print("[LOADING] Carregando resultado da hiperheurística...")
    with open(args.hh_result_file, 'r') as f:
        hh_result = json.load(f)
    
    # hh_result[0] tem os índices dos operadores
    indices_str = hh_result[0]
    operator_indices = parse_operator_indices(indices_str)
    
    print(f"[INFO] Índices dos operadores selecionados: {operator_indices}")
    print(f"[INFO] Total de operadores: {len(operator_indices)}")
    
    # ===== CARREGAR OPERADORES =====
    print("\n[LOADING] Carregando operadores da coleção...")
    selected_operators = load_operators_from_collection(args.collection_file, operator_indices)
    
    print(f"[INFO] {len(selected_operators)} operadores carregados:")
    for i, (op_name, params, selector) in enumerate(selected_operators, 1):
        print(f"[INFO]   {i}. {op_name} | params={params} | selector={selector}")
    
    # ===== CONSTRUIR SEQUÊNCIA DA METAHEURÍSTICA =====
    print("\n[PROCESS] Construindo sequência da metaheurística...")
    heur_sequence = build_metaheuristic_sequence(selected_operators)
    print(f"[INFO] Sequência de {len(heur_sequence)} operadores construída!")
    
    # ===== CARREGAR DADOS DO PROBLEMA =====
    print("\n[PROCESS] Configurando problema de otimização...")
    instance_data = read_or_library_instance(args.instance_file)
    k = None if args.cardinality == 0 else args.cardinality
    lambda_param = args.lambda_param
    
    print(f"[INFO] Número de ativos: {instance_data['n_assets']}")
    print(f"[INFO] Cardinalidade: {'Sem restrição' if k is None else k}")
    print(f"[INFO] Parâmetro lambda: {lambda_param}")
    
    # ===== CONFIGURAR LOGGER =====
    print("\n[PROCESS] Configurando logger...")
    output_dir = create_output_directory(args.instance_file, 'metaheuristic_executor')
    
    log_file_path = os.path.join(output_dir, "execution_logs.csv")
    if args.no_logger:
        logger = None
        print("[INFO] Logger desabilitado por parâmetro CLI (--no-logger).")
    else:
        logger = PortfolioLogger(log_file_path=log_file_path, buffer_size=100000)
        print(f"[INFO] Logger criado em: {output_dir}")
    
    # ===== CONFIGURAR FUNÇÃO OBJETIVO =====
    print("\n[PROCESS] Configurando função objetivo...")
    problem_config = configure_problem(instance_data, k=k, risk_free_rate=0.03, lambda_=lambda_param)
    
    evaluation_func_with_logger = partial(
        portfolio_evaluation,
        instance_data=instance_data,
        k=k,
        risk_free_rate=0.03,
        logger=logger,
        lambda_=lambda_param,
    )
    
    problem_config["function"] = lambda weights: evaluation_func_with_logger(weights)[0]
    print("[INFO] Função objetivo configurada!")
    
    # ===== EXECUTAR METAHEURÍSTICA =====
    print("\n[PROCESS] Iniciando execução da Metaheurística...")
    print(f"[INFO] Iterações: {args.num_iterations}")
    print(f"[INFO] Agentes: {problem_config.get('num_agents', 30)}")
    
    try:
        met = Metaheuristic(
            problem_config,
            heur_sequence,
            num_iterations=args.num_iterations,
            num_agents=problem_config.get('num_agents', 30)
        )
        met.verbose = True
        met.run()
        
        x_best, f_best = met.get_solution()
        print("\n[INFO] Execução concluída!")
        print(f"[INFO] Melhor solução encontrada: {f_best:.8f}")
        print(f"[INFO] Vetor de pesos (primeiros 5): {x_best[:5]}")
        
    except Exception as e:
        print(f"[ERROR] Erro durante execução da metaheurística: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n[PROCESS] Fechando logger...")
        if logger is not None:
            logger.close()
            print("[INFO] Logs salvos.")
    
    # ===== SALVAR RESULTADOS =====
    print("\n[PROCESS] Salvando resultados...")
    try:
        # Ler logs do arquivo se criado
        all_logs_from_file = []
        if os.path.exists(log_file_path):
            df = pd.read_csv(log_file_path)
            all_logs_from_file = df.to_dict('records')
        
        if all_logs_from_file:
            print(f"[INFO] Total de avaliações: {len(all_logs_from_file)}")
            best_solution = min(all_logs_from_file, key=lambda x: x.get('objective', float('inf')))
            print(f"[INFO] Melhor objetivo (do log): {best_solution.get('objective', 'N/A'):.6f}")
            print(f"[INFO] Melhor Sharpe: {best_solution.get('sharpe', 'N/A'):.4f}")
            print(f"[INFO] Retorno: {best_solution.get('expected_return', 'N/A'):.4f}")
            print(f"[INFO] Risco: {best_solution.get('risk', 'N/A'):.4f}")
        
        # Salvar metadados
        metadata = {
            "hh_result_file": args.hh_result_file,
            "instance_file": args.instance_file,
            "cardinality": args.cardinality,
            "lambda_param": lambda_param,
            "num_iterations": args.num_iterations,
            "num_agents": problem_config.get('num_agents', 30),
            "operator_indices": operator_indices,
            "num_operators": len(selected_operators),
            "timestamp": datetime.now().isoformat(),
            "best_fitness": float(f_best),
        }
        
        metadata_file = os.path.join(output_dir, "metaheuristic_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[PROCESS] Resultados salvos em: {output_dir}")
        print("[INFO] execution_logs.csv: Log de todas as avaliações")
        print("[INFO] metaheuristic_metadata.json: Metadados da execução")
        
    except Exception as e:
        print(f"[ERROR] Erro ao salvar resultados: {e}")
    
    print("\n[PROCESS] Execução finalizada com sucesso!")


if __name__ == "__main__":
    main()
