#!/usr/bin/env python3
"""
Portfolio Optimization Lambda Sweep usando CustomHyS
Executa múltiplas otimizações com diferentes valores de lambda
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from functools import partial
import time
from datetime import datetime
import shutil
from multiprocessing import Pool, cpu_count
import random
import signal

from customhys.hyperheuristic import Hyperheuristic
from portfolio_utils.instance_reader import read_or_library_instance
from portfolio_utils.portfolio_logger import PortfolioLogger 
from portfolio_utils.portfolio_evaluator import portfolio_evaluation, configure_problem
from portfolio_utils.config_utils import load_hh_config, create_output_directory, save_logs
from portfolio_utils.portfolio_analyzer import analyze_portfolio_results

def run_single_lambda_execution(params):
    """
    executa uma unica otimizacao com um valor especifico de lambda.
    essa funcao eh chamada por cada worker no pool de processos.
    
    params: tupla com (lambda_param, execution_order, instance_data, k, hh_config, 
                       args, main_output_dir, total_lambdas)
    
    retorna dict com resultado da execucao
    """
    (lambda_param, execution_order, instance_data, k, hh_config, 
     args, main_output_dir, total_lambdas) = params
    
    # seed unico por worker pra evitar resultados identicos
    seed = int(time.time() * 1000) % (2**32) + execution_order
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"[INFO] EXECUCAO {execution_order}/{total_lambdas} - Lambda: {lambda_param:.4f}")
    print(f"{'='*60}")
    
    # criar subdiretorio para esta execucao
    sub_output_dir = os.path.join(main_output_dir, f"lambda_{lambda_param:.4f}")
    os.makedirs(sub_output_dir, exist_ok=True)
    
    logger = None
    result_from_hh = None
    execution_time = 0
    
    try:
        # configuracao do logger
        log_file_path = os.path.join(sub_output_dir, "execution_logs.csv")
        if args.no_logger:
            logger = None
            print("[INFO] Logger desabilitado por parametro CLI (--no-logger) para esta execucao.")
        else:
            logger = PortfolioLogger(log_file_path=log_file_path, buffer_size=100000)

        # configuracao do problema para a HH
        print("[INFO] Configurando problema...")
        problem_config = configure_problem(instance_data, k=k, risk_free_rate=0.03, lambda_=lambda_param)

        # ajustar epsilon/delta baseado na cardinalidade
        if k is None:
            evaluation_func_with_logger = partial(
                portfolio_evaluation, 
                instance_data=instance_data, 
                k=k, 
                risk_free_rate=0.03,
                logger=logger,
                lambda_=lambda_param,
            )
        else:
            evaluation_func_with_logger = partial(
                portfolio_evaluation, 
                instance_data=instance_data, 
                k=k, 
                risk_free_rate=0.03,
                logger=logger,
                lambda_=lambda_param,
                epsilon=0.01,
                delta=1.0
            )
        
        problem_config["function"] = lambda weights: evaluation_func_with_logger(weights)[0]
        print("[INFO] Problema configurado!")

        # execucao da hyper-heuristica
        print("[PROCESS] Iniciando execucao da Hyper-Heuristica...")
        execution_start = time.time()
        
        hh = Hyperheuristic(
            heuristic_space='default_portfolio.txt', 
            problem=problem_config, 
            parameters=hh_config,
            file_label=f"lambda_{lambda_param:.4f}"
        )
        
        result_from_hh = hh.solve()
        execution_time = time.time() - execution_start
        
        print(f"[INFO] Execucao concluida em {execution_time:.1f}s!")
        
        # copiar arquivos gerados pela HH
        raw_dd = os.path.join('data_files', 'raw', f"lambda_{lambda_param:.4f}")
        dest_raw = os.path.join(sub_output_dir, 'data_files_raw')
        try:
            if os.path.exists(raw_dd):
                shutil.copytree(raw_dd, dest_raw, dirs_exist_ok=True)
                print(f"[INFO] Data files da HH copiados para: {dest_raw}")
        except Exception as e:
            print(f"[WARNING] Falha ao copiar data_files/raw: {e}")
        
    except Exception as e:
        print(f"[ERROR] Erro durante execucao do lambda {lambda_param:.4f}: {e}")
        import traceback
        traceback.print_exc()
        result_from_hh = {"error": str(e)}
        execution_time = 0
        raise  # re-raise pra o pool detectar falha e tentar retry
        
    finally:
        print("[PROCESS] Salvando logs...")
        if logger is not None:
            logger.close()
            print("[INFO] Logs salvos.")

    # analise e salvamento dos resultados
    try:
        # ler logs gerados
        all_logs_from_file = []
        log_file_path = os.path.join(sub_output_dir, "execution_logs.csv")
        if os.path.exists(log_file_path):
            df = pd.read_csv(log_file_path)
            all_logs_from_file = df.to_dict('records')

        # encontrar melhor solucao
        best_solution = None
        if all_logs_from_file:
            best_solution = min(all_logs_from_file, key=lambda x: x.get('objective', float('inf')))
            
            print(f"[ANALYTIC] Resultados:")
            print(f"   - Total de avaliacoes: {len(all_logs_from_file)}")
            print(f"   - Melhor objetivo: {best_solution.get('objective', 'N/A'):.6f}")
            print(f"   - Melhor Sharpe: {best_solution.get('sharpe', 'N/A'):.4f}")
            print(f"   - Retorno: {best_solution.get('expected_return', 'N/A'):.4f}")
            print(f"   - Risco: {best_solution.get('risk', 'N/A'):.4f}")
            
            # armazenar resultado para analise posterior
            result_summary = {
                'lambda': lambda_param,
                'execution_order': execution_order,
                'execution_time': execution_time,
                'total_evaluations': len(all_logs_from_file),
                'best_objective': best_solution.get('objective', None),
                'best_sharpe': best_solution.get('sharpe', None),
                'best_return': best_solution.get('expected_return', None),
                'best_risk': best_solution.get('risk', None),
                'output_dir': sub_output_dir
            }
        else:
            result_summary = {
                'lambda': lambda_param,
                'execution_order': execution_order,
                'execution_time': execution_time,
                'total_evaluations': 0,
                'best_objective': None,
                'best_sharpe': None,
                'best_return': None,
                'best_risk': None,
                'output_dir': sub_output_dir
            }
        
        # salvar logs individuais
        save_logs(sub_output_dir, all_logs_from_file, instance_data, hh_config, result_from_hh)
        
        # analise avancada (opcional)
        if not args.no_analysis:
            if args.frontier_file and len(all_logs_from_file) > 10:
                try:
                    print("[PROCESS] Iniciando analise avancada...")
                    analyze_portfolio_results(sub_output_dir, args.frontier_file)
                    print("[INFO] Analise avancada concluida!")
                except Exception as e:
                    print(f"[WARNING] Erro na analise avancada: {e}")
        
        return result_summary
        
    except Exception as e:
        print(f"[ERROR] Erro ao processar resultados do lambda {lambda_param:.4f}: {e}")
        return {
            'lambda': lambda_param,
            'execution_order': execution_order,
            'execution_time': execution_time,
            'total_evaluations': 0,
            'best_objective': None,
            'best_sharpe': None,
            'best_return': None,
            'best_risk': None,
            'output_dir': sub_output_dir,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Portfolio Optimization Lambda Sweep usando CustomHyS')
    parser.add_argument('instance_file', help='Arquivo da instância (ex: port1.txt)')
    parser.add_argument('cardinality', type=int, help='Restrição de cardinalidade (0 = sem restrição)')
    parser.add_argument('config_file', help='Arquivo de configuração da hyper-heurística')
    parser.add_argument('frontier_file', nargs='?', default=None, 
                        help='Arquivo da fronteira eficiente (opcional)')
    parser.add_argument('lambda_intervals', type=int, 
                        help='Número de intervalos de lambda entre 0 e 1 (ex: 50)')
    parser.add_argument('--no-logger', action='store_true', default=False,
                        help='Desabilita a criação do logger (logs em execution_logs.csv)')
    parser.add_argument('--no-analysis', action='store_true', default=False,
                        help='Pula a análise avançada (analyze_portfolio_results)')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Executa de forma sequencial ao invés de paralela')
    args = parser.parse_args()

    # Carregamento de dados e configurações
    instance_data = read_or_library_instance(args.instance_file)
    k = None if args.cardinality == 0 else args.cardinality
    hh_config = load_hh_config(args.config_file)
    
    # Gerar valores de lambda
    lambda_values = np.linspace(0.001, 1, args.lambda_intervals)  # Evitar 0 e 1 exatos
    
    print(f"[CONFIG] Lambda Sweep configurado:")
    print(f"   - Número de intervalos: {args.lambda_intervals}")
    print(f"   - Valores de lambda: {lambda_values[0]:.3f} até {lambda_values[-1]:.3f}")
    print(f"   - Total de execuções: {len(lambda_values)}")
    
    # Validar arquivo da fronteira eficiente se fornecido
    if args.frontier_file and not os.path.exists(args.frontier_file):
        print(f"[WARNING] Arquivo da fronteira eficiente não encontrado: {args.frontier_file}")
        print("   A análise será executada sem a fronteira eficiente.")
        args.frontier_file = None
    elif args.frontier_file:
        print(f"[INFO] Arquivo da fronteira eficiente encontrado: {args.frontier_file}")
    else:
        print("[INFO] Nenhum arquivo de fronteira eficiente fornecido.")
    
    # Criar diretório principal para todos os resultados
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_name = os.path.splitext(os.path.basename(args.instance_file))[0]
    scheme_name = hh_config.get('initial_scheme', 'unknown')
    main_output_dir = f"{base_timestamp}_{instance_name}_{scheme_name}_lambda_sweep_{args.lambda_intervals}"
    
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"[INFO] Diretório principal: {main_output_dir}")
    
    # lista para armazenar resultados de todas as execucoes
    all_results = []
    start_time = time.time()
    
    # preparar parametros para cada execucao
    execution_params = [
        (lambda_val, i, instance_data, k, hh_config, args, main_output_dir, len(lambda_values))
        for i, lambda_val in enumerate(lambda_values, 1)
    ]
    
    # decidir se executa paralelo ou sequencial
    if args.sequential:
        print("[INFO] Modo SEQUENCIAL ativado (--sequential)")
        print(f"[INFO] Executando {len(lambda_values)} otimizacoes em sequencia...\n")
        
        # execucao sequencial (modo original)
        for params in execution_params:
            try:
                result = run_single_lambda_execution(params)
                all_results.append(result)
            except Exception as e:
                lambda_val = params[0]
                exec_order = params[1]
                print(f"[ERROR] Falha na execucao do lambda {lambda_val:.4f}, tentando retry...")
                
                # retry uma vez
                try:
                    print(f"[RETRY] Reexecutando lambda {lambda_val:.4f}...")
                    result = run_single_lambda_execution(params)
                    all_results.append(result)
                    print(f"[SUCCESS] Retry do lambda {lambda_val:.4f} foi bem-sucedido!")
                except Exception as retry_error:
                    print(f"[ERROR] Retry falhou para lambda {lambda_val:.4f}: {retry_error}")
                    all_results.append({
                        'lambda': lambda_val,
                        'execution_order': exec_order,
                        'execution_time': 0,
                        'total_evaluations': 0,
                        'best_objective': None,
                        'best_sharpe': None,
                        'best_return': None,
                        'best_risk': None,
                        'output_dir': os.path.join(main_output_dir, f"lambda_{lambda_val:.4f}"),
                        'error': f'falhou apos retry: {str(retry_error)}'
                    })
    else:
        # execucao PARALELA (modo otimizado)
        n_workers = max(1, cpu_count() // 2)
        print("[INFO] Modo PARALELO ativado")
        print(f"[INFO] Usando {n_workers} workers (metade dos {cpu_count()} cores disponiveis)")
        print(f"[INFO] Executando {len(lambda_values)} otimizacoes em paralelo...")
        print("[INFO] Pressione Ctrl+C para interromper a execucao\n")
        
        # criar pool de workers
        # usando context manager que fecha pool automaticamente
        pool = Pool(processes=n_workers)
        
        try:
            # executar todas as tarefas em paralelo
            # map_async permite acompanhar progresso
            results_async = []
            for params in execution_params:
                result = pool.apply_async(run_single_lambda_execution, (params,))
                results_async.append((params, result))
            
            # coletar resultados conforme ficam prontos
            completed = 0
            for params, result_async in results_async:
                lambda_val = params[0]
                exec_order = params[1]
                
                try:
                    result = result_async.get(timeout=3600)  # timeout de 1 hora por lambda
                    all_results.append(result)
                    completed += 1
                    
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (len(lambda_values) - completed) * avg_time / n_workers
                    
                    print(f"\n[PROGRESS] {completed}/{len(lambda_values)} lambdas concluidos ({100*completed/len(lambda_values):.1f}%)")
                    print(f"[PROGRESS] Tempo decorrido: {elapsed/60:.1f}min | Estimado restante: {remaining/60:.1f}min")
                    
                except Exception as e:
                    print(f"[ERROR] Falha na execucao do lambda {lambda_val:.4f}, tentando retry...")
                    
                    # retry sequencial (fora do pool)
                    try:
                        print(f"[RETRY] Reexecutando lambda {lambda_val:.4f} fora do pool...")
                        result = run_single_lambda_execution(params)
                        all_results.append(result)
                        completed += 1
                        print(f"[SUCCESS] Retry do lambda {lambda_val:.4f} foi bem-sucedido!")
                    except Exception as retry_error:
                        print(f"[ERROR] Retry falhou para lambda {lambda_val:.4f}: {retry_error}")
                        all_results.append({
                            'lambda': lambda_val,
                            'execution_order': exec_order,
                            'execution_time': 0,
                            'total_evaluations': 0,
                            'best_objective': None,
                            'best_sharpe': None,
                            'best_return': None,
                            'best_risk': None,
                            'output_dir': os.path.join(main_output_dir, f"lambda_{lambda_val:.4f}"),
                            'error': f'falhou apos retry: {str(retry_error)}'
                        })
                        completed += 1
        
        except KeyboardInterrupt:
            print("\n\n[WARNING] Execucao interrompida pelo usuario (Ctrl+C)!")
            print("[PROCESS] Terminando workers ativos...")
            pool.terminate()  # mata todos os workers imediatamente
            pool.join()  # espera workers terminarem
            print("[INFO] Workers terminados. Salvando resultados parciais...")
            # all_results ja tem os lambdas que terminaram ate agora
        
        finally:
            # garante que pool seja fechado mesmo em caso de erro
            pool.close()
            pool.join()

    # analise final consolidada
    print(f"\n{'='*60}")
    print("[INFO] ANÁLISE CONSOLIDADA DOS RESULTADOS")
    print(f"{'='*60}")
    
    # Salvar resultados consolidados
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(main_output_dir, "lambda_sweep_results.csv")
    results_df.to_csv(results_file, index=False)
    
    # Estatísticas gerais
    valid_results = results_df[results_df['best_objective'].notna()]
    
    if len(valid_results) > 0:
        print(f"[INFO] Execuções bem-sucedidas: {len(valid_results)}/{len(lambda_values)}")
        print(f"[ANALYTIC] Estatísticas dos objetivos:")
        print(f"   - Melhor: {valid_results['best_objective'].min():.6f}")
        print(f"   - Pior: {valid_results['best_objective'].max():.6f}")
        print(f"   - Média: {valid_results['best_objective'].mean():.6f}")
        print(f"   - Desvio padrão: {valid_results['best_objective'].std():.6f}")
        
        # Melhor lambda encontrado
        best_row = valid_results.loc[valid_results['best_objective'].idxmin()]
        print(f"[ANALYTIC] Melhor resultado:")
        print(f"   - Lambda: {best_row['lambda']:.4f}")
        print(f"   - Objetivo: {best_row['best_objective']:.6f}")
        print(f"   - Sharpe: {best_row['best_sharpe']:.4f}")
        print(f"   - Retorno: {best_row['best_return']:.4f}")
        print(f"   - Risco: {best_row['best_risk']:.4f}")
    else:
        print("[ERROR] Nenhuma execução foi bem-sucedida!")
    
    total_time = time.time() - start_time
    print(f"\n[INFO] Tempo total de execução: {total_time/60:.1f}min")
    print(f"[INFO] Todos os resultados salvos em: {main_output_dir}")
    print(f"[INFO] Resultados consolidados em: {results_file}")
    
    print("\n[INFO] Lambda Sweep finalizado!")

if __name__ == "__main__":
    main()
