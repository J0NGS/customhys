#!/usr/bin/env python3
"""
Portfolio Optimization Lambda Sweep usando CustomHyS
Executa múltiplas otimizações com diferentes valores de lambda
Salva todas as avaliações em formato Parquet otimizado - VERSÃO CORRIGIDA

⭐ OTIMIZAÇÃO: Shared Memory para instance_data (evita pickle overhead)
"""

import os
import argparse
import pandas as pd
import numpy as np
import time
from datetime import datetime
import shutil
from multiprocessing import Pool, cpu_count
import random
import glob

from customhys.hyperheuristic import Hyperheuristic
from portfolio_utils.instance_reader import read_or_library_instance
from portfolio_utils.parquet_handler import ParquetBufferWriter, JSONToParquetConverter, validate_parquet_file
from portfolio_utils.portfolio_evaluator import configure_problem
from portfolio_utils.config_utils import load_hh_config, save_logs
from portfolio_utils.portfolio_analyzer import analyze_portfolio_results

# ⭐ SHARED MEMORY: Variable global para evitar serialização via pickle
_global_instance_data = None
_global_hh_config = None

def _init_worker(instance_data, hh_config):
    """
    ⭐ Inicializador do Pool: Define dados globais em cada worker process
    Evita que o pickle serializar instance_data a cada tarefa
    """
    global _global_instance_data, _global_hh_config
    _global_instance_data = instance_data
    _global_hh_config = hh_config


def run_single_lambda_execution(params):
    """
    ⭐ OTIMIZADO: Usa variáveis globais em vez de receber instance_data via pickle
    
    Executa uma única otimização com um valor específico de lambda.
    Salva dados em Parquet.
    
    params: tupla com (lambda_param, execution_order, k, args, main_output_dir, total_lambdas, base_timestamp)
            (instance_data e hh_config vêm via globals)
    
    Retorna dict com resultado da execução
    """
    (lambda_param, execution_order, k, args, main_output_dir, total_lambdas, base_timestamp) = params
    
    # ⭐ Recuperar dados globais (já estão em memória deste worker)
    instance_data = _global_instance_data
    hh_config = _global_hh_config
    
    # Seed único por worker para evitar resultados idênticos
    seed = int(time.time() * 1000) % (2**32) + execution_order
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"[INFO] EXECUÇÃO {execution_order}/{total_lambdas} - Lambda: {lambda_param:.4f}")
    print(f"{'='*60}")
    
    # Criar subdiretório para esta execução
    sub_output_dir = os.path.join(main_output_dir, f"lambda_{lambda_param:.4f}")
    os.makedirs(sub_output_dir, exist_ok=True)
    
    logger = None
    result_from_hh = None
    execution_time = 0
    evaluator = None
    
    try:
        # Configuração do logger em Parquet
        log_file_path = os.path.join(sub_output_dir, "execution_logs.parquet")
        if args.no_logger:
            logger = None
            print("[INFO] Logger desabilitado por parâmetro CLI (--no-logger).")
        else:
            # ⭐ OTIMIZAÇÃO NUMPY: Passar n_assets para pré-alocar matriz 2D para weights
            # ⭐ CRÍTICO: Buffer_size reduzido para evitar OOM com 4 workers paralelos
            # Antes era 5000, agora 2500 para compartilhar RAM entre workers
            n_assets = instance_data.get("n_assets", None)
            logger = ParquetBufferWriter(
                file_path=log_file_path, 
                buffer_size=2500,  # ⭐ Reduzido de 5000 → 2500 (evita OOM)
                n_assets=n_assets,  # ⭐ Ativa matriz 2D pré-alocada
                compression='snappy'
            )
            print(f"[INFO] Logger Parquet configurado: {log_file_path} (n_assets={n_assets}, buffer=2500)")

        # Configuração do problema com STATEFUL EVALUATOR
        print("[INFO] Configurando problema com Stateful Evaluator...")
        
        problem_config = configure_problem(
            instance_data, 
            k=k, 
            risk_free_rate=0.03,
            lambda_=lambda_param,
            logger=logger
        )
        
        evaluator = problem_config.get("evaluator")
        print("[INFO] Problema configurado com Stateful Evaluator!")

        # Execução da hyper-heurística
        print("[PROCESS] Iniciando execução da Hyper-Heurística...")
        execution_start = time.time()
        
        # ⭐ TIMESTAMP ISOLADO: Cada sweep tem timestamp único para evitar conflitos
        # HH escreve em: data_files/raw/{base_timestamp}_lambda_X/
        file_label_with_timestamp = f"{base_timestamp}_lambda_{lambda_param:.4f}"
        
        hh = Hyperheuristic(
            heuristic_space='default_portfolio.txt', 
            problem=problem_config, 
            parameters=hh_config,
            file_label=file_label_with_timestamp
        )
        
        result_from_hh = hh.solve()
        execution_time = time.time() - execution_start
        
        print(f"[INFO] Execução concluída em {execution_time:.1f}s!")
        
        # ⭐ COPIAR E LIMPAR: Copiar data_files/raw/{timestamp}_lambda_X/ para output
        # Depois DELETAR a pasta do data_files/raw para evitar redundância
        raw_dd = os.path.join('data_files', 'raw', file_label_with_timestamp)
        dest_raw = os.path.join(sub_output_dir, 'data_files_raw')
        try:
            if os.path.exists(raw_dd):
                shutil.copytree(raw_dd, dest_raw, dirs_exist_ok=True)
                print(f"[INFO] Data files da HH copiados para: {dest_raw}")
                
                # Converter JSONs para Parquet
                _convert_data_files_to_parquet(dest_raw)
                
                # ⭐ LIMPEZA: Deletar pasta de origem (data_files/raw/{timestamp}_lambda_X/)
                # Já foi copiada e comprimida, não precisa manter no global
                try:
                    shutil.rmtree(raw_dd)
                    print(f"[CLEANUP] Pasta removida: {raw_dd} (cópia local já salva)")
                except Exception as cleanup_error:
                    print(f"[WARNING] Falha ao remover {raw_dd}: {cleanup_error}")
        except Exception as e:
            print(f"[WARNING] Falha ao copiar/converter data_files: {e}")
        
    except Exception as e:
        print(f"[ERROR] Erro durante execução do lambda {lambda_param:.4f}: {e}")
        import traceback
        traceback.print_exc()
        result_from_hh = {"error": str(e)}
        execution_time = 0
        raise
        
    finally:
        print("[PROCESS] Finalizando...")
        # Finalizar o avaliador
        if evaluator is not None:
            evaluator.finalize()
            stats = evaluator.get_stats()
            print(f"[STATS] Total de avaliações: {stats['total_evaluations']}")
            print(f"[STATS] Melhor objetivo: {stats['best_objective']:.6e}")
            if stats['log_file_size_mb']:
                print(f"[STATS] Tamanho do arquivo de log: {stats['log_file_size_mb']:.2f} MB")
        
        # ⭐ CRÍTICO: Liberar memória ANTES de fechar o Parquet
        # Isso garante que haja RAM suficiente para serializar o footer
        import gc
        gc.collect()
        
        if logger is not None:
            logger.close()
            print("[INFO] Logger Parquet finalizado.")

    # Análise e salvamento dos resultados
    try:
        # ⭐ VALIDAÇÃO DE INTEGRIDADE: Verificar Parquet antes de ler
        df = None
        log_file_path = os.path.join(sub_output_dir, "execution_logs.parquet")
        
        # Validar arquivo Parquet
        is_valid, error_msg, csv_fallback = validate_parquet_file(log_file_path)
        
        if not is_valid:
            print(f"[WARNING] {error_msg}")
            if csv_fallback and os.path.exists(csv_fallback):
                print(f"[INFO] Usando fallback CSV: {csv_fallback}")
                log_file_path = csv_fallback
            else:
                print(f"[ERROR] Sem fallback CSV disponível, pulando análise")
                log_file_path = None
        
        if log_file_path and os.path.exists(log_file_path):
            # Carregar APENAS as colunas necessárias para estatísticas
            cols_needed = ['objective', 'expected_return', 'risk']
            df = pd.read_parquet(log_file_path, columns=cols_needed) if log_file_path.endswith('.parquet') else pd.read_csv(log_file_path, usecols=cols_needed)
            
            # ⭐ VETORIAL: Usar métodos Pandas em vez de loops Python
            # Melhor objetivo usando índice (sem criar dict)
            best_idx = df['objective'].idxmin()
            best_solution_row = df.iloc[best_idx]
            
            print(f"[ANALYTIC] Resultados:")
            print(f"   - Total de avaliações: {len(df)}")
            print(f"   - Melhor objetivo: {best_solution_row['objective']:.6f}")
            print(f"   - Retorno: {best_solution_row['expected_return']:.4f}")
            print(f"   - Risco: {best_solution_row['risk']:.4f}")
            
            # Armazenar resultado para análise posterior
            result_summary = {
                'lambda': lambda_param,
                'execution_order': execution_order,
                'execution_time': execution_time,
                'total_evaluations': len(df),
                'best_objective': float(best_solution_row['objective']),
                'best_return': float(best_solution_row['expected_return']),
                'best_risk': float(best_solution_row['risk']),
                'output_dir': sub_output_dir
            }
        else:
            result_summary = {
                'lambda': lambda_param,
                'execution_order': execution_order,
                'execution_time': execution_time,
                'total_evaluations': 0,
                'best_objective': None,
                'best_return': None,
                'best_risk': None,
                'output_dir': sub_output_dir
            }
        
        # ⭐ OTIMIZAÇÃO RAM: Passar None para all_logs_from_file
        # Dados já estão em Parquet, não precisa duplicar em JSON
        # save_logs() criará apenas o resumo estatístico (summary_stats.json)
        save_logs(sub_output_dir, None, instance_data, hh_config, result_from_hh)
        
        # Análise avançada (opcional)
        if not args.no_analysis:
            # ⭐ Verificar se Parquet tem dados suficientes para análise
            if args.frontier_file and df is not None and len(df) > 10:
                try:
                    print("[PROCESS] Iniciando análise avançada...")
                    analyze_portfolio_results(sub_output_dir, args.frontier_file, risk_free_rate=args.risk_free_rate)
                    print("[INFO] Análise avançada concluída!")
                except Exception as e:
                    print(f"[WARNING] Erro na análise avançada: {e}")
        
        # ⭐ LIMPEZA FINAL: Liberar memória do lambda antes de retornar
        import gc
        gc.collect()
        
        return result_summary
        
    except Exception as e:
        print(f"[ERROR] Erro ao processar resultados do lambda {lambda_param:.4f}: {e}")
        import gc
        gc.collect()
        
        return {
            'lambda': lambda_param,
            'execution_order': execution_order,
            'execution_time': execution_time,
            'total_evaluations': 0,
            'best_objective': None,
            'best_return': None,
            'best_risk': None,
            'output_dir': sub_output_dir,
            'error': str(e)
        }


def _convert_data_files_to_parquet(directory):
    """Converte JSON files para Parquet."""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    if not json_files:
        return
    
    print(f"[PARQUET] Convertendo {len(json_files)} JSON files...")
    for json_file in json_files:
        try:
            JSONToParquetConverter.json_to_parquet(json_file, compression='snappy')
            os.remove(json_file)
        except Exception as e:
            print(f"[WARNING] Falha ao converter {json_file}: {e}")



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
                        help='Desabilita a criação do logger')
    parser.add_argument('--no-analysis', action='store_true', default=False,
                        help='Pula a análise avançada')
    parser.add_argument('--sequential', action='store_true', default=False,
                        help='Executa de forma sequencial ao invés de paralela')
    parser.add_argument('--workers', type=int, default=None,
                        help='Número de workers paralelos (padrão: cpu_count // 2). Ignorado com --sequential')
    parser.add_argument('--risk-free-rate', type=float, default=0.00057,
                        help='Taxa livre de risco (padrão: 0.00057)')
    args = parser.parse_args()

    # Carregamento de dados e configurações
    instance_data = read_or_library_instance(args.instance_file)
    k = None if args.cardinality == 0 else args.cardinality
    hh_config = load_hh_config(args.config_file)
    
    # Gerar valores de lambda
    lambda_values = np.linspace(0.001, 1, args.lambda_intervals)
    
    print(f"[CONFIG] Lambda Sweep configurado:")
    print(f"   - Número de intervalos: {args.lambda_intervals}")
    print(f"   - Valores de lambda: {lambda_values[0]:.3f} até {lambda_values[-1]:.3f}")
    print(f"   - Total de execuções: {len(lambda_values)}")
    
    # Validar arquivo da fronteira eficiente
    if args.frontier_file and not os.path.exists(args.frontier_file):
        print(f"[WARNING] Arquivo da fronteira eficiente não encontrado: {args.frontier_file}")
        args.frontier_file = None
    elif args.frontier_file:
        print(f"[INFO] Arquivo da fronteira eficiente encontrado: {args.frontier_file}")
    else:
        print("[INFO] Nenhum arquivo de fronteira eficiente fornecido.")
    
    # Criar diretório principal
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_name = os.path.splitext(os.path.basename(args.instance_file))[0]
    scheme_name = hh_config.get('initial_scheme', 'unknown')
    main_output_dir = f"{base_timestamp}_{instance_name}_{scheme_name}_lambda_sweep_{args.lambda_intervals}"
    
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"[INFO] Diretório principal: {main_output_dir}\n")
    
    # Lista para armazenar resultados
    all_results = []
    start_time = time.time()
    
    # ⭐ OTIMIZAÇÃO: Preparar parâmetros SEM instance_data (será compartilhado via globals)
    # ⭐ TIMESTAMP ISOLADO: Incluir base_timestamp para isolar cada sweep
    execution_params = [
        (lambda_val, i, k, args, main_output_dir, len(lambda_values), base_timestamp)
        for i, lambda_val in enumerate(lambda_values, 1)
    ]
    
    # Decidir se executa paralelo ou sequencial
    if args.sequential:
        print("[INFO] Modo SEQUENCIAL ativado (--sequential)\n")
        
        for params in execution_params:
            try:
                result = run_single_lambda_execution(params)
                all_results.append(result)
            except Exception as e:
                lambda_val = params[0]
                exec_order = params[1]
                print(f"[ERROR] Falha no lambda {lambda_val:.4f}, tentando retry...")
                
                try:
                    print(f"[RETRY] Reexecutando lambda {lambda_val:.4f}...")
                    result = run_single_lambda_execution(params)
                    all_results.append(result)
                    print(f"[SUCCESS] Retry bem-sucedido!")
                except Exception as retry_error:
                    print(f"[ERROR] Retry falhou: {retry_error}")
                    all_results.append({
                        'lambda': lambda_val,
                        'execution_order': exec_order,
                        'execution_time': 0,
                        'total_evaluations': 0,
                        'best_objective': None,
                        'best_return': None,
                        'best_risk': None,
                        'output_dir': os.path.join(main_output_dir, f"lambda_{lambda_val:.4f}"),
                        'error': f'falhou: {str(retry_error)}'
                    })
    else:
        # Modo PARALELO
        # ⭐ Usar argumento --workers se fornecido, senão usar cpu_count // 2
        if args.workers is not None:
            n_workers = max(1, args.workers)
            print("[INFO] Modo PARALELO ativado")
            print(f"[INFO] Usando {n_workers} workers (especificado via --workers)")
        else:
            n_workers = max(1, cpu_count() // 2)
            print("[INFO] Modo PARALELO ativado")
            print(f"[INFO] Usando {n_workers} workers (metade dos {cpu_count()} cores disponíveis)")
        
        print(f"[INFO] Executando {len(lambda_values)} otimizações em paralelo...\n")
        
        # ⭐ OTIMIZAÇÃO: Pool com inicializador para shared memory
        pool = Pool(processes=n_workers, initializer=_init_worker, 
                   initargs=(instance_data, hh_config))
        
        try:
            results_async = []
            for params in execution_params:
                result = pool.apply_async(run_single_lambda_execution, (params,))
                results_async.append((params, result))
            
            completed = 0
            for params, result_async in results_async:
                lambda_val = params[0]
                exec_order = params[1]
                
                try:
                    result = result_async.get(timeout=3600)
                    all_results.append(result)
                    completed += 1
                    
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (len(lambda_values) - completed) * avg_time / n_workers
                    
                    print(f"\n[PROGRESS] {completed}/{len(lambda_values)} lambdas ({100*completed/len(lambda_values):.1f}%)")
                    print(f"[PROGRESS] Decorrido: {elapsed/60:.1f}min | Estimado: {remaining/60:.1f}min")
                    
                except Exception as e:
                    print(f"[ERROR] Falha no lambda {lambda_val:.4f}, retry...")
                    
                    try:
                        print(f"[RETRY] Reexecutando fora do pool...")
                        result = run_single_lambda_execution(params)
                        all_results.append(result)
                        completed += 1
                        print(f"[SUCCESS] Retry bem-sucedido!")
                    except Exception as retry_error:
                        print(f"[ERROR] Retry falhou: {retry_error}")
                        all_results.append({
                            'lambda': lambda_val,
                            'execution_order': exec_order,
                            'execution_time': 0,
                            'total_evaluations': 0,
                            'best_objective': None,
                            'best_return': None,
                            'best_risk': None,
                            'output_dir': os.path.join(main_output_dir, f"lambda_{lambda_val:.4f}"),
                            'error': f'falhou: {str(retry_error)}'
                        })
                        completed += 1
        
        except KeyboardInterrupt:
            print("\n\n[WARNING] Execução interrompida (Ctrl+C)!")
            print("[PROCESS] Terminando workers...")
            pool.terminate()
            pool.join()
            print("[INFO] Salvando resultados parciais...")
        
        finally:
            pool.close()
            pool.join()

    # Análise final consolidada
    print(f"\n{'='*60}")
    print("[INFO] ANÁLISE CONSOLIDADA DOS RESULTADOS")
    print(f"{'='*60}")
    
    # Salvar resultados consolidados
    results_df = pd.DataFrame(all_results)
    
    if 'best_objective' in results_df.columns:
        results_df['best_objective'] = pd.to_numeric(results_df['best_objective'], errors='coerce')
    
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
        
        best_row = valid_results.loc[valid_results['best_objective'].idxmin()]
        print(f"[ANALYTIC] Melhor resultado:")
        print(f"   - Lambda: {best_row['lambda']:.4f}")
        print(f"   - Objetivo: {best_row['best_objective']:.6f}")
        print(f"   - Retorno: {best_row['best_return']:.4f}")
        print(f"   - Risco: {best_row['best_risk']:.4f}")
    else:
        print("[ERROR] Nenhuma execução foi bem-sucedida!")
    
    total_time = time.time() - start_time
    print(f"\n[INFO] Tempo total: {total_time/60:.1f}min")
    print(f"[INFO] Resultados em: {main_output_dir}")
    print(f"[INFO] Consolidado em: {results_file}")
    print("\n[INFO] Lambda Sweep finalizado!")

if __name__ == "__main__":
    main()
