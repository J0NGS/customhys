#!/usr/bin/env python3
"""
Batch Metaheuristic Executor
Executa metaheuristicas com operadores fixos (descobertos pela HH) em cada lambda do sweep.
Compara performance HH vs MH e gera analises completas.
"""

import os
import sys
import json
import glob
import argparse
import time
from datetime import datetime
from multiprocessing import Pool


def parse_operator_indices(indices_str):
    """
    parseia a string de indices do hh_result.json.
    ex: "[ 98  55  53 120]" -> [98, 55, 53, 120]
    """
    indices_str = indices_str.strip().strip('[]')
    indices = [int(x) for x in indices_str.split()]
    return indices


def load_operators_from_collection(collection_file, indices):
    """
    carrega operadores do arquivo de colecao usando indices.
    retorna lista de tuplas (operator_name, params, selector).
    """
    with open(collection_file, 'r') as f:
        lines = f.read().strip().split('\n')
    
    all_operators = [eval(line) for line in lines if line.strip()]
    selected_operators = [all_operators[idx] for idx in indices if idx < len(all_operators)]
    
    return selected_operators


def build_metaheuristic_sequence(selected_operators):
    """
    constroi sequencia de operadores no formato esperado pelo Metaheuristic.
    """
    heur_sequence = []
    
    for operator_tuple in selected_operators:
        perturbador, params, seletor = operator_tuple
        heur_sequence.append((perturbador, params, seletor))
    
    return heur_sequence


def find_lambda_folders(sweep_dir):
    """
    encontra todas as subpastas lambda_X.XXXX dentro do diretorio de sweep.
    """
    pattern = os.path.join(sweep_dir, "lambda_*")
    lambda_folders = glob.glob(pattern)
    
    lambda_folders = [f for f in lambda_folders if os.path.isdir(f)]
    lambda_folders.sort(key=lambda x: float(os.path.basename(x).split('_')[1]))
    
    return lambda_folders


def extract_lambda_value(lambda_folder):
    """extrai o valor de lambda do nome da pasta."""
    return float(os.path.basename(lambda_folder).split('_')[1])


def load_hh_config_from_lambda(lambda_folder):
    """
    tenta carregar configuracoes da HH a partir dos arquivos na pasta lambda.
    retorna dict com configuracoes ou None.
    """
    # tentar ler do hh_resume.txt
    resume_file = os.path.join(lambda_folder, "data_files_raw", "hh_resume.txt")
    
    config = {}
    
    if os.path.exists(resume_file):
        try:
            with open(resume_file, 'r') as f:
                content = f.read()
                
            # parsear informacoes basicas
            for line in content.split('\n'):
                if 'num_iterations' in line.lower():
                    try:
                        config['num_iterations'] = int(line.split(':')[-1].strip())
                    except Exception:
                        pass
                elif 'num_agents' in line.lower() or 'population' in line.lower():
                    try:
                        config['num_agents'] = int(line.split(':')[-1].strip())
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARNING] Erro ao ler hh_resume.txt: {e}")
    
    return config if config else None


def compare_hh_vs_mh(lambda_folder, mh_output_dir):
    """
    compara resultados da HH vs MH.
    retorna dict com estatisticas comparativas.
    """
    import pandas as pd
    
    comparison = {
        'lambda_value': extract_lambda_value(lambda_folder),
        'hh_metrics': {},
        'mh_metrics': {},
        'gaps': {}
    }
    
    # ler logs da HH
    hh_logs_file = os.path.join(lambda_folder, "execution_logs.csv")
    if os.path.exists(hh_logs_file):
        try:
            df_hh = pd.read_csv(hh_logs_file)
            
            comparison['hh_metrics'] = {
                'total_evaluations': int(len(df_hh)),
                'best_fitness': float(df_hh['objective'].min()),
                'best_sharpe': float(df_hh['sharpe'].max()) if 'sharpe' in df_hh.columns else None,
                'best_return': float(df_hh.loc[df_hh['objective'].idxmin(), 'expected_return']),
                'best_risk': float(df_hh.loc[df_hh['objective'].idxmin(), 'risk']),
                'fitness_mean': float(df_hh['objective'].mean()),
                'fitness_median': float(df_hh['objective'].median()),
                'fitness_std': float(df_hh['objective'].std()),
                'fitness_min': float(df_hh['objective'].min()),
                'fitness_max': float(df_hh['objective'].max())
            }
        except Exception as e:
            print(f"[WARNING] Erro ao ler logs HH: {e}")
    
    # ler logs da MH
    mh_logs_file = os.path.join(mh_output_dir, "execution_logs.csv")
    if os.path.exists(mh_logs_file):
        try:
            df_mh = pd.read_csv(mh_logs_file)
            
            comparison['mh_metrics'] = {
                'total_evaluations': int(len(df_mh)),
                'best_fitness': float(df_mh['objective'].min()),
                'best_sharpe': float(df_mh['sharpe'].max()) if 'sharpe' in df_mh.columns else None,
                'best_return': float(df_mh.loc[df_mh['objective'].idxmin(), 'expected_return']),
                'best_risk': float(df_mh.loc[df_mh['objective'].idxmin(), 'risk']),
                'fitness_mean': float(df_mh['objective'].mean()),
                'fitness_median': float(df_mh['objective'].median()),
                'fitness_std': float(df_mh['objective'].std()),
                'fitness_min': float(df_mh['objective'].min()),
                'fitness_max': float(df_mh['objective'].max())
            }
        except Exception as e:
            print(f"[WARNING] Erro ao ler logs MH: {e}")
    
    # calcular gaps
    if comparison['hh_metrics'] and comparison['mh_metrics']:
        hh_best = comparison['hh_metrics']['best_fitness']
        mh_best = comparison['mh_metrics']['best_fitness']
        
        # Gap positivo = MH melhor (minimização)
        # Gap negativo = HH melhor
        comparison['gaps'] = {
            'absolute_gap': float(hh_best - mh_best),
            'relative_gap_percent': float(((hh_best - mh_best) / abs(hh_best)) * 100 if hh_best != 0 else 0),
            'mh_is_better': bool(mh_best < hh_best),
            'fitness_mean_gap': float(comparison['hh_metrics']['fitness_mean'] - comparison['mh_metrics']['fitness_mean']),
            'fitness_std_gap': float(comparison['hh_metrics']['fitness_std'] - comparison['mh_metrics']['fitness_std'])
        }
    
    return comparison


def execute_metaheuristic_for_lambda(params):
    """
    worker function para executar metaheuristica em uma pasta lambda.
    
    args:
        params: tupla com todos os parametros necessarios
    
    returns:
        dict com resultado da execucao
    """
    # Imports dentro da função para evitar reimportação no Windows spawn
    from functools import partial
    from customhys.metaheuristic import Metaheuristic
    from portfolio_utils.portfolio_logger import PortfolioLogger
    from portfolio_utils.portfolio_evaluator import portfolio_evaluation, configure_problem
    from portfolio_utils.portfolio_analyzer import analyze_portfolio_results
    
    (lambda_folder, instance_data, k, collection_file, num_iterations, 
     num_agents, analyze, idx, total) = params
    
    lambda_name = os.path.basename(lambda_folder)
    lambda_value = extract_lambda_value(lambda_folder)
    
    result = {
        'lambda_folder': lambda_folder,
        'lambda_name': lambda_name,
        'lambda_value': lambda_value,
        'index': idx,
        'success': False,
        'error': None,
        'execution_time': 0,
        'comparison': None
    }
    
    print(f"\n[{idx}/{total}] Processando {lambda_name}...")
    
    # verificar se existe hh_result.json (na raiz da pasta lambda)
    hh_result_file = os.path.join(lambda_folder, "hh_result.json")
    if not os.path.exists(hh_result_file):
        result['error'] = 'hh_result.json nao encontrado'
        print(f"[{idx}/{total}] [ERROR] {lambda_name}: {result['error']}")
        return result
    
    # criar diretorio de saida
    mh_output_dir = os.path.join(lambda_folder, "metaheuristic_rerun")
    os.makedirs(mh_output_dir, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # carregar hh_result.json
        with open(hh_result_file, 'r') as f:
            hh_result = json.load(f)
        
        indices_str = hh_result[0]
        operator_indices = parse_operator_indices(indices_str)
        
        # carregar hh_config.json para pegar initial_scheme
        hh_config_file = os.path.join(lambda_folder, "hh_config.json")
        initial_scheme = "random"  # default
        
        if os.path.exists(hh_config_file):
            try:
                with open(hh_config_file, 'r') as f:
                    hh_config = json.load(f)
                    initial_scheme = hh_config.get("initial_scheme", "random")
                    print(f"[{idx}/{total}] Initial scheme detectado: {initial_scheme}")
            except Exception as e:
                print(f"[{idx}/{total}] [WARNING] Erro ao ler hh_config.json: {e}")
        
        # carregar operadores
        selected_operators = load_operators_from_collection(collection_file, operator_indices)
        heur_sequence = build_metaheuristic_sequence(selected_operators)
        
        # configurar logger
        log_file_path = os.path.join(mh_output_dir, "execution_logs.csv")
        logger = PortfolioLogger(log_file_path=log_file_path, buffer_size=100000)
        
        # configurar problema (mesmos parametros da HH)
        problem_config = configure_problem(instance_data, k=k, risk_free_rate=0.03, lambda_=lambda_value)
        
        if k is None:
            evaluation_func_with_logger = partial(
                portfolio_evaluation,
                instance_data=instance_data,
                k=k,
                risk_free_rate=0.03,
                logger=logger,
                lambda_=lambda_value,
            )
        else:
            evaluation_func_with_logger = partial(
                portfolio_evaluation,
                instance_data=instance_data,
                k=k,
                risk_free_rate=0.03,
                logger=logger,
                lambda_=lambda_value,
                epsilon=0.01,
                delta=1.0
            )
        
        problem_config["function"] = lambda weights: evaluation_func_with_logger(weights)[0]
        
        # executar metaheuristica com initial_scheme correto
        met = Metaheuristic(
            problem_config,
            heur_sequence,
            num_iterations=num_iterations,
            num_agents=num_agents,
            initial_scheme=initial_scheme
        )
        met.verbose = False  # silencioso para nao poluir output
        met.run()
        
        _, f_best = met.get_solution()
        
        # fechar logger
        logger.close()
        
        execution_time = time.time() - start_time
        
        # salvar metadados
        metadata = {
            "lambda_value": lambda_value,
            "operator_indices": operator_indices,
            "num_operators": len(selected_operators),
            "num_iterations": num_iterations,
            "num_agents": num_agents,
            "execution_time_seconds": execution_time,
            "best_fitness": float(f_best),
            "timestamp": datetime.now().isoformat(),
            "hh_result_file": hh_result_file
        }
        
        metadata_file = os.path.join(mh_output_dir, "metaheuristic_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # comparar HH vs MH
        comparison = compare_hh_vs_mh(lambda_folder, mh_output_dir)
        
        comparison_file = os.path.join(mh_output_dir, "comparison_hh_vs_mh.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        result['comparison'] = comparison
        
        # executar analise se solicitado
        if analyze:
            print(f"[{idx}/{total}] Executando analise para {lambda_name}...")
            
            # detectar fronteira eficiente
            import re
            sweep_dir = os.path.dirname(lambda_folder)
            sweep_name = os.path.basename(sweep_dir)
            match = re.search(r'port(\d+)', sweep_name.lower())
            
            frontier_file = None
            if match:
                port_num = match.group(1)
                frontier_file = f"portef{port_num}.txt"
                if not os.path.exists(frontier_file):
                    frontier_file = None
            
            try:
                # analise usa logs da MH
                analyze_portfolio_results(mh_output_dir, frontier_file, use_parallel=False)
                print(f"[{idx}/{total}] [SUCCESS] Analise concluida para {lambda_name}")
            except Exception as e:
                print(f"[{idx}/{total}] [WARNING] Erro na analise de {lambda_name}: {e}")
        
        result['success'] = True
        result['execution_time'] = execution_time
        
        print(f"[{idx}/{total}] [SUCCESS] {lambda_name} concluido em {execution_time:.1f}s")
        print(f"[{idx}/{total}]   Best fitness MH: {f_best:.6f}")
        if comparison['gaps']:
            gap = comparison['gaps']['relative_gap_percent']
            better = "MH melhor" if comparison['gaps']['mh_is_better'] else "HH melhor"
            print(f"[{idx}/{total}]   Gap vs HH: {gap:+.2f}% ({better})")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"[{idx}/{total}] [ERROR] {lambda_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def generate_aggregate_report(sweep_dir, results):
    """
    gera relatorio agregado comparando HH vs MH em todos os lambdas.
    """
    import pandas as pd
    
    print("\n" + "="*70)
    print("GERANDO RELATORIO AGREGADO")
    print("="*70)
    
    report_data = []
    
    for result in results:
        if result['success'] and result['comparison']:
            comp = result['comparison']
            
            row = {
                'lambda_value': comp['lambda_value'],
                'hh_best_fitness': comp['hh_metrics'].get('best_fitness'),
                'mh_best_fitness': comp['mh_metrics'].get('best_fitness'),
                'gap_absolute': comp['gaps'].get('absolute_gap'),
                'gap_percent': comp['gaps'].get('relative_gap_percent'),
                'mh_is_better': comp['gaps'].get('mh_is_better'),
                'hh_evaluations': comp['hh_metrics'].get('total_evaluations'),
                'mh_evaluations': comp['mh_metrics'].get('total_evaluations'),
                'hh_fitness_mean': comp['hh_metrics'].get('fitness_mean'),
                'mh_fitness_mean': comp['mh_metrics'].get('fitness_mean'),
                'hh_fitness_std': comp['hh_metrics'].get('fitness_std'),
                'mh_fitness_std': comp['mh_metrics'].get('fitness_std')
            }
            
            report_data.append(row)
    
    if not report_data:
        print("[WARNING] Nenhum dado para relatorio")
        return
    
    df_report = pd.DataFrame(report_data)
    
    # salvar CSV
    report_file = os.path.join(sweep_dir, "hh_vs_mh_comparison.csv")
    df_report.to_csv(report_file, index=False)
    print(f"[SUCCESS] Relatorio salvo: {report_file}")
    
    # estatisticas gerais
    print("\n" + "="*70)
    print("ESTATISTICAS GERAIS")
    print("="*70)
    
    mh_wins = df_report['mh_is_better'].sum()
    total = len(df_report)
    
    print(f"[INFO] Total de lambdas processados: {total}")
    print(f"[INFO] MH melhor que HH: {mh_wins}/{total} ({100*mh_wins/total:.1f}%)")
    print(f"[INFO] HH melhor que MH: {total-mh_wins}/{total} ({100*(total-mh_wins)/total:.1f}%)")
    print(f"\n[INFO] Gap medio (MH - HH): {df_report['gap_percent'].mean():.2f}%")
    print(f"[INFO] Gap mediano: {df_report['gap_percent'].median():.2f}%")
    print(f"[INFO] Gap std: {df_report['gap_percent'].std():.2f}%")
    print(f"[INFO] Melhor gap (MH): {df_report['gap_percent'].min():.2f}%")
    print(f"[INFO] Pior gap (MH): {df_report['gap_percent'].max():.2f}%")
    
    # estatisticas de fitness
    print(f"\n[INFO] Fitness medio HH: {df_report['hh_fitness_mean'].mean():.6f}")
    print(f"[INFO] Fitness medio MH: {df_report['mh_fitness_mean'].mean():.6f}")
    
    return df_report


def main():
    parser = argparse.ArgumentParser(
        description='executa metaheuristicas com operadores fixos da HH em cada lambda do sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemplos de uso:
  python batch_metaheuristic_executor.py <sweep_dir> --instance port1.txt --cardinality 10 --collection default_portfolio.txt --iterations 1000
  python batch_metaheuristic_executor.py <sweep_dir> --instance port1.txt --cardinality 10 --collection default_portfolio.txt --iterations 1000 --workers 4
  python batch_metaheuristic_executor.py <sweep_dir> --instance port1.txt --cardinality 10 --collection default_portfolio.txt --iterations 1000 --analyze
        """
    )
    
    parser.add_argument(
        'sweep_dir',
        help='diretorio do lambda sweep'
    )
    
    parser.add_argument(
        '--instance', '-i',
        required=True,
        help='arquivo da instancia (ex: port1.txt)'
    )
    
    parser.add_argument(
        '--cardinality', '-k',
        type=int,
        required=True,
        help='restricao de cardinalidade (0 = sem restricao)'
    )
    
    parser.add_argument(
        '--collection', '-c',
        required=True,
        help='arquivo de colecao de operadores (ex: default_portfolio.txt)'
    )
    
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        required=True,
        help='numero de iteracoes da metaheuristica'
    )
    
    parser.add_argument(
        '--agents', '-a',
        type=int,
        default=None,
        help='numero de agentes (default: usa mesmo da HH se disponivel, senao 30)'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='executar analyze_portfolio_results automaticamente'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='numero de processos paralelos (default: metade dos CPUs disponiveis)'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='limita o numero de lambdas a processar (util para testes)'
    )
    
    args = parser.parse_args()
    
    # calcular workers default
    if args.workers is None:
        args.workers = max(1, os.cpu_count() // 2)
    
    # validacoes
    if not os.path.isdir(args.sweep_dir):
        print(f"[ERROR] Diretorio nao encontrado: {args.sweep_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.instance):
        print(f"[ERROR] Arquivo de instancia nao encontrado: {args.instance}")
        sys.exit(1)
    
    if not os.path.exists(args.collection):
        print(f"[ERROR] Arquivo de colecao nao encontrado: {args.collection}")
        sys.exit(1)
    
    print("="*70)
    print("BATCH METAHEURISTIC EXECUTOR")
    print("="*70)
    print(f"[INFO] Sweep directory: {os.path.abspath(args.sweep_dir)}")
    print(f"[INFO] Instance: {args.instance}")
    print(f"[INFO] Cardinality: {'No restriction' if args.cardinality == 0 else args.cardinality}")
    print(f"[INFO] Collection: {args.collection}")
    print(f"[INFO] Iterations: {args.iterations}")
    print(f"[INFO] Workers: {args.workers}")
    print(f"[INFO] Analyze: {'Yes' if args.analyze else 'No'}")
    
    # Import dentro da main para evitar problemas no spawn
    from portfolio_utils.instance_reader import read_or_library_instance
    
    # carregar dados da instancia
    print("\n[PROCESS] Carregando instancia...")
    instance_data = read_or_library_instance(args.instance)
    k = None if args.cardinality == 0 else args.cardinality
    print(f"[INFO] Numero de ativos: {instance_data['n_assets']}")
    
    # encontrar pastas lambda
    lambda_folders = find_lambda_folders(args.sweep_dir)
    
    if not lambda_folders:
        print(f"[ERROR] Nenhuma pasta lambda_* encontrada em {args.sweep_dir}")
        sys.exit(1)
    
    print(f"[INFO] Encontradas {len(lambda_folders)} pastas lambda")
    
    # aplicar limite se especificado
    if args.limit:
        lambda_folders = lambda_folders[:args.limit]
        print(f"[INFO] Limitando a {args.limit} pastas (--limit)")
    
    # determinar numero de agentes
    num_agents = args.agents
    if num_agents is None:
        # tentar detectar do primeiro lambda
        first_config = load_hh_config_from_lambda(lambda_folders[0])
        if first_config and 'num_agents' in first_config:
            num_agents = first_config['num_agents']
            print(f"[INFO] Numero de agentes detectado da HH: {num_agents}")
        else:
            num_agents = 30
            print(f"[INFO] Numero de agentes (default): {num_agents}")
    else:
        print(f"[INFO] Numero de agentes (CLI): {num_agents}")
    
    # preparar parametros para workers
    params_list = [
        (folder, instance_data, k, args.collection, args.iterations, 
         num_agents, args.analyze, i, len(lambda_folders))
        for i, folder in enumerate(lambda_folders, 1)
    ]
    
    # executar em paralelo
    print("\n" + "="*70)
    print(f"INICIANDO PROCESSAMENTO COM {args.workers} WORKERS")
    print("="*70)
    
    start_time = time.time()
    results = []
    
    try:
        with Pool(processes=args.workers) as pool:
            # chunksize=1 garante distribuição uniforme
            results = pool.map(execute_metaheuristic_for_lambda, params_list, chunksize=1)
        
        print("\n[INFO] Processamento concluido!")
        
    except KeyboardInterrupt:
        print("\n[WARNING] Processamento interrompido pelo usuario (Ctrl+C)")
    
    # relatorio final
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("RELATORIO FINAL")
    print("="*70)
    
    success_count = sum(1 for r in results if r['success'])
    failed_count = len(results) - success_count
    
    print(f"[INFO] Total processado: {len(results)}")
    print(f"[SUCCESS] Sucessos: {success_count}")
    print(f"[ERROR] Falhas: {failed_count}")
    print(f"[TIME] Tempo total: {total_time/60:.1f}min")
    
    if len(results) > 0:
        print(f"[TIME] Tempo medio por lambda: {total_time/len(results):.1f}s")
    
    # gerar relatorio agregado
    if success_count > 0:
        generate_aggregate_report(args.sweep_dir, results)
    
    print("\n[PROCESS] Execucao finalizada!")


if __name__ == "__main__":
    main()
