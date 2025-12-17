#!/usr/bin/env python3
"""
script para executar analyze_portfolio_results em cada subpasta lambda de um sweep.
itera por todas as pastas lambda_X.XXXX dentro do diretorio de sweep e gera
analises individuais (pareto, metricas, graficos) para cada uma.

tambem pode agregar a fronteira eficiente unica de todo o sweep.
"""

import os
import sys
import glob
import argparse
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# importa a funcao de analise
try:
    from portfolio_utils.portfolio_analyzer import analyze_portfolio_results
except ImportError:
    print("[ERROR] Nao foi possivel importar portfolio_analyzer")
    print("   Certifique-se de estar no diretorio raiz do projeto")
    sys.exit(1)


def find_lambda_folders(sweep_dir):
    """
    encontra todas as subpastas lambda_X.XXXX dentro do diretorio de sweep.
    
    args:
        sweep_dir: diretorio do sweep
    
    returns:
        list: lista de caminhos das pastas lambda ordenadas
    """
    pattern = os.path.join(sweep_dir, "lambda_*")
    lambda_folders = glob.glob(pattern)
    
    # filtra apenas diretorios
    lambda_folders = [f for f in lambda_folders if os.path.isdir(f)]
    
    # ordena por valor de lambda
    lambda_folders.sort(key=lambda x: float(os.path.basename(x).split('_')[1]))
    
    return lambda_folders


def detect_instance_from_sweep(sweep_dir):
    """
    tenta detectar a instancia (port1, port2, etc) pelo nome da pasta sweep.
    
    args:
        sweep_dir: diretorio do sweep
    
    returns:
        str: instancia detectada (ex: 'port1') ou None
    """
    import re
    folder_name = os.path.basename(sweep_dir)
    match = re.search(r'port(\d+)', folder_name.lower())
    if match:
        return f"port{match.group(1)}"
    return None


def get_frontier_file(instance):
    """
    retorna o arquivo da fronteira eficiente para uma instancia.
    
    args:
        instance: instancia (ex: 'port1')
    
    returns:
        str: caminho do arquivo ou None
    """
    frontier_map = {
        'port1': 'portef1.txt',
        'port2': 'portef2.txt',
        'port3': 'portef3.txt',
        'port4': 'portef4.txt',
        'port5': 'portef5.txt'
    }
    
    frontier = frontier_map.get(instance)
    if frontier and os.path.exists(frontier):
        return frontier
    return None


def analyze_lambda_folder(params):
    """
    executa analyze_portfolio_results em uma pasta lambda individual.
    funcao wrapper para multiprocessing.
    
    args:
        params: tupla (lambda_folder, frontier_file, skip_existing, index, total)
    
    returns:
        dict: {'success': bool, 'lambda_name': str, 'skipped': bool, 'time': float}
    """
    lambda_folder, frontier_file, skip_existing, idx, _ = params
    lambda_name = os.path.basename(lambda_folder)
    
    # verifica se ja tem analise
    metrics_file = os.path.join(lambda_folder, "analysis_metrics.csv")
    if skip_existing and os.path.exists(metrics_file):
        return {
            'success': True,
            'lambda_name': lambda_name,
            'skipped': True,
            'time': 0,
            'index': idx
        }
    
    # verifica se tem execution_logs.csv
    logs_file = os.path.join(lambda_folder, "execution_logs.csv")
    if not os.path.exists(logs_file):
        return {
            'success': False,
            'lambda_name': lambda_name,
            'skipped': False,
            'time': 0,
            'index': idx,
            'error': 'execution_logs.csv nao encontrado'
        }
    
    try:
        start = time.time()
        
        # executa analise com use_parallel=False (estamos em um daemon process)
        # daemon processes nao podem criar child processes
        analyze_portfolio_results(lambda_folder, frontier_file, use_parallel=False)
        
        elapsed = time.time() - start
        return {
            'success': True,
            'lambda_name': lambda_name,
            'skipped': False,
            'time': elapsed,
            'index': idx
        }
        
    except Exception as e:
        return {
            'success': False,
            'lambda_name': lambda_name,
            'skipped': False,
            'time': 0,
            'index': idx,
            'error': str(e)
        }


def is_dominated(point, other_points):
    """
    verifica se um ponto eh dominado por algum outro ponto.
    minimizacao de risco, maximizacao de retorno.
    
    args:
        point: tupla (risk, return)
        other_points: array nx2 de (risk, return)
    
    returns:
        bool: True se dominado
    """
    risk, ret = point
    
    # um ponto domina outro se tem menor risco E maior retorno (ou igual em ambos mas melhor em pelo menos um)
    for other_risk, other_ret in other_points:
        if other_risk <= risk and other_ret >= ret:
            if other_risk < risk or other_ret > ret:  # pelo menos um estritamente melhor
                return True
    return False


def aggregate_efficient_frontier(sweep_dir, output_file='aggregated_frontier.csv'):
    """
    agrega todas as fronteiras de pareto dos lambdas em uma unica fronteira eficiente.
    
    args:
        sweep_dir: diretorio do sweep
        output_file: nome do arquivo de saida
    
    returns:
        pd.DataFrame: fronteira agregada ou None se falhou
    """
    print("\n" + "="*70)
    print("AGREGANDO FRONTEIRA EFICIENTE DO SWEEP")
    print("="*70)
    
    lambda_folders = find_lambda_folders(sweep_dir)
    
    if not lambda_folders:
        print("[ERROR] Nenhuma pasta lambda encontrada")
        return None
    
    all_points = []
    
    print(f"[INFO] Coletando pontos de {len(lambda_folders)} lambdas...")
    
    for lambda_folder in lambda_folders:
        pareto_file = os.path.join(lambda_folder, "population_pareto_frontier.csv")
        
        if not os.path.exists(pareto_file):
            lambda_name = os.path.basename(lambda_folder)
            print(f"[WARNING] {lambda_name}: pareto_frontier.csv nao encontrado")
            continue
        
        try:
            df = pd.read_csv(pareto_file)
            
            # adiciona coluna lambda_value
            lambda_value = float(os.path.basename(lambda_folder).split('_')[1])
            df['lambda_value'] = lambda_value
            
            all_points.append(df)
            
        except Exception as e:
            print(f"[ERROR] Falha ao ler {pareto_file}: {e}")
    
    if not all_points:
        print("[ERROR] Nenhum ponto coletado")
        return None
    
    # combinar todos os pontos
    combined = pd.concat(all_points, ignore_index=True)
    
    print(f"[INFO] Total de pontos coletados: {len(combined)}")
    
    # calcular fronteira de pareto (nao-dominados)
    print("[INFO] Calculando fronteira de Pareto agregada...")
    
    points_array = combined[['risk', 'expected_return']].values
    non_dominated_mask = []
    
    for i, point in enumerate(points_array):
        # verificar se eh dominado por algum outro ponto
        other_points = np.delete(points_array, i, axis=0)
        dominated = is_dominated(point, other_points)
        non_dominated_mask.append(not dominated)
    
    frontier = combined[non_dominated_mask].copy()
    frontier = frontier.sort_values('risk').reset_index(drop=True)
    
    print(f"[INFO] Fronteira agregada: {len(frontier)} pontos nao-dominados")
    
    # salvar
    output_path = os.path.join(sweep_dir, output_file)
    frontier.to_csv(output_path, index=False)
    
    print(f"[SUCCESS] Fronteira agregada salva: {output_path}")
    
    # estatisticas
    print(f"\n[STATS] Faixa de risco: [{frontier['risk'].min():.6f}, {frontier['risk'].max():.6f}]")
    print(f"[STATS] Faixa de retorno: [{frontier['expected_return'].min():.6f}, {frontier['expected_return'].max():.6f}]")
    print(f"[STATS] Lambdas representados: {frontier['lambda_value'].nunique()}")
    
    # detectar instancia e carregar fronteiras do problema
    instance = detect_instance_from_sweep(sweep_dir)
    frontier_file = None
    if instance:
        frontier_file = get_frontier_file(instance)
    
    # gerar grafico
    print("\n[INFO] Gerando grafico da fronteira agregada...")
    plot_aggregated_frontier(sweep_dir, combined, frontier, frontier_file)
    
    return frontier


def plot_aggregated_frontier(sweep_dir, all_points, frontier, frontier_file=None):
    """
    gera grafico mostrando todos os pontos e a fronteira agregada.
    
    args:
        sweep_dir: diretorio do sweep
        all_points: DataFrame com todos os pontos coletados
        frontier: DataFrame com a fronteira agregada
        frontier_file: arquivo da fronteira eficiente do problema (opcional)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    _, ax = plt.subplots(figsize=(12, 8))
    
    # carregar fronteira eficiente do problema se disponivel
    efficient_frontier = None
    if frontier_file and os.path.exists(frontier_file):
        try:
            df_eff = pd.read_csv(frontier_file, sep=r"\s+", header=None, names=["mean_return", "variance"])
            df_eff["std_dev"] = np.sqrt(df_eff["variance"])
            efficient_frontier = df_eff
            
            # plotar fronteira eficiente (azul)
            ax.plot(efficient_frontier["std_dev"], efficient_frontier["mean_return"], 
                    'b-', linewidth=2, label='Fronteira Eficiente', zorder=6)
            print(f"[INFO] Fronteira eficiente carregada: {len(efficient_frontier)} pontos")
        except Exception as e:
            print(f"[WARNING] Erro ao carregar fronteira eficiente: {e}")
    
    # plotar todos os pontos (cinza claro, pequenos)
    ax.scatter(all_points['risk'], all_points['expected_return'], 
               s=10, alpha=0.3, color='gray', label=f'Todos os pontos Pareto ({len(all_points)})')
    
    # plotar fronteira agregada (vermelho, maior)
    ax.scatter(frontier['risk'], frontier['expected_return'], 
               s=80, color='red', marker='o', edgecolors='darkred', 
               linewidths=1.5, label=f'Fronteira Agregada ({len(frontier)} pontos)', zorder=5)
    
    # linha conectando fronteira agregada
    frontier_sorted = frontier.sort_values('risk')
    ax.plot(frontier_sorted['risk'], frontier_sorted['expected_return'], 
            'r--', alpha=0.5, linewidth=1.5, zorder=4)
    
    ax.set_xlabel('Risco (Desvio Padrão)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Retorno Esperado', fontsize=12, fontweight='bold')
    ax.set_title('Fronteira Eficiente Agregada do Lambda Sweep', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # salvar
    output_path = os.path.join(sweep_dir, 'aggregated_frontier.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Grafico salvo: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='executa analyze_portfolio_results em cada lambda de um sweep (paralelo)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemplos de uso:
  python batch_analyze_lambda_sweep.py <sweep_dir>
  python batch_analyze_lambda_sweep.py 20251126_port1_random_lambda_sweep_50
  python batch_analyze_lambda_sweep.py <sweep_dir> --frontier portef2.txt
  python batch_analyze_lambda_sweep.py <sweep_dir> --skip-existing
  python batch_analyze_lambda_sweep.py <sweep_dir> --limit 10
  python batch_analyze_lambda_sweep.py <sweep_dir> --aggregate-frontier
  python batch_analyze_lambda_sweep.py <sweep_dir> --workers 4
        """
    )
    
    parser.add_argument(
        'sweep_dir',
        help='diretorio do lambda sweep'
    )
    
    parser.add_argument(
        '--frontier', '-f',
        help='arquivo da fronteira eficiente (auto-detecta se nao especificado)'
    )
    
    parser.add_argument(
        '--skip-existing', '-s',
        action='store_true',
        help='pula pastas que ja tem analise_metricas.csv'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='limita o numero de pastas lambda a processar (util para testes)'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='mostra o que seria executado sem executar'
    )
    
    parser.add_argument(
        '--aggregate-frontier', '-a',
        action='store_true',
        help='gera fronteira eficiente agregada do sweep completo'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=cpu_count() // 2,
        help=f'numero de processos paralelos (default: {cpu_count() // 2})'
    )
    
    args = parser.parse_args()
    
    # validar diretorio
    if not os.path.isdir(args.sweep_dir):
        print(f"[ERROR] Diretorio nao encontrado: {args.sweep_dir}")
        sys.exit(1)
    
    print("="*70)
    print("BATCH ANALYZE LAMBDA SWEEP (PARALELO)")
    print("="*70)
    print(f"[INFO] Diretorio: {os.path.abspath(args.sweep_dir)}")
    
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
    
    # determinar arquivo de fronteira
    frontier_file = args.frontier
    if not frontier_file:
        instance = detect_instance_from_sweep(args.sweep_dir)
        if instance:
            frontier_file = get_frontier_file(instance)
            if frontier_file:
                print(f"[INFO] Fronteira detectada: {frontier_file} (instancia: {instance})")
            else:
                print(f"[WARNING] Fronteira nao encontrada para {instance}")
        else:
            print("[WARNING] Nao foi possivel detectar instancia do nome da pasta")
    
    if not frontier_file:
        print("[ERROR] Fronteira eficiente nao especificada e nao pode ser auto-detectada")
        print("   Use --frontier <arquivo> para especificar manualmente")
        sys.exit(1)
    
    if not os.path.exists(frontier_file):
        print(f"[ERROR] Arquivo de fronteira nao encontrado: {frontier_file}")
        sys.exit(1)
    
    print(f"[INFO] Usando fronteira: {frontier_file}")
    
    if args.skip_existing:
        print("[INFO] Modo skip-existing ativado")
    
    print(f"[INFO] Usando {args.workers} workers paralelos")
    
    if args.dry_run:
        print("[INFO] Modo DRY RUN - apenas simulacao")
        print("\n[DRY RUN] Pastas que seriam processadas:")
        for folder in lambda_folders:
            print(f"   - {os.path.basename(folder)}")
        sys.exit(0)
    
    # preparar parametros para workers
    params_list = [
        (folder, frontier_file, args.skip_existing, i, len(lambda_folders))
        for i, folder in enumerate(lambda_folders, 1)
    ]
    
    # processar em paralelo
    print("\n" + "="*70)
    print("INICIANDO PROCESSAMENTO PARALELO")
    print("="*70 + "\n")
    
    stats = {
        'total': len(lambda_folders),
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    start_time = time.time()
    
    try:
        with Pool(processes=args.workers) as pool:
            results = pool.map(analyze_lambda_folder, params_list)
        
        # processar resultados
        for result in results:
            if result['success']:
                if result['skipped']:
                    stats['skipped'] += 1
                    print(f"[{result['index']}/{stats['total']}] [SKIP] {result['lambda_name']}: analise ja existe")
                else:
                    stats['success'] += 1
                    print(f"[{result['index']}/{stats['total']}] [SUCCESS] {result['lambda_name']} concluido em {result['time']:.1f}s")
            else:
                stats['failed'] += 1
                error_msg = result.get('error', 'erro desconhecido')
                print(f"[{result['index']}/{stats['total']}] [ERROR] {result['lambda_name']}: {error_msg}")
    
    except KeyboardInterrupt:
        print("\n[WARNING] Processamento interrompido pelo usuario (Ctrl+C)")
        print("[INFO] Alguns lambdas podem nao ter sido processados")
    
    # relatorio final
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("RELATORIO FINAL")
    print("="*70)
    print(f"[INFO] Total processado: {stats['total']}")
    print(f"[SUCCESS] Sucessos: {stats['success']}")
    print(f"[ERROR] Falhas: {stats['failed']}")
    print(f"[SKIP] Pulados: {stats['skipped']}")
    print(f"[TIME] Tempo total: {total_time/60:.1f}min")
    
    if stats['total'] > 0:
        print(f"[TIME] Tempo medio por pasta: {total_time/stats['total']:.1f}s")
    
    if stats['success'] > 0:
        print("\n[INFO] Analises concluidas com sucesso!")
        print("[INFO] Verifique os arquivos gerados em cada pasta lambda_*:")
        print("   - analysis_metrics.csv")
        print("   - population_pareto_frontier.csv")
        print("   - population_best_efficient.csv")
        print("   - population_best_pareto.csv")
        print("   - graficos PNG")
    
    if stats['failed'] > 0:
        print(f"\n[WARNING] {stats['failed']} pasta(s) falharam. Verifique os erros acima.")
    
    # agregar fronteira se solicitado
    if args.aggregate_frontier:
        aggregate_efficient_frontier(args.sweep_dir)


if __name__ == "__main__":
    main()
