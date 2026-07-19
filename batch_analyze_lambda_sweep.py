#!/usr/bin/env python3
"""
Análise consolidada de um lambda sweep.

Suporta arquivos em formato CSV e Parquet.
Em vez de analisar cada lambda individualmente, este script:
1. Carrega todos os population_pareto_frontier (CSV ou Parquet) de cada lambda
2. Combina em uma única população agregada
3. Calcula pareto sobre essa população
4. Aplica métricas de qualidade:
   - Erro de interpolação (Chang et al. 2000)
   - IGD+
   - Hipervolume
   - Cardinalidade
5. Gera gráficos e saídas consolidadas

Resultado: 1 análise consolidada da qualidade do sweep inteiro

⭐ NOVO: Suporte para lazy loading para datasets grandes
"""

import os
import sys
import glob
import argparse
import time
import gc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# importa as funções de análise
try:
    from portfolio_utils.portfolio_analyzer import (
        calculate_pareto_frontier,
        calculate_pareto_frontier_lazy,
        calculate_igd_plus,
        calculate_igd_plus_parallel,
        load_efficient_frontier,
        calculate_interpolation_errors,
        calculate_hypervolume_2d,
        save_metrics_to_csv,
        save_filtered_populations,
        _plot_efficient_vs_all,
        _plot_efficient_vs_pareto,
        _plot_cardinalidade_histogram,
    )
except ImportError:
    print("[ERROR] Nao foi possivel importar portfolio_analyzer")
    sys.exit(1)


def _load_dataframe_flexible(folder, filename_base):
    """
    Carrega um arquivo CSV ou Parquet.
    Tenta carregar filename_base.parquet primeiro, depois filename_base.csv
    """
    parquet_path = os.path.join(folder, f"{filename_base}.parquet")
    csv_path = os.path.join(folder, f"{filename_base}.csv")
    
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return None


def find_lambda_folders(sweep_dir):
    """
    Encontra todas as subpastas lambda_X.XXXX dentro do diretório de sweep.
    """
    pattern = os.path.join(sweep_dir, "lambda_*")
    lambda_folders = glob.glob(pattern)
    
    # Filtra apenas diretórios
    lambda_folders = [f for f in lambda_folders if os.path.isdir(f)]
    
    # Ordena por valor de lambda
    lambda_folders.sort(key=lambda x: float(os.path.basename(x).split('_')[1]))
    
    return lambda_folders


def detect_instance_from_sweep(sweep_dir):
    """
    Tenta detectar a instância pelo nome da pasta sweep.
    """
    import re
    folder_name = os.path.basename(sweep_dir)
    match = re.search(r'port(\d+)', folder_name.lower())
    if match:
        return f"port{match.group(1)}"
    return None


def get_frontier_file(instance):
    """
    Retorna o arquivo da fronteira eficiente para uma instância.
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


def _plot_efficient_vs_all_consolidated(output_dir, df_logs, efficient_frontier, pareto_frontier=None):
    """
    ⭐ Versão SCATTER para análise consolidada (batch_analyze).
    
    Plota fronteira eficiente vs TODAS as soluções usando scatter.
    Diferente da versão hexbin usada em lambdas individuais.
    """
    plt.figure(figsize=(12, 8))
    
    print(f"[PLOT] Plotando {len(df_logs)} soluções consolidadas via scatter")
    
    # SCATTER: Todos os dados individuais para análise consolidada
    plt.scatter(df_logs["risk"], df_logs["expected_return"], 
               s=6, color='gray', alpha=0.6, label="Todas as Soluções", rasterized=True)
    
    # Destacar a fronteira de Pareto se fornecida
    if pareto_frontier is not None and len(pareto_frontier) > 0:
        plt.scatter(pareto_frontier["risk"], pareto_frontier["expected_return"], 
                   s=12, color='red', alpha=0.8, label="Fronteira de Pareto", zorder=4, 
                   edgecolors='darkred', linewidths=0.5)
    
    plt.plot(efficient_frontier["std_dev"], efficient_frontier["mean_return"], 
            color='blue', linewidth=2, label="Fronteira Eficiente")

    # Coletar todas as soluções especiais para detectar sobreposições
    special_solutions = []
    
    # Melhor Sharpe
    if 'sharpe' in df_logs.columns:
        best_sharpe = df_logs.loc[df_logs["sharpe"].idxmax()]
        special_solutions.append({
            'risk': best_sharpe["risk"],
            'return': best_sharpe["expected_return"],
            'color': 'gold',
            'label': f"Melhor Sharpe ({best_sharpe['sharpe']:.4f})",
            'type': 'sharpe'
        })
    
    # Menor risco
    best_risk = df_logs.loc[df_logs["risk"].idxmin()]
    special_solutions.append({
        'risk': best_risk["risk"],
        'return': best_risk["expected_return"],
        'color': 'cyan',
        'label': f"Menor Risco ({best_risk['risk']:.4f})",
        'type': 'risk'
    })
    
    # Maior retorno
    best_return = df_logs.loc[df_logs["expected_return"].idxmax()]
    special_solutions.append({
        'risk': best_return["risk"],
        'return': best_return["expected_return"],
        'color': 'orange',
        'label': f"Maior Retorno ({best_return['expected_return']:.4f})",
        'type': 'return'
    })
    
    # Melhor objetivo
    best_objective = df_logs.loc[df_logs["objective"].idxmin()]
    special_solutions.append({
        'risk': best_objective["risk"],
        'return': best_objective["expected_return"],
        'color': 'purple',
        'label': f"Melhor Objetivo ({best_objective['objective']:.4f})",
        'type': 'objective'
    })
    
    # Plotar soluções especiais com sistema de deslocamento para sobreposições
    plotted_positions = []
    offset_distance = 0.001
    
    for i, solution in enumerate(special_solutions):
        risk_pos = solution['risk']
        return_pos = solution['return']
        
        # Verificar se já existe uma solução plotada na mesma posição
        position_occupied = False
        for prev_pos in plotted_positions:
            if abs(prev_pos[0] - risk_pos) < 1e-10 and abs(prev_pos[1] - return_pos) < 1e-10:
                position_occupied = True
                break
        
        # Se a posição está ocupada, aplicar um pequeno deslocamento circular
        if position_occupied:
            import math
            angle = (i * 2 * math.pi) / len(special_solutions)
            risk_pos += offset_distance * math.cos(angle)
            return_pos += offset_distance * math.sin(angle)
        
        plt.scatter(risk_pos, return_pos, s=120, color=solution['color'], 
                   edgecolor='black', alpha=0.8, label=solution['label'], zorder=5)
        
        plotted_positions.append((risk_pos, return_pos))
    
    plt.title("Fronteira Eficiente vs Todas as Soluções (Análise Consolidada)")
    plt.xlabel("Risco (Desvio Padrão)")
    plt.ylabel("Retorno Esperado")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fronteira_eficiente_vs_todas.png"), dpi=150, bbox_inches='tight')
    
    # Garbage collection
    plt.close('all')
    import gc
    gc.collect()


def aggregate_pareto_populations(sweep_dir):
    """
    Agrega todas as fronteiras de pareto dos lambdas em uma única população.
    Suporta tanto CSV quanto Parquet.
    """
    print("\n" + "="*70)
    print("AGREGANDO POPULAÇÃO PARETO DE TODOS OS LAMBDAS")
    print("="*70)
    
    lambda_folders = find_lambda_folders(sweep_dir)
    
    if not lambda_folders:
        print("[ERROR] Nenhuma pasta lambda encontrada")
        return None
    
    all_populations = []
    
    print(f"[INFO] Coletando population_pareto_frontier (CSV ou Parquet) de {len(lambda_folders)} lambdas...")
    
    for idx, lambda_folder in enumerate(lambda_folders, 1):
        lambda_name = os.path.basename(lambda_folder)
        
        # Tenta carregar Parquet ou CSV
        df = _load_dataframe_flexible(lambda_folder, "population_pareto_frontier")
        
        if df is None:
            print(f"  [{idx}/{len(lambda_folders)}] [SKIP] {lambda_name}: population_pareto_frontier não encontrado")
            continue
        
        try:
            # Adiciona coluna lambda_value para rastreabilidade
            lambda_value = float(os.path.basename(lambda_folder).split('_')[1])
            df['lambda_value'] = lambda_value
            
            all_populations.append(df)
            file_format = "parquet" if os.path.exists(os.path.join(lambda_folder, "population_pareto_frontier.parquet")) else "csv"
            print(f"  [{idx}/{len(lambda_folders)}] [OK] {lambda_name}: {len(df)} pontos ({file_format})")
            
        except Exception as e:
            print(f"  [{idx}/{len(lambda_folders)}] [ERROR] {lambda_name}: {e}")
    
    if not all_populations:
        print("[ERROR] Nenhuma população coletada")
        return None
    
    # Combinar todas as populações
    aggregated = pd.concat(all_populations, ignore_index=True)
    print(f"\n[INFO] Total de pontos coletados: {len(aggregated)}")
    print(f"[INFO] Lambdas representados: {aggregated['lambda_value'].nunique()}")
    
    return aggregated


def analyze_sweep_consolidated(sweep_dir, efficient_frontier_file, use_parallel=True, use_lazy=False, risk_free_rate=0.03):
    """
    Executa análise na população agregada de todos os lambdas.
    
    Args:
        use_lazy: Se True, usa lazy loading para datasets grandes
                 (reduz picos de RAM em 60-80%)
    """
    print("\n" + "="*70)
    print("ANALISANDO POPULAÇÃO CONSOLIDADA DO SWEEP")
    print("="*70)
    # 1. AGREGAR POPULAÇÕES
    aggregated_population = aggregate_pareto_populations(sweep_dir)
    if aggregated_population is None or len(aggregated_population) == 0:
        print("[ERROR] Falha ao agregar população")
        return None
    
    # 2. CRIAR DIRETÓRIO DE SAÍDA
    output_dir = os.path.join(sweep_dir, "consolidated_analysis")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Diretório de saída: {output_dir}")
    
    # 3. CARREGAR FRONTEIRA EFICIENTE
    print(f"\n[INFO] Carregando fronteira eficiente: {efficient_frontier_file}")
    efficient_frontier = load_efficient_frontier(efficient_frontier_file)
    if efficient_frontier is None:
        print("[ERROR] Falha ao carregar fronteira eficiente")
        return None
    print(f"[INFO] Fronteira eficiente: {len(efficient_frontier)} pontos")
    
    # 4. CALCULAR PARETO SOBRE A POPULAÇÃO AGREGADA
    print(f"\n[INFO] Calculando fronteira de Pareto sobre {len(aggregated_population)} pontos...")
    print(f"[INFO] Modo: {'LAZY (otimizado para RAM)' if use_lazy and len(aggregated_population) > 100000 else 'NORMAL'}")
    start_time = time.time()
    
    # ⭐ NOVO: Usar lazy loading para datasets muito grandes
    if use_lazy and len(aggregated_population) > 100000:
        # Salvar agregado temporariamente
        temp_parquet = os.path.join(output_dir, "_temp_aggregated.parquet")
        aggregated_population.to_parquet(temp_parquet, compression='snappy')
        pareto_frontier = calculate_pareto_frontier_lazy(temp_parquet, chunk_size=50000)
        os.remove(temp_parquet)
    else:
        pareto_frontier = calculate_pareto_frontier(aggregated_population)
    
    elapsed = time.time() - start_time
    print(f"[SUCCESS] Pareto calculada: {len(pareto_frontier)} pontos em {elapsed:.2f}s")
    
    # 5. CALCULAR ERROS DE INTERPOLAÇÃO (OTIMIZAÇÃO: uma única passada)
    print(f"\n[INTERPOLACAO] Calculando erros de interpolação...")
    aggregated_with_errors = calculate_interpolation_errors(aggregated_population, efficient_frontier)
    
    # OTIMIZAÇÃO: Filtra os erros já calculados usando índices ao invés de recalcular
    pareto_with_errors = aggregated_with_errors.loc[pareto_frontier.index]
    
    # 6. CALCULAR IGD+
    print(f"\n[IGD+] Calculando IGD+...")
    igd_plus = None
    if use_parallel:
        igd_plus = calculate_igd_plus_parallel(pareto_frontier, efficient_frontier)
    else:
        igd_plus = calculate_igd_plus(pareto_frontier, efficient_frontier)
    
    if igd_plus is not None:
        print(f" - IGD+: {igd_plus:.6e}")
    
    # 7. CALCULAR HIPERVOLUME
    print(f"\n[HIPERVOLUME] Calculando hipervolume...")
    
    # Referência: pior ponto possível (maior risco, menor retorno)
    ref_return = aggregated_population['expected_return'].min() - 0.001
    ref_risk = aggregated_population['risk'].max() + 0.001
    reference_point = [ref_return, ref_risk]
    
    hv_general = calculate_hypervolume_2d(
        aggregated_population[['expected_return', 'risk']].values,
        reference_point
    )
    hv_pareto = calculate_hypervolume_2d(
        pareto_frontier[['expected_return', 'risk']].values,
        reference_point
    )
    
    print(f" - Hipervolume geral: {hv_general:.6e}")
    print(f" - Hipervolume Pareto: {hv_pareto:.6e}")
    
    # 8. CALCULAR SHARPE
    print(f"\n[SHARPE] Calculando índice de Sharpe com taxa livre de risco = {risk_free_rate:.4f}")
    aggregated_population['sharpe'] = (aggregated_population['expected_return'] - risk_free_rate) / aggregated_population['risk']
    pareto_frontier['sharpe'] = (pareto_frontier['expected_return'] - risk_free_rate) / pareto_frontier['risk']
    
    best_sharpe_all = aggregated_population['sharpe'].max()
    best_sharpe_pareto = pareto_frontier['sharpe'].max()
    print(f" - Melhor Sharpe (geral): {best_sharpe_all:.6f}")
    print(f" - Melhor Sharpe (Pareto): {best_sharpe_pareto:.6f}")
    
    # 9. COMPUTAR MÉTRICAS CONSOLIDADAS (CHANG ET AL. 2000)
    print(f"\n[METRICS] Computando métricas consolidadas...")
    
    # Métricas de erro de interpolação
    avg_interp_all = aggregated_with_errors['percent_error'].mean()
    median_interp_all = aggregated_with_errors['percent_error'].median()
    min_interp_all = aggregated_with_errors['percent_error'].min()
    max_interp_all = aggregated_with_errors['percent_error'].max()
    avg_interp_pareto = pareto_with_errors['percent_error'].mean()
    median_interp_pareto = pareto_with_errors['percent_error'].median()
    min_interp_pareto = pareto_with_errors['percent_error'].min()
    max_interp_pareto = pareto_with_errors['percent_error'].max()
    
    # Contar pontos com erro válido (bracketing)
    valid_error_all = aggregated_with_errors['percent_error'].notna().sum()
    valid_error_pareto = pareto_with_errors['percent_error'].notna().sum()
    
    print(f" - Erro médio (todos): {avg_interp_all:.2f}%")
    print(f" - Erro mediano (todos): {median_interp_all:.2f}%")
    print(f" - Erro mín (todos): {min_interp_all:.2f}%")
    print(f" - Erro máx (todos): {max_interp_all:.2f}%")
    print(f" - Erro médio (Pareto): {avg_interp_pareto:.2f}%")
    print(f" - Erro mediano (Pareto): {median_interp_pareto:.2f}%")
    print(f" - Erro mín (Pareto): {min_interp_pareto:.2f}%")
    print(f" - Erro máx (Pareto): {max_interp_pareto:.2f}%")
    
    # 9. SALVAR MÉTRICAS
    metrics = {
        'total_aggregated_points': len(aggregated_population),
        'pareto_frontier_size': len(pareto_frontier),
        'avg_interpolation_error_all': avg_interp_all,
        'median_interpolation_error_all': median_interp_all,
        'min_interpolation_error_all': min_interp_all,
        'max_interpolation_error_all': max_interp_all,
        'avg_interpolation_error_pareto': avg_interp_pareto,
        'median_interpolation_error_pareto': median_interp_pareto,
        'min_interpolation_error_pareto': min_interp_pareto,
        'max_interpolation_error_pareto': max_interp_pareto,
        'valid_error_points_all': valid_error_all,
        'valid_error_points_pareto': valid_error_pareto,
        'igd_plus': igd_plus,
        'hypervolume_general': hv_general,
        'hypervolume_pareto': hv_pareto,
        'reference_point_return': ref_return,
        'reference_point_risk': ref_risk,
        'best_sharpe': best_sharpe_all,
        'best_sharpe_pareto': best_sharpe_pareto,
    }
    
    # Salvar métricas
    metrics_file = save_metrics_to_csv(output_dir, metrics)
    print(f"\n[SUCCESS] Métricas salvas: {metrics_file}")
    
    # 10. SALVAR POPULAÇÕES
    print(f"\n[INFO] Salvando populações...")
    save_filtered_populations(
        output_dir,
        aggregated_population=aggregated_population,
        pareto_frontier=pareto_frontier
    )
    print(f"[SUCCESS] Populações salvas em {output_dir}/population_*.csv")
    
    # 11. GERAR GRÁFICOS
    print(f"\n[PLOT] Gerando gráficos...")
    
    # Gráfico: população agregada vs fronteira eficiente (com SCATTER para análise consolidada)
    _plot_efficient_vs_all_consolidated(output_dir, aggregated_population, efficient_frontier, pareto_frontier)
    print(f"[SUCCESS] Gráfico fronteira_eficiente_vs_todas.png gerado")
    
    # Gráfico: Pareto vs fronteira eficiente
    _plot_efficient_vs_pareto(output_dir, efficient_frontier, pareto_frontier)
    print(f"[SUCCESS] Gráfico fronteira_eficiente_vs_pareto.png gerado")
    
    # Gráfico: histograma de cardinalidade
    _plot_cardinalidade_histogram(output_dir, aggregated_population)
    print(f"[SUCCESS] Gráfico cardinalidade_histogram.png gerado")
    
    print(f"\n" + "="*70)
    print("[SUCCESS] ANÁLISE CONSOLIDADA CONCLUÍDA!")
    print("="*70)
    print(f"[INFO] Resultados salvos em: {output_dir}")
    print(f"[INFO] Arquivos gerados:")
    print(f"   - analysis_metrics.csv")
    print(f"   - population_aggregated_population.csv")
    print(f"   - population_pareto_frontier.csv")
    print(f"   - fronteira_eficiente_vs_todas.png")
    print(f"   - fronteira_eficiente_vs_pareto.png")
    print(f"   - cardinalidade_histogram.png")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Análise consolidada de um lambda sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemplos de uso:
  python batch_analyze_lambda_sweep.py <sweep_dir>
  python batch_analyze_lambda_sweep.py 20251218_port2_sobol_lambda_sweep_50
  python batch_analyze_lambda_sweep.py <sweep_dir> --frontier portef2.txt
  python batch_analyze_lambda_sweep.py <sweep_dir> --sequential  (sem paralelização)
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
        '--sequential',
        action='store_true',
        help='executa análise sequencial (sem paralelização)'
    )
    
    parser.add_argument(
        '--lazy',
        action='store_true',
        help='⭐ NOVO: ativa lazy loading para datasets grandes (reduz RAM em 60-80%%)'
    )
    
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.00057,
        help='taxa livre de risco para cálculo do índice de Sharpe (padrão: 0.00057 = 3%% anual convertido para semanal)'
    )
    
    args = parser.parse_args()
    
    # validar diretorio
    if not os.path.isdir(args.sweep_dir):
        print(f"[ERROR] Diretorio nao encontrado: {args.sweep_dir}")
        sys.exit(1)
    
    print("="*70)
    print("BATCH ANALYZE LAMBDA SWEEP - ANÁLISE CONSOLIDADA")
    print("="*70)
    print(f"[INFO] Diretorio: {os.path.abspath(args.sweep_dir)}")
    
    # verificar pastas lambda
    lambda_folders = find_lambda_folders(args.sweep_dir)
    
    if not lambda_folders:
        print(f"[ERROR] Nenhuma pasta lambda_* encontrada em {args.sweep_dir}")
        sys.exit(1)
    
    print(f"[INFO] Encontradas {len(lambda_folders)} pastas lambda")
    
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
    
    # executar análise consolidada
    start_time = time.time()
    use_parallel = not args.sequential
    use_lazy = args.lazy
    
    if use_lazy:
        print("[INFO] ⭐ Lazy loading ATIVADO - RAM otimizada para datasets grandes")
    
    try:
        metrics = analyze_sweep_consolidated(
            args.sweep_dir, 
            frontier_file,
            use_parallel=use_parallel,
            use_lazy=use_lazy,
            risk_free_rate=args.risk_free_rate
        )
        
        if metrics is not None:
            total_time = time.time() - start_time
            print(f"\n[INFO] Tempo total de análise: {total_time:.1f}s")
        
    except Exception as e:
        print(f"[ERROR] Falha na análise consolidada: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
