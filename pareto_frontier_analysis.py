#!/usr/bin/env python3
"""
Script para análise de fronteira de Pareto agregada
Analisa múltiplas execuções de sweep e compara com fronteira eficiente de referência

Autor: Sistema de análise de portfólio
Data: 2025-08-03
"""

import os
import pandas as pd
import numpy as np
import argparse
import glob
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def find_pareto_frontier(points):
    """
    Encontra a fronteira de Pareto de um conjunto de pontos (retorno, risco).
    Usa a mesma implementação do portfolio_analyzer.
    """
    if len(points) == 0:
        return np.array([])
    
    points = np.array(points)
    
    # Ordenar por retorno (descendente) para otimização - igual ao portfolio_analyzer
    sorted_indices = np.argsort(-points[:, 1])  # Ordenar por retorno decrescente
    
    pareto_indices = []
    min_risk_so_far = float('inf')
    
    # Algoritmo otimizado O(n log n) igual ao portfolio_analyzer
    for idx in sorted_indices:
        current_risk = points[idx, 0]  # risco
        if current_risk < min_risk_so_far:
            pareto_indices.append(idx)
            min_risk_so_far = current_risk
    
    pareto_points = points[pareto_indices]
    
    # Ordena por risco crescente para visualização
    if len(pareto_points) > 0:
        sorted_risk_indices = np.argsort(pareto_points[:, 0])
        pareto_points = pareto_points[sorted_risk_indices]
    
    return pareto_points


def load_efficient_frontier(filepath):
    """
    Carrega a fronteira eficiente de um arquivo txt.
    Assume formato: retorno variância (separados por espaço ou tab)
    """
    try:
        # Usar o mesmo formato do portfolio_analyzer
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['mean_return', 'variance'])
        df['std_dev'] = np.sqrt(df['variance'])
        return df
    except Exception as e:
        print(f"Erro ao carregar fronteira eficiente de {filepath}: {e}")
        return None


def collect_pareto_data(sweep_folders):
    """
    Coleta todos os dados de population_pareto_frontier.csv das pastas de sweep.
    Retorna pontos no formato (risco, retorno) para compatibilidade com find_pareto_frontier.
    """
    all_points = []
    collected_files = []
    
    for sweep_folder in sweep_folders:
        print(f"Processando pasta de sweep: {sweep_folder}")
        
        # Procura por arquivos population_pareto_frontier.csv em subpastas
        pattern = os.path.join(sweep_folder, "**/population_pareto_frontier.csv")
        pareto_files = glob.glob(pattern, recursive=True)
        
        for file_path in pareto_files:
            try:
                df = pd.read_csv(file_path)
                # Extrai risco e retorno - invertendo a ordem para (risco, retorno)
                points = df[['risk', 'expected_return']].values
                all_points.extend(points)
                collected_files.append(file_path)
                print(f"  Coletados {len(points)} pontos de {file_path}")
            except Exception as e:
                print(f"  Erro ao processar {file_path}: {e}")
    
    print(f"\nTotal de arquivos processados: {len(collected_files)}")
    print(f"Total de pontos coletados: {len(all_points)}")
    
    return np.array(all_points), collected_files


def calculate_interpolation_error(heuristic_point, efficient_frontier):
    """
    Calcula o erro de interpolação entre um ponto da heurística e a fronteira eficiente.
    Usa a mesma implementação do portfolio_analyzer.
    
    Para um portfólio da heurística com (risco_real, retorno_real):
    1. Calcula erro percentual do risco (interpolando retorno esperado)
    2. Calcula erro percentual do retorno (interpolando risco esperado)
    3. Retorna o mínimo dos dois erros
    """
    risco_real, retorno_real = heuristic_point
    
    try:
        # Verificar se há dados suficientes para interpolação
        if len(efficient_frontier) < 2:
            return np.nan
        
        # Interpolação otimizada igual ao portfolio_analyzer
        interp_return = interp1d(
            efficient_frontier["std_dev"], 
            efficient_frontier["mean_return"], 
            kind='linear', 
            bounds_error=False, 
            fill_value="extrapolate"
        )
        interp_risk = interp1d(
            efficient_frontier["mean_return"], 
            efficient_frontier["std_dev"], 
            kind='linear', 
            bounds_error=False, 
            fill_value="extrapolate"
        )
        
        # Calcular retorno esperado dado o risco real
        retorno_esperado = interp_return(risco_real)
        
        # Calcular risco esperado dado o retorno real
        risco_esperado = interp_risk(retorno_real)
        
        # Cálculo dos erros com proteção contra divisão por zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # Evitar divisão por zero
            denominator_ret = abs(retorno_esperado) if abs(retorno_esperado) > 1e-10 else 1e-10
            denominator_risk = abs(risco_esperado) if abs(risco_esperado) > 1e-10 else 1e-10
            
            erro_retorno = abs(retorno_real - retorno_esperado) / denominator_ret
            erro_risco = abs(risco_real - risco_esperado) / denominator_risk
        
        # Retorna o mínimo dos dois erros (sem multiplicar por 100 para manter consistência)
        return min(erro_retorno, erro_risco)
        
    except Exception:
        return np.nan


def analyze_pareto_vs_efficient(pareto_points, efficient_frontier):
    """
    Analisa a fronteira de Pareto agregada comparada com a fronteira eficiente.
    """
    results = {
        'total_pontos_agregados': len(pareto_points),
        'pontos_fronteira_pareto': 0,
        'percentual_fronteira': 0.0,
        'erros_percentuais': [],
        'erro_medio': np.nan,
        'erro_mediana': np.nan,
        'erro_std': np.nan,
        'erro_min': np.nan,
        'erro_max': np.nan,
        'pontos_validos': 0
    }
    
    if len(pareto_points) == 0:
        return results
    
    # Encontra fronteira de Pareto
    pareto_frontier = find_pareto_frontier(pareto_points)
    results['pontos_fronteira_pareto'] = len(pareto_frontier)
    results['percentual_fronteira'] = (len(pareto_frontier) / len(pareto_points)) * 100
    
    if len(pareto_frontier) == 0 or efficient_frontier is None:
        return results
    
    # Calcula erros para cada ponto da fronteira de Pareto
    errors = []
    for point in pareto_frontier:
        error = calculate_interpolation_error(point, efficient_frontier)
        if not np.isnan(error):
            errors.append(error * 100)  # Converter para percentual para exibição
    
    results['erros_percentuais'] = errors
    results['pontos_validos'] = len(errors)
    
    if len(errors) > 0:
        results['erro_medio'] = np.mean(errors)
        results['erro_mediana'] = np.median(errors)
        results['erro_std'] = np.std(errors)
        results['erro_min'] = np.min(errors)
        results['erro_max'] = np.max(errors)
    
    return results


def save_results(results, efficient_frontier_path, sweep_folders, output_dir):
    """
    Salva os resultados da análise em arquivo.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Arquivo de estatísticas
    stats_file = os.path.join(output_dir, "pareto_statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("ESTATÍSTICAS DE ANÁLISE DE PARETO\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DADOS DA ANÁLISE:\n")
        f.write(f"Total de pontos agregados: {results['total_pontos_agregados']}\n")
        f.write(f"Soluções não dominadas (Fronteira de Pareto): {results['pontos_fronteira_pareto']}\n")
        f.write(f"Percentual de soluções na fronteira: {results['percentual_fronteira']:.2f}%\n\n")
        
        f.write("ESTATÍSTICAS DE ERRO (apenas fronteira de Pareto):\n")
        f.write(f"Erro Percentual Médio: {results['erro_medio']:.4f}%\n")
        f.write(f"Erro Percentual Mediana: {results['erro_mediana']:.4f}%\n")
        f.write(f"Quantidade de pontos de Pareto: {results['pontos_fronteira_pareto']}\n")
        f.write(f"Desvio Padrão do Erro: {results['erro_std']:.4f}%\n")
        f.write(f"Erro Mínimo: {results['erro_min']:.4f}%\n")
        f.write(f"Erro Máximo: {results['erro_max']:.4f}%\n")
        f.write(f"Pontos Totais Processados: {results['pontos_fronteira_pareto']}\n")
        f.write(f"Pontos Válidos: {results['pontos_validos']}\n\n")
        
        f.write("CONFIGURAÇÃO DA ANÁLISE:\n")
        f.write(f"Fronteira eficiente: {efficient_frontier_path}\n")
        f.write("Pastas de sweep analisadas:\n")
        for folder in sweep_folders:
            f.write(f"  - {folder}\n")
    
    print(f"Estatísticas salvas em: {stats_file}")
    return stats_file


def plot_comparison(pareto_points, efficient_frontier, pareto_frontier, output_dir):
    """
    Cria gráficos de comparação entre fronteira de Pareto e fronteira eficiente.
    """
    plt.style.use('default')
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Todos os pontos vs fronteiras
    if len(pareto_points) > 0:
        ax1.scatter(pareto_points[:, 0], pareto_points[:, 1], 
                   alpha=0.3, s=10, color='lightblue', label='Todos os pontos')
    
    if len(pareto_frontier) > 0:
        ax1.scatter(pareto_frontier[:, 0], pareto_frontier[:, 1], 
                   color='red', s=30, label='Fronteira de Pareto', zorder=5)
        ax1.plot(pareto_frontier[:, 0], pareto_frontier[:, 1], 
                color='red', linewidth=2, alpha=0.7, zorder=4)
    
    if efficient_frontier is not None:
        ax1.plot(efficient_frontier['std_dev'], efficient_frontier['mean_return'], 
                color='green', linewidth=3, label='Fronteira Eficiente', zorder=6)
    
    ax1.set_xlabel('Risco')
    ax1.set_ylabel('Retorno')
    ax1.set_title('Comparação de Fronteiras')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Apenas fronteiras
    if len(pareto_frontier) > 0:
        ax2.scatter(pareto_frontier[:, 0], pareto_frontier[:, 1], 
                   color='red', s=30, label='Fronteira de Pareto', zorder=5)
        ax2.plot(pareto_frontier[:, 0], pareto_frontier[:, 1], 
                color='red', linewidth=2, alpha=0.7, zorder=4)
    
    if efficient_frontier is not None:
        ax2.plot(efficient_frontier['std_dev'], efficient_frontier['mean_return'], 
                color='green', linewidth=3, label='Fronteira Eficiente', zorder=6)
    
    ax2.set_xlabel('Risco')
    ax2.set_ylabel('Retorno')
    ax2.set_title('Fronteiras Comparadas')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salva o gráfico
    plot_file = os.path.join(output_dir, "pareto_comparison.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico salvo em: {plot_file}")


def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(description='Análise de fronteira de Pareto agregada')
    parser.add_argument('efficient_frontier', 
                       help='Caminho para arquivo txt da fronteira eficiente')
    parser.add_argument('sweep_folders', nargs='+',
                       help='Caminhos para as pastas de sweep')
    parser.add_argument('--output', '-o', default='./test_results',
                       help='Diretório de saída (padrão: ./test_results)')
    
    args = parser.parse_args()
    
    print("=== ANÁLISE DE FRONTEIRA DE PARETO AGREGADA ===\n")
    
    # Carrega fronteira eficiente
    print(f"Carregando fronteira eficiente de: {args.efficient_frontier}")
    efficient_frontier = load_efficient_frontier(args.efficient_frontier)
    
    if efficient_frontier is None:
        print("Erro: Não foi possível carregar a fronteira eficiente.")
        return
    
    print(f"Fronteira eficiente carregada: {len(efficient_frontier)} pontos")
    
    # Coleta dados das pastas de sweep
    print(f"\nColetando dados de {len(args.sweep_folders)} pastas de sweep...")
    all_points, _ = collect_pareto_data(args.sweep_folders)
    
    if len(all_points) == 0:
        print("Erro: Nenhum ponto foi coletado das pastas de sweep.")
        return
    
    # Encontra fronteira de Pareto agregada
    print(f"\nEncontrando fronteira de Pareto dos {len(all_points)} pontos coletados...")
    pareto_frontier = find_pareto_frontier(all_points)
    print(f"Fronteira de Pareto encontrada: {len(pareto_frontier)} pontos")
    
    # Analisa comparação com fronteira eficiente
    print("\nAnalisando comparação com fronteira eficiente...")
    results = analyze_pareto_vs_efficient(all_points, efficient_frontier)
    
    # Salva resultados
    print("\nSalvando resultados...")
    save_results(results, args.efficient_frontier, args.sweep_folders, args.output)
    
    # Cria gráficos
    print("\nCriando gráficos...")
    plot_comparison(all_points, efficient_frontier, pareto_frontier, args.output)
    
    # Resumo final
    print("\n=== RESUMO DOS RESULTADOS ===")
    print(f"Total de pontos coletados: {results['total_pontos_agregados']}")
    print(f"Pontos na fronteira de Pareto: {results['pontos_fronteira_pareto']}")
    print(f"Percentual na fronteira: {results['percentual_fronteira']:.2f}%")
    
    if not np.isnan(results['erro_medio']):
        print(f"Erro percentual médio: {results['erro_medio']:.4f}%")
        print(f"Erro percentual mediana: {results['erro_mediana']:.4f}%")
        print(f"Pontos válidos para análise: {results['pontos_validos']}")
    else:
        print("Não foi possível calcular erros (pontos fora do domínio da fronteira eficiente)")
    
    print(f"\nResultados salvos em: {args.output}")


if __name__ == "__main__":
    main()
