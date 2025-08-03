#!/usr/bin/env python3
"""
Analisador de dados para resultados de otimização de portfólio.
Calcula métricas de performance, fronteiras de Pareto e gera visualizações.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
from multiprocessing import Pool, cpu_count
import time
warnings.filterwarnings('ignore')

# Constantes para labels dos gráficos
EFFICIENT_FRONTIER_LABEL = "Fronteira Eficiente"
RISK_LABEL = "Risco (Desvio Padrão)"
RETURN_LABEL = "Retorno Esperado"

def load_efficient_frontier(file_path):
    """Carrega a fronteira eficiente de um arquivo."""
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=["mean_return", "variance"])
        df["std_dev"] = np.sqrt(df["variance"])
        return df
    except Exception as e:
        print(f"Erro ao carregar fronteira eficiente: {e}")
        return None

def calculate_pareto_frontier_parallel(df_logs, n_processes=None):
    """Calcula a fronteira de Pareto das soluções - versão paralelizada para datasets grandes."""
    n_solutions = len(df_logs)
    
    # Para datasets pequenos, usar método sequencial otimizado
    if n_solutions < 10000:
        return calculate_pareto_frontier(df_logs)
    
    print(f"🚀 Calculando fronteira de Pareto com paralelização para {n_solutions} soluções")
    
    if n_processes is None:
        n_processes = min(cpu_count(), 6)  # Limite para evitar overhead
    
    solutions = df_logs[['expected_return', 'risk']].values
    indices = df_logs.index.values
    
    # Dividir em chunks para processamento paralelo
    chunk_size = max(1000, n_solutions // (n_processes * 2))
    chunks = []
    for i in range(0, n_solutions, chunk_size):
        end_idx = min(i + chunk_size, n_solutions)
        chunks.append((solutions[i:end_idx], indices[i:end_idx], i))
    
    print(f"🔄 Processando {len(chunks)} chunks em {n_processes} processos")
    
    try:
        with Pool(processes=n_processes) as pool:
            chunk_results = pool.map(_find_pareto_in_chunk, chunks)
        
        # Combinar resultados e encontrar Pareto global
        all_pareto_indices = []
        for chunk_pareto_indices in chunk_results:
            all_pareto_indices.extend(chunk_pareto_indices)
        
        # Segunda passada: encontrar Pareto entre os candidatos
        if len(all_pareto_indices) > 1000:  # Se ainda há muitos candidatos
            candidates_df = df_logs.loc[all_pareto_indices]
            final_pareto = calculate_pareto_frontier(candidates_df)
        else:
            final_pareto = df_logs.loc[all_pareto_indices]
            final_pareto = calculate_pareto_frontier(final_pareto)
        
        print(f"✅ Fronteira de Pareto calculada: {len(final_pareto)} pontos")
        return final_pareto
        
    except Exception as e:
        print(f"⚠️ Erro no cálculo paralelo de Pareto: {e}")
        print("🔄 Fallback para método sequencial...")
        return calculate_pareto_frontier(df_logs)

def _find_pareto_in_chunk(chunk_data):
    """Encontra pontos Pareto dentro de um chunk."""
    solutions, indices, _ = chunk_data  # offset não é usado
    n = len(solutions)
    
    if n == 0:
        return []
    
    # Algoritmo otimizado para encontrar Pareto no chunk
    sorted_indices = np.argsort(-solutions[:, 0])  # Ordenar por retorno decrescente
    
    pareto_mask = np.zeros(n, dtype=bool)
    min_risk_so_far = float('inf')
    
    for i in sorted_indices:
        current_risk = solutions[i, 1]
        if current_risk < min_risk_so_far:
            pareto_mask[i] = True
            min_risk_so_far = current_risk
    
    return indices[pareto_mask].tolist()

def calculate_pareto_frontier(df_logs):
    """Calcula a fronteira de Pareto das soluções - versão otimizada."""
    solutions = df_logs[['expected_return', 'risk']].values
    n = len(solutions)
    
    if n == 0:
        return df_logs.iloc[0:0].copy()  # DataFrame vazio
    
    # Ordenar por retorno (descendente) para otimização
    sorted_indices = np.argsort(-solutions[:, 0])  # Ordenar por retorno decrescente
    
    pareto_indices = []
    min_risk_so_far = float('inf')
    
    # Algoritmo otimizado O(n log n) em vez de O(n²)
    for idx in sorted_indices:
        current_risk = solutions[idx, 1]
        if current_risk < min_risk_so_far:
            pareto_indices.append(idx)
            min_risk_so_far = current_risk
    
    pareto_df = df_logs.iloc[pareto_indices].copy().sort_values('risk')
    return pareto_df

def calculate_igd_plus_parallel(pareto_front, reference_front, n_processes=None):
    """Calcula o IGD+ entre a fronteira de Pareto e a fronteira de referência - versão paralelizada."""
    if reference_front is None or len(reference_front) == 0:
        return None
        
    pareto_points = pareto_front[['expected_return', 'risk']].values
    ref_points = reference_front[['mean_return', 'std_dev']].values
    
    if len(pareto_points) == 0:
        return float('inf')
    
    n_ref = len(ref_points)
    n_pareto = len(pareto_points)
    
    # Para datasets pequenos, usar método sequencial
    if n_ref * n_pareto < 50000:
        return calculate_igd_plus(pareto_front, reference_front)
    
    print(f"🚀 Calculando IGD+ paralelo: {n_ref} pontos de referência vs {n_pareto} pontos Pareto")
    
    if n_processes is None:
        n_processes = min(cpu_count(), 6)
    
    # Dividir pontos de referência em chunks
    chunk_size = max(100, n_ref // (n_processes * 2))
    chunks = []
    for i in range(0, n_ref, chunk_size):
        end_idx = min(i + chunk_size, n_ref)
        chunks.append((ref_points[i:end_idx], pareto_points))
    
    try:
        with Pool(processes=n_processes) as pool:
            chunk_results = pool.map(_calculate_igd_chunk, chunks)
        
        # Combinar resultados
        all_min_distances = []
        for chunk_distances in chunk_results:
            all_min_distances.extend(chunk_distances)
        
        return np.mean(all_min_distances)
        
    except Exception as e:
        print(f"⚠️ Erro no cálculo paralelo de IGD+: {e}")
        return calculate_igd_plus(pareto_front, reference_front)

def _calculate_igd_chunk(chunk_data):
    """Calcula IGD para um chunk de pontos de referência."""
    ref_chunk, pareto_points = chunk_data
    
    # Usar broadcasting para calcular distâncias
    ref_expanded = ref_chunk[:, np.newaxis, :]  # (n_ref_chunk, 1, 2)
    pareto_expanded = pareto_points[np.newaxis, :, :]  # (1, n_pareto, 2)
    
    # Calcular todas as distâncias euclidianas
    distances = np.sqrt(np.sum((ref_expanded - pareto_expanded)**2, axis=2))
    
    # Encontrar a distância mínima para cada ponto de referência no chunk
    min_distances = np.min(distances, axis=1)
    
    return min_distances.tolist()

def calculate_igd_plus(pareto_front, reference_front):
    """Calcula o IGD+ entre a fronteira de Pareto e a fronteira de referência - versão otimizada."""
    if reference_front is None or len(reference_front) == 0:
        return None
        
    pareto_points = pareto_front[['expected_return', 'risk']].values
    ref_points = reference_front[['mean_return', 'std_dev']].values
    
    if len(pareto_points) == 0:
        return float('inf')
    
    # Usar broadcasting para calcular todas as distâncias de uma vez
    # Reshape para permitir broadcasting: (n_ref, 1, 2) - (1, n_pareto, 2)
    ref_expanded = ref_points[:, np.newaxis, :]  # (n_ref, 1, 2)
    pareto_expanded = pareto_points[np.newaxis, :, :]  # (1, n_pareto, 2)
    
    # Calcular todas as distâncias euclidianas de uma vez
    distances = np.sqrt(np.sum((ref_expanded - pareto_expanded)**2, axis=2))  # (n_ref, n_pareto)
    
    # Encontrar a distância mínima para cada ponto de referência
    min_distances = np.min(distances, axis=1)
    
    return np.mean(min_distances)

def calculate_interpolation_errors(df_logs, efficient_frontier):
    """Calcula erros de interpolação para todos os pontos - versão otimizada."""
    if efficient_frontier is None:
        return df_logs
    
    df_logs = df_logs.copy()
    
    try:
        # Verificar se há dados suficientes para interpolação
        if len(efficient_frontier) < 2:
            print("⚠️ Fronteira eficiente tem poucos pontos para interpolação")
            return df_logs
        
        # Interpolação otimizada com tratamento de erros
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
        
        # Aplicar interpolação vetorizada
        df_logs["interp_ret_from_risk"] = interp_return(df_logs["risk"].values)
        df_logs["interp_risk_from_ret"] = interp_risk(df_logs["expected_return"].values)
        
        # Cálculo vetorizado dos erros com proteção contra divisão por zero
        with np.errstate(divide='ignore', invalid='ignore'):
            denominator_ret = np.abs(df_logs["interp_ret_from_risk"])
            denominator_risk = np.abs(df_logs["interp_risk_from_ret"])
            
            # Evitar divisão por zero
            denominator_ret = np.where(denominator_ret == 0, 1e-10, denominator_ret)
            denominator_risk = np.where(denominator_risk == 0, 1e-10, denominator_risk)
            
            df_logs["error_return"] = np.abs(df_logs["expected_return"] - df_logs["interp_ret_from_risk"]) / denominator_ret
            df_logs["error_risk"] = np.abs(df_logs["risk"] - df_logs["interp_risk_from_ret"]) / denominator_risk
        
        # Usar numpy para operação vetorizada
        df_logs["percent_error"] = np.minimum(df_logs["error_return"], df_logs["error_risk"])
        
        # Tratamento de valores inválidos de forma vetorizada
        df_logs["percent_error"] = np.where(
            np.isfinite(df_logs["percent_error"]), 
            df_logs["percent_error"], 
            np.nanmax(df_logs["percent_error"][np.isfinite(df_logs["percent_error"])]) * 2
        )
        
        # Se ainda há NaN, usar valor padrão
        if np.isnan(df_logs["percent_error"]).any():
            df_logs["percent_error"] = df_logs["percent_error"].fillna(1.0)
            
    except Exception as e:
        print(f"⚠️ Erro na interpolação: {e}")
        df_logs["percent_error"] = 1.0  # Valor padrão em caso de erro
    
    return df_logs

def get_best_solutions_for_frontier(df_logs, n_solutions=100):
    """Seleciona as n melhores soluções com menor erro percentual (método antigo - mantido para compatibilidade)."""
    if 'percent_error' not in df_logs.columns:
        return df_logs.nsmallest(n_solutions, 'objective')
    
    # Ordenar por erro percentual e pegar as melhores
    best_solutions = df_logs.nsmallest(min(n_solutions, len(df_logs)), 'percent_error')
    return best_solutions

def _process_frontier_batch_parallel(batch_data):
    """Função auxiliar para processar um batch de pontos da fronteira em paralelo."""
    (batch_frontier_risk, batch_frontier_return, solutions_risk, 
     solutions_return, solutions_error, solutions_indices, n_solutions_per_point) = batch_data
    
    n_solutions = len(solutions_risk)
    n_candidates = min(n_solutions_per_point * 2, n_solutions)
    selected_indices_set = set()
    
    # Processar todos os pontos deste batch
    for i in range(len(batch_frontier_risk)):
        # Calcular distâncias para este ponto da fronteira
        risk_diff = solutions_risk - batch_frontier_risk[i]
        return_diff = solutions_return - batch_frontier_return[i]
        distances = np.sqrt(risk_diff**2 + return_diff**2)
        
        # Encontrar candidatos mais próximos - corrigir para evitar erro de índice
        # argpartition precisa de k < len(array)
        k = min(n_candidates, len(distances) - 1) if len(distances) > 1 else 0
        if k > 0:
            closest_candidates_idx = np.argpartition(distances, k)[:n_candidates]
        else:
            closest_candidates_idx = np.arange(len(distances))
        
        # Entre os candidatos próximos, pegar os com menor erro
        candidate_errors = solutions_error[closest_candidates_idx]
        
        # Ordenar por erro e pegar os melhores
        best_error_order = np.argsort(candidate_errors)[:n_solutions_per_point]
        selected_for_this_point = closest_candidates_idx[best_error_order]
        
        # Adicionar ao conjunto de índices selecionados
        selected_indices_set.update(solutions_indices[selected_for_this_point])
    
    return list(selected_indices_set)

def get_best_solutions_per_frontier_point_parallel(df_logs, frontier_points, n_solutions_per_point=100, n_processes=None):
    """
    Versão paralelizada da seleção das n melhores soluções para cada ponto da fronteira.
    
    Args:
        df_logs: DataFrame com as soluções
        frontier_points: DataFrame com os pontos da fronteira
        n_solutions_per_point: Número de melhores soluções por ponto da fronteira
        n_processes: Número de processos paralelos (None = auto)
    
    Returns:
        DataFrame com todas as melhores soluções selecionadas
    """
    if 'percent_error' not in df_logs.columns:
        print("⚠️ Coluna 'percent_error' não encontrada, usando método alternativo")
        return df_logs.nsmallest(len(frontier_points) * n_solutions_per_point, 'objective')
    
    n_frontier_points = len(frontier_points)
    n_solutions = len(df_logs)
    
    # Determinar número de processos
    if n_processes is None:
        n_processes = min(cpu_count(), 8)  # Máximo 8 processos para evitar overhead
    
    print(f"🚀 Processamento paralelo: {n_processes} processos")
    print(f"🔍 Selecionando {n_solutions_per_point} melhores soluções para cada um dos {n_frontier_points} pontos da fronteira")
    
    # Converter para arrays numpy para otimização
    solutions_risk = df_logs['risk'].values
    solutions_return = df_logs['expected_return'].values
    solutions_error = df_logs['percent_error'].values
    solutions_indices = df_logs.index.values
    
    # Determinar o tipo de fronteira e extrair coordenadas
    if 'std_dev' in frontier_points.columns:  # Fronteira eficiente
        frontier_risk = frontier_points['std_dev'].values
        frontier_return = frontier_points['mean_return'].values
    else:  # Fronteira de Pareto
        frontier_risk = frontier_points['risk'].values
        frontier_return = frontier_points['expected_return'].values
    
    # Calcular tamanho do batch otimizado para paralelização
    # Considerando overhead de criação de processos vs. eficiência
    ideal_batch_size = max(1, n_frontier_points // (n_processes * 2))  # 2 batches por processo
    batch_size = min(100, max(10, ideal_batch_size))  # Entre 10 e 100
    
    print(f"🔄 Processando em {n_processes} processos paralelos com batches de {batch_size} pontos")
    
    # Preparar dados para processamento paralelo
    batch_data_list = []
    for batch_start in range(0, n_frontier_points, batch_size):
        batch_end = min(batch_start + batch_size, n_frontier_points)
        batch_frontier_risk = frontier_risk[batch_start:batch_end]
        batch_frontier_return = frontier_return[batch_start:batch_end]
        
        batch_data = (
            batch_frontier_risk, batch_frontier_return,
            solutions_risk, solutions_return, solutions_error, solutions_indices,
            n_solutions_per_point
        )
        batch_data_list.append(batch_data)
    
    # Executar processamento paralelo
    start_time = time.time()
    try:
        with Pool(processes=n_processes) as pool:
            results = pool.map(_process_frontier_batch_parallel, batch_data_list)
        
        # Combinar resultados de todos os batches
        selected_indices_set = set()
        for batch_result in results:
            selected_indices_set.update(batch_result)
        
        processing_time = time.time() - start_time
        print(f"⚡ Processamento paralelo concluído em {processing_time:.2f}s")
        
    except Exception as e:
        print(f"⚠️ Erro no processamento paralelo: {e}")
        print("🔄 Fallback para processamento sequencial...")
        return get_best_solutions_per_frontier_point(df_logs, frontier_points, n_solutions_per_point)
    
    # Converter conjunto para lista e criar DataFrame das soluções selecionadas
    selected_indices_list = list(selected_indices_set)
    combined_solutions = df_logs.loc[selected_indices_list].copy()
    
    print(f"✅ Selecionadas {len(combined_solutions)} soluções únicas")
    print(f"📊 Erro médio das soluções selecionadas: {combined_solutions['percent_error'].mean():.6e}")
    
    return combined_solutions

def get_best_solutions_per_frontier_point(df_logs, frontier_points, n_solutions_per_point=100):
    """
    Seleciona as n melhores soluções para cada ponto da fronteira baseado no erro de interpolação - versão otimizada com baixo uso de memória.
    
    Args:
        df_logs: DataFrame com as soluções
        frontier_points: DataFrame com os pontos da fronteira (eficiente ou Pareto)
        n_solutions_per_point: Número de melhores soluções por ponto da fronteira
    
    Returns:
        DataFrame com todas as melhores soluções selecionadas
    """
    if 'percent_error' not in df_logs.columns:
        print("⚠️ Coluna 'percent_error' não encontrada, usando método alternativo")
        return df_logs.nsmallest(len(frontier_points) * n_solutions_per_point, 'objective')
    
    n_frontier_points = len(frontier_points)
    print(f"🔍 Selecionando {n_solutions_per_point} melhores soluções para cada um dos {n_frontier_points} pontos da fronteira")
    
    # Converter para arrays numpy para otimização
    solutions_risk = df_logs['risk'].values
    solutions_return = df_logs['expected_return'].values
    solutions_error = df_logs['percent_error'].values
    solutions_indices = df_logs.index.values
    
    # Determinar o tipo de fronteira e extrair coordenadas
    if 'std_dev' in frontier_points.columns:  # Fronteira eficiente
        frontier_risk = frontier_points['std_dev'].values
        frontier_return = frontier_points['mean_return'].values
    else:  # Fronteira de Pareto
        frontier_risk = frontier_points['risk'].values
        frontier_return = frontier_points['expected_return'].values
    
    # Coletar índices das melhores soluções para cada ponto da fronteira
    selected_indices_set = set()
    n_candidates = min(n_solutions_per_point * 2, len(df_logs))  # Limitar candidatos para eficiência
    
    # Calcular tamanho do batch baseado na memória disponível
    # Estimativa: 8 bytes por float64, queremos usar no máximo ~100MB por batch
    max_memory_mb = 100
    max_elements_per_batch = (max_memory_mb * 1024 * 1024) // 8  # Elementos float64
    batch_size = max(1, min(500, max_elements_per_batch // len(df_logs)))  # Pelo menos 1, máximo 500
    
    print(f"🔄 Processando em batches de {batch_size} pontos da fronteira para otimizar memória")
    
    # Processamento em batches para evitar problemas de memória
    for batch_start in range(0, n_frontier_points, batch_size):
        batch_end = min(batch_start + batch_size, n_frontier_points)
        batch_frontier_risk = frontier_risk[batch_start:batch_end]
        batch_frontier_return = frontier_return[batch_start:batch_end]
        
        # Cálculo vetorizado apenas para este batch
        # Formato: (batch_size, n_solutions)
        risk_diff = solutions_risk[np.newaxis, :] - batch_frontier_risk[:, np.newaxis]
        return_diff = solutions_return[np.newaxis, :] - batch_frontier_return[:, np.newaxis]
        distances_matrix_batch = np.sqrt(risk_diff**2 + return_diff**2)
        
        # Processar cada ponto da fronteira no batch atual
        for batch_idx in range(len(batch_frontier_risk)):
            # Encontrar os candidatos mais próximos para este ponto da fronteira
            frontier_distances = distances_matrix_batch[batch_idx, :]
            # Corrigir para evitar erro de índice - argpartition precisa de k < len(array)
            k = min(n_candidates, len(frontier_distances) - 1) if len(frontier_distances) > 1 else 0
            if k > 0:
                closest_candidates_idx = np.argpartition(frontier_distances, k)[:n_candidates]
            else:
                closest_candidates_idx = np.arange(len(frontier_distances))
            
            # Entre os candidatos próximos, pegar os com menor erro
            candidate_errors = solutions_error[closest_candidates_idx]
            
            # Ordenar por erro e pegar os melhores
            best_error_order = np.argsort(candidate_errors)[:n_solutions_per_point]
            selected_for_this_point = closest_candidates_idx[best_error_order]
            
            # Adicionar ao conjunto de índices selecionados
            selected_indices_set.update(solutions_indices[selected_for_this_point])
        
        # Liberação de memória explícita
        del distances_matrix_batch, risk_diff, return_diff
        
        # Progress report
        progress = (batch_end / n_frontier_points) * 100
        print(f"   📈 Progresso: {progress:.1f}% ({batch_end}/{n_frontier_points} pontos processados)")
    
    # Converter conjunto para lista e criar DataFrame das soluções selecionadas
    selected_indices_list = list(selected_indices_set)
    combined_solutions = df_logs.loc[selected_indices_list].copy()
    
    print(f"✅ Selecionadas {len(combined_solutions)} soluções únicas")
    print(f"📊 Erro médio das soluções selecionadas: {combined_solutions['percent_error'].mean():.6e}")
    
    return combined_solutions

def analyze_portfolio_results_fast(output_dir, efficient_frontier_file=None, n_processes=None):
    """
    Versão otimizada e paralelizada da análise de portfólio para datasets grandes.
    Usa automaticamente processamento paralelo quando benéfico.
    
    Args:
        output_dir: Diretório contendo os resultados
        efficient_frontier_file: Arquivo da fronteira eficiente (opcional)
        n_processes: Número de processos paralelos (None = auto)
    
    Returns:
        dict: Métricas calculadas
    """
    return analyze_portfolio_results(
        output_dir=output_dir,
        efficient_frontier_file=efficient_frontier_file,
        use_parallel=True,
        n_processes=n_processes
    )

def analyze_portfolio_results_sequential(output_dir, efficient_frontier_file=None):
    """
    Versão sequencial da análise de portfólio (compatibilidade com versão original).
    
    Args:
        output_dir: Diretório contendo os resultados
        efficient_frontier_file: Arquivo da fronteira eficiente (opcional)
    
    Returns:
        dict: Métricas calculadas
    """
    return analyze_portfolio_results(
        output_dir=output_dir,
        efficient_frontier_file=efficient_frontier_file,
        use_parallel=False,
        n_processes=None
    )

def calculate_hypervolume_2d(points, reference_point):
    """Calcula o hipervolume para problemas 2D (retorno vs risco)."""
    if len(points) == 0:
        return 0.0
    
    # Converter para array numpy
    points = np.array(points)
    
    # Para maximizar retorno e minimizar risco, vamos transformar o problema
    # Transformamos para um problema de maximização pura
    transformed_points = [[point[0], -point[1]] for point in points]
    transformed_ref = [reference_point[0], -reference_point[1]]
    
    # Remover pontos dominados
    non_dominated = _get_non_dominated_points(transformed_points)
    
    if len(non_dominated) == 0:
        return 0.0
    
    return _calculate_2d_hypervolume(non_dominated, transformed_ref)

def _get_non_dominated_points(points):
    """Auxiliar para encontrar pontos não dominados - versão otimizada."""
    points = np.array(points)
    n = len(points)
    
    if n == 0:
        return []
    
    # Ordenar por primeira coordenada (descendente) para otimização
    sorted_indices = np.argsort(-points[:, 0])
    sorted_points = points[sorted_indices]
    
    non_dominated_mask = np.ones(n, dtype=bool)
    min_y_so_far = float('inf')
    
    # Algoritmo otimizado O(n log n)
    for i, point in enumerate(sorted_points):
        if point[1] < min_y_so_far:
            min_y_so_far = point[1]
        else:
            non_dominated_mask[sorted_indices[i]] = False
    
    return points[non_dominated_mask].tolist()

def _calculate_2d_hypervolume(non_dominated_points, reference_point):
    """Auxiliar para calcular hipervolume 2D."""
    # Ordenar por primeira coordenada
    non_dominated_points.sort()
    
    # Calcular hipervolume
    hypervolume = 0.0
    for i, point in enumerate(non_dominated_points):
        if i == 0:
            width = point[0] - reference_point[0]
        else:
            width = point[0] - non_dominated_points[i-1][0]
        
        height = point[1] - reference_point[1]
        hypervolume += max(0, width * height)
    
    return abs(hypervolume)

def save_metrics_to_csv(output_dir, metrics):
    """Salva as métricas calculadas em um arquivo CSV."""
    metrics_df = pd.DataFrame([metrics])
    csv_path = os.path.join(output_dir, "analysis_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    return csv_path

def save_filtered_populations(output_dir, **populations):
    """Salva as populações filtradas em arquivos CSV."""
    for name, population in populations.items():
        if population is not None and len(population) > 0:
            csv_path = os.path.join(output_dir, f"population_{name}.csv")
            population.to_csv(csv_path, index=False)

def _calculate_cardinalidade_optimized(df_logs):
    """Calcula cardinalidade de forma otimizada."""
    try:
        # Tentar parsing vetorizado primeiro para strings simples como "[1,2,3]"
        df_logs["num_selected"] = df_logs["selected_assets"].str.count(',') + 1
        # Verificar se funcionou (valores NaN indicam strings que não são listas simples)
        if df_logs["num_selected"].isna().any():
            raise ValueError("Fallback needed")
    except (AttributeError, ValueError):
        # Fallback para método seguro
        def _count_selected_assets(x):
            try:
                if isinstance(x, str):
                    return len(eval(x))
                elif hasattr(x, '__len__'):
                    return len(x)
                else:
                    return 0
            except Exception:
                return 0
        
        df_logs["num_selected"] = df_logs["selected_assets"].apply(_count_selected_assets)
    
    return df_logs

def _plot_efficient_vs_all(output_dir, df_logs, efficient_frontier, pareto_frontier=None):
    """Plota fronteira eficiente vs todas as soluções."""
    plt.figure(figsize=(12, 8))
    
    # Plotar TODAS as soluções sem subsampling
    n_points = len(df_logs)
    print(f"📊 Plotando TODAS as {n_points} soluções")
    
    # Usar alpha e tamanho de ponto adaptativos para melhor visualização
    if n_points > 50000:
        alpha = 0.2
        point_size = 4
    elif n_points > 10000:
        alpha = 0.4
        point_size = 6
    else:
        alpha = 0.6
        point_size = 8
    
    plt.scatter(df_logs["risk"], df_logs["expected_return"], 
               s=point_size, color='gray', alpha=alpha, label="Todas as Soluções", rasterized=True)
    
    # Destacar a fronteira de Pareto se fornecida (ela já está incluída em df_logs)
    if pareto_frontier is not None and len(pareto_frontier) > 0:
        plt.scatter(pareto_frontier["risk"], pareto_frontier["expected_return"], 
                   s=point_size*2, color='red', alpha=0.8, label="Fronteira de Pareto", zorder=4, 
                   edgecolors='darkred', linewidths=0.5)
    
    plt.plot(efficient_frontier["std_dev"], efficient_frontier["mean_return"], 
            color='blue', linewidth=2, label=EFFICIENT_FRONTIER_LABEL)

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
    offset_distance = 0.001  # Pequeno deslocamento para evitar sobreposição total
    
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
            angle = (i * 2 * math.pi) / len(special_solutions)  # Distribuir em círculo
            risk_pos += offset_distance * math.cos(angle)
            return_pos += offset_distance * math.sin(angle)
        
        plt.scatter(risk_pos, return_pos, s=120, color=solution['color'], 
                   edgecolor='black', alpha=0.8, label=solution['label'], zorder=5)
        
        plotted_positions.append((risk_pos, return_pos))
    
    plt.title("Fronteira Eficiente vs Todas as Soluções")
    plt.xlabel(RISK_LABEL)
    plt.ylabel(RETURN_LABEL)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fronteira_eficiente_vs_todas.png"), dpi=150, bbox_inches='tight')
    plt.close()

def _plot_efficient_vs_pareto(output_dir, efficient_frontier, pareto_frontier):
    """Plota fronteira eficiente vs fronteira de Pareto."""
    plt.figure(figsize=(12, 8))
    plt.plot(efficient_frontier["std_dev"], efficient_frontier["mean_return"], 
            color='blue', linewidth=2, label=EFFICIENT_FRONTIER_LABEL)
    plt.scatter(pareto_frontier["risk"], pareto_frontier["expected_return"], s=50, color='red', 
               alpha=0.8, label="Fronteira de Pareto", zorder=4, rasterized=True)
    plt.title("Fronteira Eficiente vs Fronteira de Pareto")
    plt.xlabel(RISK_LABEL)
    plt.ylabel(RETURN_LABEL)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fronteira_eficiente_vs_pareto.png"), dpi=150, bbox_inches='tight')
    plt.close()

def _plot_efficient_vs_best(output_dir, efficient_frontier, best_solutions, filename, label):
    """Plota fronteira eficiente vs melhores soluções."""
    plt.figure(figsize=(12, 8))
    plt.plot(efficient_frontier["std_dev"], efficient_frontier["mean_return"], 
            color='blue', linewidth=2, label=EFFICIENT_FRONTIER_LABEL)
    color = 'green' if 'interpolacao' in filename else 'purple'
    plt.scatter(best_solutions["risk"], best_solutions["expected_return"], s=30, color=color, 
               alpha=0.7, label=label, zorder=4, rasterized=True)
    plt.title(f"Fronteira Eficiente vs {label}")
    plt.xlabel(RISK_LABEL)
    plt.ylabel(RETURN_LABEL)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def _plot_cardinalidade_histogram(output_dir, df_logs):
    """Plota histograma da cardinalidade."""
    df_logs = _calculate_cardinalidade_optimized(df_logs)
    
    plt.figure(figsize=(10, 6))
    plt.hist(df_logs["num_selected"], bins=range(1, df_logs["num_selected"].max()+2), 
            align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title("Distribuição da Cardinalidade (Número de Ativos Selecionados)")
    plt.xlabel("Número de Ativos")
    plt.ylabel("Frequência")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, df_logs["num_selected"].max()+1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histograma_cardinalidade.png"), dpi=150, bbox_inches='tight')
    plt.close()

def plot_frontiers_comparison(output_dir, df_logs, efficient_frontier=None, pareto_frontier=None, 
                            best_efficient=None, best_pareto=None):
    """Gera gráficos comparativos das fronteiras - versão otimizada e modular."""
    
    # Configurar matplotlib para melhor performance
    plt.ioff()  # Desabilitar modo interativo
    
    try:
        # Gráfico 1: Fronteira eficiente vs todas as soluções
        if efficient_frontier is not None:
            _plot_efficient_vs_all(output_dir, df_logs, efficient_frontier, pareto_frontier)
        
        # Gráfico 2: Fronteira eficiente vs fronteira de Pareto
        if efficient_frontier is not None and pareto_frontier is not None:
            _plot_efficient_vs_pareto(output_dir, efficient_frontier, pareto_frontier)
        
        # Gráfico 3: Fronteira eficiente vs melhores soluções (interpolação)
        if efficient_frontier is not None and best_efficient is not None:
            _plot_efficient_vs_best(
                output_dir, efficient_frontier, best_efficient, 
                "fronteira_eficiente_vs_melhores_interpolacao.png",
                "100 Melhores (Erro Interpolação)"
            )
        
        # Gráfico 4: Fronteira eficiente vs melhores soluções (Pareto)
        if efficient_frontier is not None and best_pareto is not None:
            _plot_efficient_vs_best(
                output_dir, efficient_frontier, best_pareto,
                "fronteira_eficiente_vs_melhores_pareto.png", 
                "100 Melhores (Pareto)"
            )
        
        # Gráfico 5: Histograma da cardinalidade
        if 'selected_assets' in df_logs.columns:
            _plot_cardinalidade_histogram(output_dir, df_logs)
            
    finally:
        # Reabilitar modo interativo
        plt.ion()

def _load_and_validate_data(output_dir, efficient_frontier_file):
    """Carrega e valida os dados necessários para análise."""
    # Carregar logs
    logs_path = os.path.join(output_dir, "execution_logs.csv")
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Arquivo execution_logs.csv não encontrado em {output_dir}")
    
    df_logs = pd.read_csv(logs_path)
    print(f"✅ Carregados {len(df_logs)} logs de execução")
    
    # Carregar fronteira eficiente (opcional)
    efficient_frontier = None
    if efficient_frontier_file and os.path.exists(efficient_frontier_file):
        efficient_frontier = load_efficient_frontier(efficient_frontier_file)
        if efficient_frontier is not None:
            print(f"✅ Carregada fronteira eficiente com {len(efficient_frontier)} pontos")
        else:
            print("⚠️ Falha ao carregar fronteira eficiente")
    else:
        print("ℹ️ Fronteira eficiente não fornecida ou não encontrada")
    
    return df_logs, efficient_frontier

def _calculate_metrics(df_logs, efficient_frontier, pareto_frontier, use_parallel=True):
    """Calcula todas as métricas de avaliação com opção de paralelização."""
    metrics = {}
    
    # Erros de interpolação
    if efficient_frontier is not None:
        print("📏 Calculando erros de interpolação...")
        df_logs = calculate_interpolation_errors(df_logs, efficient_frontier)
        metrics['avg_interpolation_error'] = df_logs['percent_error'].mean()
        metrics['median_interpolation_error'] = df_logs['percent_error'].median()
        print(f"✅ Erro médio de interpolação: {metrics['avg_interpolation_error']:.12e}")
        print(f"✅ Erro mediano de interpolação: {metrics['median_interpolation_error']:.12e}")
    else:
        metrics['avg_interpolation_error'] = None
        metrics['median_interpolation_error'] = None
        print("⚠️ Erro de interpolação não calculado (fronteira eficiente indisponível)")
    
    # IGD+ com paralelização opcional
    if efficient_frontier is not None:
        print("📐 Calculando IGD+...")
        if use_parallel:
            metrics['igd_plus'] = calculate_igd_plus_parallel(pareto_frontier, efficient_frontier)
        else:
            metrics['igd_plus'] = calculate_igd_plus(pareto_frontier, efficient_frontier)
        print(f"✅ IGD+ calculado: {metrics['igd_plus']:.12e}")
    else:
        metrics['igd_plus'] = None
        print("⚠️ IGD+ não calculado (fronteira eficiente indisponível)")
    
    return df_logs, metrics

def _calculate_hypervolumes(df_logs, best_efficient_solutions, best_pareto_solutions):
    """Calcula os hipervolumes para diferentes conjuntos de soluções."""
    print("📏 Calculando hipervolumes...")
    
    # Ponto de referência: ligeiramente pior que os piores valores para garantir dominação
    # Retorno ligeiramente menor que o mínimo, risco ligeiramente maior que o máximo
    min_return = df_logs['expected_return'].min()
    max_risk = df_logs['risk'].max()
    
    # Adicionar uma pequena margem (1% do range) para garantir dominação
    return_range = df_logs['expected_return'].max() - min_return
    risk_range = max_risk - df_logs['risk'].min()
    
    ref_point = [
        min_return - 0.01 * return_range,  # Retorno 1% pior que o mínimo
        max_risk + 0.01 * risk_range       # Risco 1% pior que o máximo
    ]
    
    print(f"🎯 Ponto de referência: Retorno={ref_point[0]:.10f}, Risco={ref_point[1]:.10f}")
    
    # Hipervolume geral
    all_points = df_logs[['expected_return', 'risk']].values
    hv_general = calculate_hypervolume_2d(all_points, ref_point)
    print(f"✅ Hipervolume geral: {hv_general:.12e}")
    
    # Hipervolume das melhores soluções (Pareto)
    hv_best_pareto = None
    if best_pareto_solutions is not None:
        pareto_points = best_pareto_solutions[['expected_return', 'risk']].values
        hv_best_pareto = calculate_hypervolume_2d(pareto_points, ref_point)
        print(f"✅ Hipervolume melhores Pareto: {hv_best_pareto:.12e}")
    
    # Hipervolume das melhores soluções (fronteira eficiente)
    hv_best_efficient = None
    if best_efficient_solutions is not None:
        efficient_points = best_efficient_solutions[['expected_return', 'risk']].values
        hv_best_efficient = calculate_hypervolume_2d(efficient_points, ref_point)
        print(f"✅ Hipervolume melhores eficientes: {hv_best_efficient:.12e}")
    
    return {
        'hypervolume_general': hv_general,
        'hypervolume_best_pareto': hv_best_pareto,
        'hypervolume_best_efficient': hv_best_efficient,
        'reference_point_return': ref_point[0],
        'reference_point_risk': ref_point[1]
    }

def analyze_portfolio_results(output_dir, efficient_frontier_file=None, use_parallel=True, n_processes=None):
    """
    Função principal para análise dos resultados de otimização de portfólio - versão paralelizada.
    
    Args:
        output_dir: Diretório contendo os resultados
        efficient_frontier_file: Arquivo da fronteira eficiente (opcional)
        use_parallel: Se True, usa processamento paralelo para acelerar cálculos (padrão: True)
        n_processes: Número de processos paralelos (None = auto)
    """
    print(f"\n📊 Iniciando análise de dados para: {output_dir}")
    
    if use_parallel:
        if n_processes is None:
            n_processes = min(cpu_count(), 8)
        print(f"🚀 Modo paralelo ativado com {n_processes} processos")
    else:
        print("🐌 Modo sequencial ativado")
    
    start_time = time.time()
    
    try:
        # 1. Carregar e validar dados
        df_logs, efficient_frontier = _load_and_validate_data(output_dir, efficient_frontier_file)
        
        # 2. Calcular fronteira de Pareto com paralelização opcional
        print("🔍 Calculando fronteira de Pareto...")
        if use_parallel and len(df_logs) > 10000:
            pareto_frontier = calculate_pareto_frontier_parallel(df_logs, n_processes)
        else:
            pareto_frontier = calculate_pareto_frontier(df_logs)
        print(f"✅ Fronteira de Pareto calculada com {len(pareto_frontier)} pontos")
        
        # 3. Calcular métricas com paralelização opcional
        df_logs, metrics = _calculate_metrics(df_logs, efficient_frontier, pareto_frontier, use_parallel)
        
        # 4. Selecionar melhores soluções usando versão paralelizada
        best_efficient_solutions = None
        best_pareto_solutions = None
        
        if efficient_frontier is not None and 'percent_error' in df_logs.columns:
            print("🏆 Selecionando melhores soluções para cada ponto da fronteira eficiente...")
            if use_parallel:
                best_efficient_solutions = get_best_solutions_per_frontier_point_parallel(
                    df_logs, efficient_frontier, n_solutions_per_point=100, n_processes=n_processes
                )
            else:
                best_efficient_solutions = get_best_solutions_per_frontier_point(
                    df_logs, efficient_frontier, n_solutions_per_point=100
                )
            print(f"✅ Selecionadas {len(best_efficient_solutions)} melhores soluções para fronteira eficiente")
        
        print("🏆 Selecionando melhores soluções para cada ponto da fronteira de Pareto...")
        if 'percent_error' in df_logs.columns:
            if use_parallel:
                best_pareto_solutions = get_best_solutions_per_frontier_point_parallel(
                    df_logs, pareto_frontier, n_solutions_per_point=100, n_processes=n_processes
                )
            else:
                best_pareto_solutions = get_best_solutions_per_frontier_point(
                    df_logs, pareto_frontier, n_solutions_per_point=100
                )
        else:
            # Fallback se não há erro de interpolação
            n_best = min(100, len(pareto_frontier))
            best_pareto_solutions = get_best_solutions_for_frontier(pareto_frontier, n_best)
        print(f"✅ Selecionadas {len(best_pareto_solutions)} melhores soluções para fronteira de Pareto")
        
        # 5. Calcular hipervolumes
        hv_metrics = _calculate_hypervolumes(df_logs, best_efficient_solutions, best_pareto_solutions)
        metrics.update(hv_metrics)
        
        # 6. Adicionar métricas básicas
        metrics.update({
            'total_evaluations': len(df_logs),
            'pareto_frontier_size': len(pareto_frontier),
            'best_sharpe': df_logs['sharpe'].max() if 'sharpe' in df_logs.columns else None,
            'processing_mode': 'parallel' if use_parallel else 'sequential',
            'n_processes_used': n_processes if use_parallel else 1
        })
        
        # 7. Salvar dados
        _save_analysis_results(output_dir, metrics, pareto_frontier, best_efficient_solutions, best_pareto_solutions)
        
        # 8. Gerar gráficos (apenas se necessário)
        print("📈 Gerando gráficos...")
        plot_frontiers_comparison(
            output_dir, df_logs, efficient_frontier, pareto_frontier, 
            best_efficient_solutions, best_pareto_solutions
        )
        
        # 9. Exibir resumo
        total_time = time.time() - start_time
        print(f"\n⏱️ Tempo total de processamento: {total_time:.2f}s")
        _display_metrics_summary(metrics)
        
        print("="*60)
        print(f"✅ Análise completa! Resultados salvos em: {output_dir}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")
        raise

def _save_analysis_results(output_dir, metrics, pareto_frontier, best_efficient_solutions, best_pareto_solutions):
    """Salva os resultados da análise."""
    print("💾 Salvando métricas e populações filtradas...")
    
    # Salvar métricas
    metrics_path = save_metrics_to_csv(output_dir, metrics)
    print(f"✅ Métricas salvas em: {metrics_path}")
    
    # Salvar populações
    save_filtered_populations(
        output_dir,
        pareto_frontier=pareto_frontier,
        best_efficient=best_efficient_solutions,
        best_pareto=best_pareto_solutions
    )

def _display_metrics_summary(metrics):
    """Exibe resumo das métricas calculadas."""
    print("\n" + "="*60)
    print("📊 RESUMO DAS MÉTRICAS CALCULADAS")
    print("="*60)
    print(f"📈 Total de avaliações: {metrics['total_evaluations']}")
    print(f"🎯 Tamanho da fronteira de Pareto: {metrics['pareto_frontier_size']}")
    
    if 'processing_mode' in metrics:
        mode_icon = "🚀" if metrics['processing_mode'] == 'parallel' else "🐌"
        print(f"{mode_icon} Modo de processamento: {metrics['processing_mode']}")
        if metrics['processing_mode'] == 'parallel':
            print(f"⚙️ Processos utilizados: {metrics.get('n_processes_used', 'N/A')}")
    
    if metrics['igd_plus'] is not None:
        print(f"📐 IGD+: {metrics['igd_plus']:.12e}")
    
    if metrics['avg_interpolation_error'] is not None:
        print(f"📏 Erro médio de interpolação: {metrics['avg_interpolation_error']:.4e}")
        print(f"📊 Erro mediano de interpolação: {metrics['median_interpolation_error']:.4e}")
    
    print(f"📦 Hipervolume geral: {metrics['hypervolume_general']:.12e}")
    
    if metrics['hypervolume_best_pareto'] is not None:
        print(f"🏆 Hipervolume melhores Pareto: {metrics['hypervolume_best_pareto']:.12e}")
    
    if metrics['hypervolume_best_efficient'] is not None:
        print(f"💎 Hipervolume melhores eficientes: {metrics['hypervolume_best_efficient']:.12e}")
    
    if metrics['best_sharpe'] is not None:
        print(f"⭐ Melhor Sharpe ratio: {metrics['best_sharpe']:.12f}")
