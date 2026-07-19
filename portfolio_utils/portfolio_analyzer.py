#!/usr/bin/env python3
"""
Analisador de dados para resultados de otimização de portfólio.
Calcula métricas de performance, fronteiras de Pareto e gera visualizações.

⭐ NOVO: Suporte para lazy loading de arquivos grandes
"""

import os
import gc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo, evita problemas com tkinter em multiprocessing
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

# ⭐ NOVO: Lazy Loading Support
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


def _downcast_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    ⭐ OTIMIZAÇÃO: Reduz memória fazendo downcast de float64 → float32.
    
    Reduz uso de RAM pela metade com perda mínima de precisão (de ~15 dígitos para ~7).
    
    Args:
        df: DataFrame para downcast
        
    Returns:
        DataFrame com colunas float downcastadas para float32
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        original_size = df.memory_usage(deep=True).sum() / (1024**2)
        df[float_cols] = df[float_cols].astype('float32')
        new_size = df.memory_usage(deep=True).sum() / (1024**2)
        print(f"[MEMORY] Downcast float64→float32: {original_size:.1f}MB → {new_size:.1f}MB ({100*new_size/original_size:.1f}%)")
    return df


def _load_dataframe_flexible(folder, filename_base):
    """
    Carrega um arquivo CSV ou Parquet com otimização de memória.
    Tenta carregar filename_base.parquet primeiro, depois filename_base.csv
    
    ⭐ Aplica downcast automático de float64 → float32 para reduzir RAM pela metade.
    """
    parquet_path = os.path.join(folder, f"{filename_base}.parquet")
    csv_path = os.path.join(folder, f"{filename_base}.csv")
    
    df = None
    if os.path.exists(parquet_path):
        print(f"[INFO] Carregando {filename_base}.parquet")
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"[INFO] Carregando {filename_base}.csv")
        df = pd.read_csv(csv_path)
    
    # ⭐ Downcast automático de float64 → float32
    if df is not None:
        df = _downcast_float_columns(df)
    
    return df


def _load_dataframe_lazy(file_path: str, chunk_size: int = 50000):
    """
    ⭐ NOVO: Carregador lazy para arquivos Parquet/CSV grandes.
    
    Yields chunks de dados em vez de carregar tudo na memória.
    Reduz picos de RAM drasticamente.
    
    Args:
        file_path: Caminho do arquivo (parquet ou csv)
        chunk_size: Número de linhas por chunk
    
    Yields:
        DataFrame de até chunk_size linhas
    """
    if file_path.endswith('.parquet'):
        # Lazy loading de Parquet usando PyArrow
        if PYARROW_AVAILABLE:
            parquet_file = pq.ParquetFile(file_path)
            for i in range(parquet_file.num_row_groups):
                chunk = parquet_file.read_row_group(i).to_pandas()
                # Se chunk é maior que chunk_size, dividir
                for j in range(0, len(chunk), chunk_size):
                    yield chunk.iloc[j:j+chunk_size]
        else:
            # Fallback: carrega tudo (sem lazy)
            yield pd.read_parquet(file_path)
    else:
        # CSV: usar pandas read_csv com chunksize
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            yield chunk


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


def calculate_pareto_frontier_lazy(file_path: str, chunk_size: int = 50000):
    """
    ⭐ NOVO: Calcula fronteira de Pareto usando lazy loading.
    
    Processa arquivo grande em chunks, eliminando dominados
    incrementalmente para reduzir RAM.
    
    Args:
        file_path: Caminho do arquivo Parquet/CSV
        chunk_size: Tamanho do chunk
    
    Returns:
        DataFrame com pontos Pareto (completo)
    """
    print(f"[LAZY-PARETO] Calculando Pareto de forma incremental (chunk_size={chunk_size})...")
    
    pareto_frontier = None
    chunk_count = 0
    
    for chunk in _load_dataframe_lazy(file_path, chunk_size):
        chunk_count += 1
        
        if pareto_frontier is None:
            # Primeiro chunk
            pareto_frontier = calculate_pareto_frontier(chunk)
            print(f"[LAZY-PARETO] Chunk {chunk_count}: {len(pareto_frontier)} candidatos Pareto")
        else:
            # Combinar com Pareto anterior e recalcular
            combined = pd.concat([pareto_frontier, chunk], ignore_index=True)
            pareto_frontier = calculate_pareto_frontier(combined)
            print(f"[LAZY-PARETO] Chunk {chunk_count}: {len(pareto_frontier)} candidatos Pareto (total combinado)")
            
            # Limpeza de memória
            del combined
    
    print(f"[LAZY-PARETO] Processamento completo: {len(pareto_frontier)} pontos Pareto finais")
    return pareto_frontier


def calculate_pareto_frontier_parallel(df_logs, n_processes=None):
    """Calcula a fronteira de Pareto das soluções - versão paralelizada para datasets grandes."""
    n_solutions = len(df_logs)
    
    # Para datasets pequenos, usar método sequencial otimizado
    if n_solutions < 10000:
        return calculate_pareto_frontier(df_logs)
    
    print(f"[INFO] Calculando fronteira de Pareto com paralelização para {n_solutions} soluções")
    
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
    
    print(f"[INFO] Processando {len(chunks)} chunks em {n_processes} processos")
    
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
        
        print(f"[INFO] Fronteira de Pareto calculada: {len(final_pareto)} pontos")
        return final_pareto
        
    except Exception as e:
        print(f"[WARNING] Erro no cálculo paralelo de Pareto: {e}")
        print("[INFO] Fallback para método sequencial...")
        return calculate_pareto_frontier(df_logs)

def _find_pareto_in_chunk(chunk_data):
    """
    Encontra pontos Pareto dentro de um chunk - versão otimizada.
    
    Usa a mesma lógica de calculate_pareto_frontier para garantir
    tratamento correto de empates.
    """
    solutions, indices, _ = chunk_data  # offset não é usado
    n = len(solutions)
    
    if n == 0:
        return []
    
    if n == 1:
        return indices.tolist()
    
    # Ordenar por retorno DESC
    order = np.argsort(-solutions[:, 0])
    sorted_risks = solutions[order, 1]
    
    # Skyline vetorizado
    min_risk_cumulative = np.minimum.accumulate(sorted_risks)
    
    pareto_mask = np.empty(n, dtype=bool)
    pareto_mask[0] = True
    pareto_mask[1:] = sorted_risks[1:] < min_risk_cumulative[:-1]
    
    # Candidatos locais
    local_candidates = order[pareto_mask]
    
    # Filtrar empates de retorno (manter o de menor risco por grupo de retorno igual)
    if len(local_candidates) > 1:
        cand_returns = solutions[local_candidates, 0]
        cand_risks = solutions[local_candidates, 1]
        
        final_local = []
        i = 0
        while i < len(local_candidates):
            j = i + 1
            while j < len(local_candidates) and cand_returns[j] == cand_returns[i]:
                j += 1
            # Do grupo [i:j], pegar o de menor risco
            best_in_group = i + np.argmin(cand_risks[i:j])
            final_local.append(local_candidates[best_in_group])
            i = j
        local_candidates = np.array(final_local)
    
    return indices[local_candidates].tolist()

def calculate_pareto_frontier(df_logs):
    """
    Calcula a fronteira de Pareto das soluções - versão otimizada SEM cópias desnecessárias.
    
    ⭐ CRÍTICO: Usa índices (arrays numpy) até o final, evita .copy() do DataFrame inteiro.
    Retorna apenas filtro final sem duplicações de memória.
    
    Critério de dominância (problema biobjetivo):
        - Minimizar risco
        - Maximizar retorno
    
    Um ponto A domina B se: A.retorno >= B.retorno AND A.risco <= B.risco,
    com pelo menos uma desigualdade estrita.
    
    O algoritmo usa ordenação simples + filtragem vetorizada para O(n log n)
    e trata empates corretamente com uma passada extra O(k) onde k = |Pareto|.
    """
    solutions = df_logs[['expected_return', 'risk']].values  # Array view, não cópia
    n = len(solutions)
    
    if n == 0:
        return df_logs.iloc[0:0]  # Sem .copy() - retorna view vazia
    
    if n == 1:
        return df_logs  # Sem .copy() - retorna referência ao original
    
    # Passo 1: Ordenar por retorno DESC (argsort simples, O(n log n))
    order = np.argsort(-solutions[:, 0])
    sorted_risks = solutions[order, 1]
    
    # Passo 2: Skyline vetorizado - seleciona candidatos onde risco < min acumulado
    min_risk_cumulative = np.minimum.accumulate(sorted_risks)
    
    pareto_mask = np.empty(n, dtype=bool)
    pareto_mask[0] = True
    pareto_mask[1:] = sorted_risks[1:] < min_risk_cumulative[:-1]
    
    # Passo 3: Filtrar empates de retorno (manter apenas o de menor risco por retorno)
    # Isso é necessário porque argsort simples não garante ordem de risco em empates
    candidate_indices = order[pareto_mask]
    if len(candidate_indices) > 1:
        candidate_returns = solutions[candidate_indices, 0]
        candidate_risks = solutions[candidate_indices, 1]
        
        # Detectar onde o retorno muda (ou é o primeiro elemento)
        keep_mask = np.empty(len(candidate_indices), dtype=bool)
        keep_mask[0] = True
        keep_mask[1:] = candidate_returns[1:] != candidate_returns[:-1]
        
        # Para retornos repetidos, manter apenas o primeiro (que pode não ser o de menor risco)
        # Precisamos verificar e corrigir: para cada grupo de retorno igual, manter o de menor risco
        if not keep_mask.all():
            # Há empates - precisamos resolver
            final_indices = []
            i = 0
            while i < len(candidate_indices):
                # Encontrar fim do grupo com mesmo retorno
                j = i + 1
                while j < len(candidate_indices) and candidate_returns[j] == candidate_returns[i]:
                    j += 1
                # Do grupo [i:j], pegar o de menor risco
                group_risks = candidate_risks[i:j]
                best_in_group = i + np.argmin(group_risks)
                final_indices.append(candidate_indices[best_in_group])
                i = j
            candidate_indices = np.array(final_indices)
    
    # ⭐ OTIMIZAÇÃO: Apenas filtra o DataFrame no retorno final, sem .copy() intermediários
    pareto_df = df_logs.iloc[candidate_indices].sort_values('risk')
    return pareto_df

def calculate_igd_plus_parallel(pareto_front, reference_front, n_processes=None):
    """
    Calcula o IGD+ (Inverted Generational Distance Plus) de forma paralela.
    
    IGD+ = (1/|R|) * Σ min(d(ref_i, pareto_j)) para i ∈ R
    
    Onde:
    - R = conjunto de pontos de referência (Fronteira Verdadeira)
    - d() = distância Euclidiana
    - Quanto MENOR, melhor (0 = perfeito)
    
    Prova automaticamente:
    1. Convergência: se Pareto longe da referência → IGD+ alto
    2. Spread: se faltam Pareto em algumas regiões → IGD+ alto
    """
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
    
    print(f"[IGD+] Calculando IGD+ paralelo: {n_ref} pontos de referência vs {n_pareto} pontos Pareto")
    
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
        print(f"[IGD+] [Error] Erro no cálculo paralelo de IGD+: {e}")
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
    """
    Calcula o IGD+ (Inverted Generational Distance Plus) de forma sequencial.
    
    IGD+ = (1/|R|) * Σ min(d(ref_i, pareto_j)) para i ∈ R
    
    Onde:
    - R = conjunto de pontos de referência (Fronteira Verdadeira)
    - d() = distância Euclidiana
    - Quanto MENOR, melhor (0 = perfeito)
    
    Prova automaticamente:
    1. Convergência: se Pareto longe da referência → IGD+ alto
    2. Spread: se faltam Pareto em algumas regiões → IGD+ alto
    """
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
    
    # ⭐ CRÍTICO: Modificar DataFrame *in-place* para economizar 50% de RAM
    # Sem .copy(), o DataFrame é modificado diretamente
    
    try:
        # Verificar se há dados suficientes para interpolação
        if len(efficient_frontier) < 2:
            print("[INTERPOLACAO] [ERROR] Fronteira eficiente tem poucos pontos para interpolação")
            return df_logs
        
        # Interpolação otimizada com tratamento de erros
        # Nota: interp1d pode falhar/ficar instável com valores duplicados no eixo x.
        # Por isso, removemos duplicatas e garantimos ordenação para cada interpolação.
        ef_by_risk = (
            efficient_frontier[["std_dev", "mean_return"]]
            .dropna()
            .drop_duplicates(subset=["std_dev"], keep="first")
            .sort_values("std_dev")
        )
        ef_by_return = (
            efficient_frontier[["mean_return", "std_dev"]]
            .dropna()
            .drop_duplicates(subset=["mean_return"], keep="first")
            .sort_values("mean_return")
        )

        if len(ef_by_risk) < 2 or len(ef_by_return) < 2:
            print("[INTERPOLACAO] [ERROR] Fronteira eficiente insuficiente após limpeza (duplicatas/NaN)")
            return df_logs

        interp_return = interp1d(
            ef_by_risk["std_dev"].values,
            ef_by_risk["mean_return"].values,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
        )
        interp_risk = interp1d(
            ef_by_return["mean_return"].values,
            ef_by_return["std_dev"].values,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan,
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
            
            # Chang et al. (2000), Seção 5.2.2: erro percentual = 100 * |diferença| / |valor_interpolado|
            df_logs["error_return"] = 100.0 * np.abs(df_logs["expected_return"] - df_logs["interp_ret_from_risk"]) / denominator_ret
            df_logs["error_risk"] = 100.0 * np.abs(df_logs["risk"] - df_logs["interp_risk_from_ret"]) / denominator_risk
        
        # Métricas alternativas de erro (ajuda a comparar com definições da literatura)
        # - min: erro “otimista” (mantém compatibilidade com a versão anterior)
        # - max: erro “conservador”
        # - mean: média simples
        # Combinação das duas direções conforme Chang et al. (2000): min(erro_horizontal, erro_vertical)
        # Importante: quando uma direção não tem bracketing (fora do range), o erro daquela direção fica NaN.
        # np.fmin/np.fmax ignoram NaN quando possível.
        err_ret = df_logs["error_return"].to_numpy()
        err_risk = df_logs["error_risk"].to_numpy()
        df_logs["percent_error"] = np.fmin(err_ret, err_risk)
        df_logs["percent_error_max"] = np.fmax(err_ret, err_risk)
        df_logs["percent_error_mean"] = np.nanmean(np.vstack([err_ret, err_risk]), axis=0)
        
        # Tratamento de valores inválidos de forma vetorizada
        df_logs["percent_error"] = np.where(
            np.isfinite(df_logs["percent_error"]),
            df_logs["percent_error"],
            np.nanmax(df_logs["percent_error"][np.isfinite(df_logs["percent_error"])]) * 2
        )
        df_logs["percent_error_max"] = np.where(
            np.isfinite(df_logs["percent_error_max"]),
            df_logs["percent_error_max"],
            np.nanmax(df_logs["percent_error_max"][np.isfinite(df_logs["percent_error_max"])]) * 2
        )
        df_logs["percent_error_mean"] = np.where(
            np.isfinite(df_logs["percent_error_mean"]),
            df_logs["percent_error_mean"],
            np.nanmax(df_logs["percent_error_mean"][np.isfinite(df_logs["percent_error_mean"])]) * 2
        )
        
        # Se ainda há NaN, usar valor padrão
        if np.isnan(df_logs["percent_error"]).any():
            df_logs["percent_error"] = df_logs["percent_error"].fillna(1.0)
        if np.isnan(df_logs["percent_error_max"]).any():
            df_logs["percent_error_max"] = df_logs["percent_error_max"].fillna(1.0)
        if np.isnan(df_logs["percent_error_mean"]).any():
            df_logs["percent_error_mean"] = df_logs["percent_error_mean"].fillna(1.0)
            
    except Exception as e:
        print(f"[INTERPOLACAO] [ERROR] Erro na interpolação: {e}")
        df_logs["percent_error"] = 1.0  # Valor padrão em caso de erro
        df_logs["percent_error_max"] = 1.0
        df_logs["percent_error_mean"] = 1.0
    
    return df_logs

def analyze_portfolio_results_fast(output_dir, efficient_frontier_file=None, n_processes=None, risk_free_rate=0.00057):
    """
    Versão otimizada e paralelizada da análise de portfólio para datasets grandes.
    Usa automaticamente processamento paralelo quando benéfico.
    
    Args:
        output_dir: Diretório contendo os resultados
        efficient_frontier_file: Arquivo da fronteira eficiente (opcional)
        n_processes: Número de processos paralelos (None = auto)
        risk_free_rate: Taxa livre de risco para cálculo do índice de Sharpe (padrão: 0.00057 = 3% anual convertido para semanal)
    
    Returns:
        dict: Métricas calculadas
    """
    return analyze_portfolio_results(
        output_dir=output_dir,
        efficient_frontier_file=efficient_frontier_file,
        use_parallel=True,
        n_processes=n_processes,
        risk_free_rate=risk_free_rate
    )

def analyze_portfolio_results_sequential(output_dir, efficient_frontier_file=None, risk_free_rate=0.00057):
    """
    Versão sequencial da análise de portfólio (compatibilidade com versão original).
    
    Args:
        output_dir: Diretório contendo os resultados
        efficient_frontier_file: Arquivo da fronteira eficiente (opcional)
        risk_free_rate: Taxa livre de risco para cálculo do índice de Sharpe (padrão: 0.00057 = 3% anual convertido para semanal)
    
    Returns:
        dict: Métricas calculadas
    """
    return analyze_portfolio_results(
        output_dir=output_dir,
        efficient_frontier_file=efficient_frontier_file,
        use_parallel=False,
        n_processes=None,
        risk_free_rate=risk_free_rate
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
    """Salva as populações filtradas em arquivos CSV E PARQUET.
    
    Parquet é preferido pois preserva tipos nativos (lists, arrays) sem conversão de string.
    CSV é mantido para compatibilidade.
    """
    for name, population in populations.items():
        if population is not None and len(population) > 0:
            # Salvar em CSV
            csv_path = os.path.join(output_dir, f"population_{name}.csv")
            population.to_csv(csv_path, index=False)
            
            # Salvar em Parquet (preserva tipos nativos)
            parquet_path = os.path.join(output_dir, f"population_{name}.parquet")
            population.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)

def _calculate_cardinalidade_optimized(df_logs):
    """Calcula cardinalidade de forma robusta.
    
    Trata vários formatos:
    - list/np.ndarray (de Parquet): usa len() direto
    - str com vírgulas "[1,2,3]" (CSV parseado): usa ast.literal_eval
    - str com espaços "[1 2 3]" (numpy array convertido): usa split()
    - NaN / strings vazias: retorna 0
    """
    def _parse_selected_assets_robust(x):
        """Parse robusto de ativos selecionados."""
        try:
            # Caso 1: list ou np.ndarray (vindo de Parquet)
            if isinstance(x, (list, np.ndarray)):
                return int(len(x))
            
            # Caso 2: NaN ou None
            if pd.isna(x) or x is None:
                return 0
            
            # Caso 3: string vazia ou "[]"
            if isinstance(x, str):
                x = x.strip()
                if not x or x == '[]':
                    return 0
                
                # Tenta remover colchetes se tiver
                if x.startswith('[') and x.endswith(']'):
                    x = x[1:-1].strip()
                
                # Caso 3a: Vírgulas separando ("1,2,3")
                if ',' in x:
                    elements = [e.strip() for e in x.split(',') if e.strip()]
                    return len(elements)
                
                # Caso 3b: Espaços separando ("1 2 3")
                elif ' ' in x:
                    elements = [e.strip() for e in x.split() if e.strip()]
                    return len(elements)
                
                # Caso 3c: Único elemento
                elif x and x.isdigit():
                    return 1
                
                # Caso 3d: Tentar ast.literal_eval como última chance
                else:
                    result = ast.literal_eval(f'[{x}]') if not x.startswith('[') else ast.literal_eval(x)
                    return len(result)
            
            # Caso 4: outro tipo desconhecido
            return 0
            
        except Exception as e:
            # Falha silenciosa: retorna 0
            return 0
    
    # Aplicar parser robusto
    df_logs["num_selected"] = df_logs["selected_assets"].apply(_parse_selected_assets_robust)
    
    return df_logs

def _plot_efficient_vs_all(output_dir, df_logs, efficient_frontier, pareto_frontier=None):
    """
    Plota fronteira eficiente vs todas as soluções sem alocação de memória por ponto.
    
    ⭐ Usa plt.hexbin em vez de plt.scatter para visualizar todos os dados sem 
    alocação de memória individual para cada ponto.
    """
    plt.figure(figsize=(12, 8))
    
    print(f"[PLOT] Plotando {len(df_logs)} soluções via hexbin (sem sampling)")
    
    # ⭐ HEXBIN em vez de SCATTER: Todos os dados, sem memória por ponto
    # gridsize=100 define resolução da grade hexagonal
    # cmap='Greys' visualiza densidade de pontos
    # bins='log' escala logarítmica para melhor contraste
    plt.hexbin(df_logs["risk"], df_logs["expected_return"],
              gridsize=100, cmap='Greys', bins='log', mincnt=1, 
              alpha=0.8, edgecolors='none', label="Densidade de Soluções")
    
    # Destacar a fronteira de Pareto se fornecida (ela já está incluída em df_logs)
    if pareto_frontier is not None and len(pareto_frontier) > 0:
        plt.scatter(pareto_frontier["risk"], pareto_frontier["expected_return"], 
                   s=12, color='red', alpha=0.8, label="Fronteira de Pareto", zorder=4, 
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
    
    # ⭐ GARBAGE COLLECTION: Liberar memória explicitamente
    plt.close('all')
    gc.collect()

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

def plot_frontiers_comparison(output_dir, df_logs, efficient_frontier=None, pareto_frontier=None):
    """Gera gráficos comparativos das fronteiras - versão simplificada."""
    
    # Configurar matplotlib para melhor performance
    plt.ioff()  # Desabilitar modo interativo
    
    try:
        # Gráfico 1: Fronteira eficiente vs todas as soluções
        if efficient_frontier is not None:
            _plot_efficient_vs_all(output_dir, df_logs, efficient_frontier, pareto_frontier)
        
        # Gráfico 2: Fronteira eficiente vs fronteira de Pareto
        if efficient_frontier is not None and pareto_frontier is not None:
            _plot_efficient_vs_pareto(output_dir, efficient_frontier, pareto_frontier)
        
        # Gráfico 3: Histograma da cardinalidade
        if 'selected_assets' in df_logs.columns:
            _plot_cardinalidade_histogram(output_dir, df_logs)
            
    finally:
        # Reabilitar modo interativo
        plt.ion()

def _load_and_validate_data(output_dir, efficient_frontier_file):
    """Carrega e valida os dados necessários para análise."""
    # Carregar logs (tenta parquet primeiro, depois csv)
    df_logs = _load_dataframe_flexible(output_dir, "execution_logs")
    if df_logs is None:
        raise FileNotFoundError(f"[ERROR] Arquivo execution_logs.parquet ou execution_logs.csv não encontrado em {output_dir}")
    
    print(f"[INFO] Carregados {len(df_logs)} logs de execução")
    
    # Carregar fronteira eficiente (opcional)
    efficient_frontier = None
    if efficient_frontier_file and os.path.exists(efficient_frontier_file):
        efficient_frontier = load_efficient_frontier(efficient_frontier_file)
        if efficient_frontier is not None:
            print(f"[INFO] Carregada fronteira eficiente com {len(efficient_frontier)} pontos")
        else:
            print("[WARNING] Falha ao carregar fronteira eficiente")
    else:
        print("[INFO] Fronteira eficiente não fornecida ou não encontrada")
    
    return df_logs, efficient_frontier

def _calculate_metrics(df_logs, efficient_frontier, pareto_frontier, use_parallel=True):
    """Calcula todas as métricas de avaliação com opção de paralelização."""
    metrics = {}
    
    # Erros de interpolação (OTIMIZAÇÃO: uma única passada, depois filtra para Pareto)
    if efficient_frontier is not None:
        print("[INTERPOLACAO] [INIT] Calculando erros de interpolação...")
        # UMA ÚNICA CHAMADA: calcula erros para toda a população
        df_logs = calculate_interpolation_errors(df_logs, efficient_frontier)
        
        # Métricas para TODA A POPULAÇÃO
        metrics['avg_interpolation_error'] = df_logs['percent_error'].mean()
        metrics['median_interpolation_error'] = df_logs['percent_error'].median()
        metrics['min_interpolation_error'] = df_logs['percent_error'].min()
        metrics['max_interpolation_error'] = df_logs['percent_error'].max()
        if 'percent_error_max' in df_logs.columns:
            metrics['avg_interpolation_error_max'] = df_logs['percent_error_max'].mean()
            metrics['median_interpolation_error_max'] = df_logs['percent_error_max'].median()
        if 'percent_error_mean' in df_logs.columns:
            metrics['avg_interpolation_error_mean'] = df_logs['percent_error_mean'].mean()
            metrics['median_interpolation_error_mean'] = df_logs['percent_error_mean'].median()
        print(f" - [INTERPOLACAO] Erro médio de interpolação: {metrics['avg_interpolation_error']:.12e}")
        print(f" - [INTERPOLACAO] Erro mediano de interpolação: {metrics['median_interpolation_error']:.12e}")
        print(f" - [INTERPOLACAO] Erro mín de interpolação: {metrics['min_interpolation_error']:.12e}")
        print(f" - [INTERPOLACAO] Erro máx de interpolação: {metrics['max_interpolation_error']:.12e}")

        # Métricas de erro restritas ao conjunto não-dominado (Pareto)
        # OTIMIZAÇÃO: Filtra o df_logs já calculado ao invés de chamar calculate_interpolation_errors novamente
        if pareto_frontier is not None and len(pareto_frontier) > 0:
            # Usar índices do pareto_frontier para filtrar os erros já calculados
            pareto_with_errors = df_logs.loc[pareto_frontier.index]
            
            metrics['avg_interpolation_error_pareto'] = pareto_with_errors['percent_error'].mean(skipna=True)
            metrics['median_interpolation_error_pareto'] = pareto_with_errors['percent_error'].median(skipna=True)
            metrics['min_interpolation_error_pareto'] = pareto_with_errors['percent_error'].min()
            metrics['max_interpolation_error_pareto'] = pareto_with_errors['percent_error'].max()

            # Métricas com nomes explícitos para comparação com Chang et al. (2000)
            # Chang define o erro percentual por portfólio como min(erro-horizontal, erro-vertical)
            # e, para o caso com cardinalidade, recomenda comparar o conjunto H (histórico) filtrado para não-dominados.
            metrics['chang_mean_percentage_error_H_undominated'] = metrics['avg_interpolation_error_pareto']
            metrics['chang_median_percentage_error_H_undominated'] = metrics['median_interpolation_error_pareto']
            metrics['chang_num_undominated_points_H'] = int(len(pareto_with_errors))
            metrics['chang_num_error_points_H'] = int(pareto_with_errors['percent_error'].notna().sum())
        else:
            metrics['avg_interpolation_error_pareto'] = None
            metrics['median_interpolation_error_pareto'] = None
            metrics['min_interpolation_error_pareto'] = None
            metrics['max_interpolation_error_pareto'] = None
            metrics['chang_mean_percentage_error_H_undominated'] = None
            metrics['chang_median_percentage_error_H_undominated'] = None
            metrics['chang_num_undominated_points_H'] = int(len(pareto_frontier)) if pareto_frontier is not None else None
            metrics['chang_num_error_points_H'] = 0
    else:
        metrics['avg_interpolation_error'] = None
        metrics['median_interpolation_error'] = None
        metrics['min_interpolation_error'] = None
        metrics['max_interpolation_error'] = None
        metrics['avg_interpolation_error_max'] = None
        metrics['median_interpolation_error_max'] = None
        metrics['avg_interpolation_error_mean'] = None
        metrics['median_interpolation_error_mean'] = None
        metrics['avg_interpolation_error_pareto'] = None
        metrics['median_interpolation_error_pareto'] = None
        metrics['min_interpolation_error_pareto'] = None
        metrics['max_interpolation_error_pareto'] = None
        print("[INTERPOLACAO] [WARNING] Erro de interpolação não calculado (fronteira eficiente indisponível)")
    
    # IGD+ com paralelização opcional
    if efficient_frontier is not None:
        print(" [IGD+] [INIT] Calculando IGD+...")
        if use_parallel:
            metrics['igd_plus'] = calculate_igd_plus_parallel(pareto_frontier, efficient_frontier)
        else:
            metrics['igd_plus'] = calculate_igd_plus(pareto_frontier, efficient_frontier)
        print(f" - [IGD+] IGD+ calculado: {metrics['igd_plus']:.12e}")
    else:
        metrics['igd_plus'] = None
        print("[IGD+] [WARNING] IGD+ não calculado (fronteira eficiente indisponível)")
    
    return df_logs, metrics

def _calculate_hypervolumes(df_logs, pareto_frontier):
    """Calcula os hipervolumes para diferentes conjuntos de soluções."""
    print("[HIPERVOLUME] [INIT] Calculando hipervolumes...")
    
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
    
    print(f"[HIPERVOLUME] [INFO] Ponto de referência: Retorno={ref_point[0]:.10f}, Risco={ref_point[1]:.10f}")
    
    # Hipervolume geral
    all_points = df_logs[['expected_return', 'risk']].values
    hv_general = calculate_hypervolume_2d(all_points, ref_point)
    print(f"[HIPERVOLUME] [INFO] Hipervolume geral: {hv_general:.12e}")
    
    # Hipervolume da fronteira de Pareto
    hv_pareto = None
    if pareto_frontier is not None and len(pareto_frontier) > 0:
        pareto_points = pareto_frontier[['expected_return', 'risk']].values
        hv_pareto = calculate_hypervolume_2d(pareto_points, ref_point)
        print(f" - [HIPERVOLUME] Hipervolume fronteira de Pareto: {hv_pareto:.12e}")
    
    return {
        'hypervolume_general': hv_general,
        'hypervolume_pareto': hv_pareto,
        'reference_point_return': ref_point[0],
        'reference_point_risk': ref_point[1]
    }

def analyze_portfolio_results(output_dir, efficient_frontier_file=None, use_parallel=True, n_processes=None, risk_free_rate=0.00057):
    """
    Função principal para análise dos resultados de otimização de portfólio - versão paralelizada.
    
    Args:
        output_dir: Diretório contendo os resultados
        efficient_frontier_file: Arquivo da fronteira eficiente (opcional)
        use_parallel: Se True, usa processamento paralelo para acelerar cálculos (padrão: True)
        n_processes: Número de processos paralelos (None = auto)
        risk_free_rate: Taxa livre de risco para cálculo do índice de Sharpe (padrão: 0.00057 = 3% anual convertido para semanal)
    """
    print(f"\n Iniciando análise de dados para: {output_dir}")
    
    if use_parallel:
        if n_processes is None:
            n_processes = min(cpu_count(), 8)
        print(f"[INFO] Modo paralelo ativado com {n_processes} processos")
    else:
        print("[INFO] Modo sequencial ativado")
    
    start_time = time.time()
    print(f"[INFO] Início da análise: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    try:
        # 1. Carregar e validar dados
        df_logs, efficient_frontier = _load_and_validate_data(output_dir, efficient_frontier_file)
        
        # 2. Calcular fronteira de Pareto com paralelização opcional
        print("[INFO] Calculando fronteira de Pareto...")
        if use_parallel and len(df_logs) > 10000:
            pareto_frontier = calculate_pareto_frontier_parallel(df_logs, n_processes)
        else:
            pareto_frontier = calculate_pareto_frontier(df_logs)
        print(f"[INFO] Fronteira de Pareto calculada com {len(pareto_frontier)} pontos")
        
        # 3. Calcular métricas com paralelização opcional
        df_logs, metrics = _calculate_metrics(df_logs, efficient_frontier, pareto_frontier, use_parallel)
        
        # 3.5 Calcular Sharpe
        print(f"\n[SHARPE] Calculando índice de Sharpe com taxa livre de risco = {risk_free_rate:.4f}")
        df_logs['sharpe'] = (df_logs['expected_return'] - risk_free_rate) / df_logs['risk']
        
        # Adicionar métricas de Sharpe
        metrics['best_sharpe'] = df_logs['sharpe'].max()
        metrics['avg_sharpe'] = df_logs['sharpe'].mean()
        metrics['median_sharpe'] = df_logs['sharpe'].median()
        
        if len(pareto_frontier) > 0:
            pareto_sharpe = (pareto_frontier['expected_return'] - risk_free_rate) / pareto_frontier['risk']
            metrics['best_sharpe_pareto'] = pareto_sharpe.max()
            metrics['avg_sharpe_pareto'] = pareto_sharpe.mean()
        
        print(f" - Melhor Sharpe (geral): {metrics['best_sharpe']:.6f}")
        if metrics.get('best_sharpe_pareto'):
            print(f" - Melhor Sharpe (Pareto): {metrics['best_sharpe_pareto']:.6f}")
        
        # 4. Calcular hipervolumes
        hv_metrics = _calculate_hypervolumes(df_logs, pareto_frontier)
        metrics.update(hv_metrics)
        
        # 5. Adicionar métricas básicas
        metrics.update({
            'total_evaluations': len(df_logs),
            'pareto_frontier_size': len(pareto_frontier),
            'best_sharpe': df_logs['sharpe'].max() if 'sharpe' in df_logs.columns else None,
            'processing_mode': 'parallel' if use_parallel else 'sequential',
            'n_processes_used': n_processes if use_parallel else 1
        })
        
        # 6. Salvar dados
        _save_analysis_results(output_dir, metrics, pareto_frontier)
        
        # 7. Gerar gráficos (apenas se necessário)
        print(" [INFO] Gerando gráficos...")
        plot_frontiers_comparison(
            output_dir, df_logs, efficient_frontier, pareto_frontier
        )
        
        # 8. Exibir resumo
        total_time = time.time() - start_time
        print(f"\n[INFO] Tempo total de processamento: {total_time:.2f}s")
        _display_metrics_summary(metrics)
        
        print("="*60)
        print(f"[INFO] Análise completa! Resultados salvos em: {output_dir}")
        
        # ⭐ GARBAGE COLLECTION: Liberar memória ao final da análise
        del df_logs, pareto_frontier, efficient_frontier
        gc.collect()
        print("[MEMORY] Garbage collection executado")
        
        return metrics
        
    except Exception as e:
        print(f"[ERROR] Erro durante análise: {e}")
        # ⭐ Mesmo em erro, tentar limpar memória
        gc.collect()
        raise

def _save_analysis_results(output_dir, metrics, pareto_frontier):
    """Salva os resultados da análise."""
    print("[INFO] Salvando métricas e populações...")
    
    # Salvar métricas
    metrics_path = save_metrics_to_csv(output_dir, metrics)
    print(f"[INFO] Métricas salvas em: {metrics_path}")
    
    # Salvar populações
    save_filtered_populations(
        output_dir,
        pareto_frontier=pareto_frontier
    )

def _display_metrics_summary(metrics):
    """Exibe resumo das métricas calculadas."""
    print("\n" + "="*60)
    print("[INFO] RESUMO DAS MÉTRICAS CALCULADAS")
    print("="*60)
    print(f"[INFO] Total de avaliações: {metrics['total_evaluations']}")
    print(f"[INFO] Tamanho da fronteira de Pareto: {metrics['pareto_frontier_size']}")
    
    if 'processing_mode' in metrics:
        mode_icon = "🚀" if metrics['processing_mode'] == 'parallel' else "🐌"
        print(f"{mode_icon} Modo de processamento: {metrics['processing_mode']}")
        if metrics['processing_mode'] == 'parallel':
            print(f" [INFO] Processos utilizados: {metrics.get('n_processes_used', 'N/A')}")
    
    if metrics['igd_plus'] is not None:
        print(f"[INFO] IGD+: {metrics['igd_plus']:.12e}")
    
    if metrics['avg_interpolation_error'] is not None:
        print(f"[INFO] Erro médio de interpolação: {metrics['avg_interpolation_error']:.4e}")
        print(f"[INFO] Erro mediano de interpolação: {metrics['median_interpolation_error']:.4e}")
    
    print(f"[INFO] Hipervolume geral: {metrics['hypervolume_general']:.12e}")
    
    if metrics.get('hypervolume_pareto') is not None:
        print(f"[INFO] Hipervolume fronteira de Pareto: {metrics['hypervolume_pareto']:.12e}")
    
    if metrics['best_sharpe'] is not None:
        print(f"[INFO] Melhor Sharpe ratio: {metrics['best_sharpe']:.12f}")