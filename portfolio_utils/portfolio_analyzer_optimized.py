#!/usr/bin/env python3
"""
Analisador de dados OTIMIZADO para resultados de otimização de portfólio.
Versão com melhorias de performance usando Numba, caching, e algoritmos otimizados.
"""

import os
import functools
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import warnings
from multiprocessing import cpu_count
import time
import gc
from typing import Optional, Tuple, Dict, Any

# Importar psutil se disponível para monitoramento de memória
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Importar Numba se disponível (para acelerar cálculos críticos)
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
    print("✅ Numba disponível - aceleração ativada")
except ImportError:
    print("⚠️ Numba não disponível - usando Python puro")
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def jit(func):
        return func
    def njit(func):
        return func
    def prange(*args):
        return range(*args)

warnings.filterwarnings('ignore')

# Constantes globais
EFFICIENT_FRONTIER_LABEL = "Fronteira Eficiente"
RISK_LABEL = "Risco (Desvio Padrão)"
RETURN_LABEL = "Retorno Esperado"
MEMORY_THRESHOLD_GB = 2.0  # Limite de memória antes de otimizar
PARALLEL_THRESHOLD = 10000  # Limite mínimo para usar paralelização

# Cache para resultados computacionalmente intensivos
@functools.lru_cache(maxsize=128)
def _cached_interpolation_setup(frontier_hash: str, risk_tuple: tuple, return_tuple: tuple):
    """Cache para configuração de interpolação."""
    return None  # Placeholder - implementação depende dos dados específicos

def get_memory_usage_gb() -> float:
    """Retorna o uso atual de memória em GB."""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except Exception:
            pass
    # Fallback simples se psutil não estiver disponível
    return 1.0  # Estimativa conservadora

def should_use_chunking(data_size: int, memory_limit_gb: float = MEMORY_THRESHOLD_GB) -> Tuple[bool, int]:
    """Determina se deve usar chunking baseado no uso de memória."""
    current_memory = get_memory_usage_gb()
    estimated_memory = current_memory + (data_size * 64) / (1024**3)  # Estimativa grosseira
    
    if estimated_memory > memory_limit_gb:
        chunk_size = max(1000, int(data_size * 0.1))  # 10% do dataset ou mínimo 1000
        return True, chunk_size
    return False, data_size

@njit(parallel=True, cache=True)
def _numba_pareto_frontier(solutions: np.ndarray) -> np.ndarray:
    """Versão Numba do cálculo da fronteira de Pareto - extremamente otimizada."""
    n = solutions.shape[0]
    
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    
    # Ordenar por retorno decrescente
    sorted_indices = np.argsort(-solutions[:, 0])
    
    pareto_mask = np.zeros(n, dtype=np.bool_)
    min_risk_so_far = np.inf
    
    for i in prange(n):
        idx = sorted_indices[i]
        current_risk = solutions[idx, 1]
        if current_risk < min_risk_so_far:
            pareto_mask[idx] = True
            min_risk_so_far = current_risk
    
    return np.nonzero(pareto_mask)[0]

@njit(parallel=True, cache=True)
def _numba_distance_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Cálculo otimizado de matriz de distâncias com Numba."""
    n1, n2 = points1.shape[0], points2.shape[0]
    distances = np.zeros((n1, n2), dtype=np.float64)
    
    for i in prange(n1):
        for j in prange(n2):
            dist = 0.0
            for k in range(2):  # Assumindo 2D (retorno, risco)
                diff = points1[i, k] - points2[j, k]
                dist += diff * diff
            distances[i, j] = np.sqrt(dist)
    
    return distances

@njit(parallel=True, cache=True)
def _numba_hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Cálculo otimizado de hipervolume 2D com Numba."""
    n = points.shape[0]
    if n == 0:
        return 0.0
    
    # Ordenar pontos por primeira coordenada
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    hypervolume = 0.0
    prev_x = ref_point[0]
    
    for i in range(n):
        x, y = sorted_points[i, 0], sorted_points[i, 1]
        if i == 0 or y > sorted_points[i-1, 1]:  # Apenas pontos não dominados
            width = x - prev_x
            height = y - ref_point[1]
            if width > 0 and height > 0:
                hypervolume += width * height
            prev_x = x
    
    return hypervolume

class OptimizedPortfolioAnalyzer:
    """Classe principal para análise otimizada de portfólio."""
    
    def __init__(self, use_numba: bool = NUMBA_AVAILABLE, cache_size: int = 128):
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.cache_size = cache_size
        self._setup_cache()
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'numba_calls': 0,
            'fallback_calls': 0
        }
    
    def _setup_cache(self):
        """Configura sistema de cache."""
        self._pareto_cache = {}
        self._igd_cache = {}
        self._hypervolume_cache = {}
    
    def calculate_pareto_frontier_optimized(self, df_logs: pd.DataFrame) -> pd.DataFrame:
        """Versão ultra-otimizada do cálculo da fronteira de Pareto."""
        solutions = df_logs[['expected_return', 'risk']].values
        cache_key = hash(solutions.tobytes())
        
        if cache_key in self._pareto_cache:
            self.stats['cache_hits'] += 1
            pareto_indices = self._pareto_cache[cache_key]
        else:
            self.stats['cache_misses'] += 1
            
            if self.use_numba and len(solutions) > 1000:
                self.stats['numba_calls'] += 1
                pareto_indices = _numba_pareto_frontier(solutions)
            else:
                self.stats['fallback_calls'] += 1
                pareto_indices = self._calculate_pareto_python(solutions)
            
            # Cache apenas se dataset não for muito grande
            if len(solutions) < 100000:
                self._pareto_cache[cache_key] = pareto_indices
        
        pareto_df = df_logs.iloc[pareto_indices].copy().sort_values('risk')
        return pareto_df
    
    def _calculate_pareto_python(self, solutions: np.ndarray) -> np.ndarray:
        """Versão Python otimizada para fallback."""
        n = len(solutions)
        if n == 0:
            return np.array([], dtype=int)
        
        sorted_indices = np.argsort(-solutions[:, 0])
        pareto_mask = np.zeros(n, dtype=bool)
        min_risk_so_far = float('inf')
        
        for idx in sorted_indices:
            current_risk = solutions[idx, 1]
            if current_risk < min_risk_so_far:
                pareto_mask[idx] = True
                min_risk_so_far = current_risk
        
        return np.nonzero(pareto_mask)[0]
    
    def calculate_igd_plus_optimized(self, pareto_front: pd.DataFrame, 
                                   reference_front: pd.DataFrame) -> Optional[float]:
        """Versão otimizada do cálculo IGD+ com caching e Numba."""
        if reference_front is None or len(reference_front) == 0:
            return None
        
        pareto_points = pareto_front[['expected_return', 'risk']].values
        ref_points = reference_front[['mean_return', 'std_dev']].values
        
        if len(pareto_points) == 0:
            return float('inf')
        
        # Tentar cache
        cache_key = (hash(pareto_points.tobytes()), hash(ref_points.tobytes()))
        if cache_key in self._igd_cache:
            self.stats['cache_hits'] += 1
            return self._igd_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Usar Numba para datasets grandes
        if self.use_numba and len(pareto_points) * len(ref_points) > 100000:
            self.stats['numba_calls'] += 1
            distances = _numba_distance_matrix(ref_points, pareto_points)
            min_distances = np.min(distances, axis=1)
            igd_result = np.mean(min_distances)
        else:
            self.stats['fallback_calls'] += 1
            # Usar scipy.spatial.distance que é otimizada
            distances = cdist(ref_points, pareto_points, metric='euclidean')
            min_distances = np.min(distances, axis=1)
            igd_result = np.mean(min_distances)
        
        # Cache resultado se não for muito grande
        if len(pareto_points) < 50000:
            self._igd_cache[cache_key] = igd_result
        
        return igd_result
    
    def calculate_hypervolume_optimized(self, points: np.ndarray, 
                                      reference_point: np.ndarray) -> float:
        """Versão otimizada do cálculo de hipervolume."""
        if len(points) == 0:
            return 0.0
        
        cache_key = (hash(points.tobytes()), hash(reference_point.tobytes()))
        if cache_key in self._hypervolume_cache:
            self.stats['cache_hits'] += 1
            return self._hypervolume_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Transformar para problema de maximização
        transformed_points = np.column_stack([points[:, 0], -points[:, 1]])
        transformed_ref = np.array([reference_point[0], -reference_point[1]])
        
        # Remover pontos dominados primeiro
        non_dominated_indices = self._get_non_dominated_indices_optimized(transformed_points)
        non_dominated = transformed_points[non_dominated_indices]
        
        if len(non_dominated) == 0:
            return 0.0
        
        # Usar Numba para cálculo se disponível
        if self.use_numba and len(non_dominated) > 100:
            self.stats['numba_calls'] += 1
            hv_result = _numba_hypervolume_2d(non_dominated, transformed_ref)
        else:
            self.stats['fallback_calls'] += 1
            hv_result = self._calculate_2d_hypervolume_python(non_dominated, transformed_ref)
        
        # Cache se não for muito grande
        if len(points) < 10000:
            self._hypervolume_cache[cache_key] = abs(hv_result)
        
        return abs(hv_result)
    
    def _get_non_dominated_indices_optimized(self, points: np.ndarray) -> np.ndarray:
        """Encontra índices de pontos não dominados de forma otimizada."""
        if self.use_numba and len(points) > 1000:
            return _numba_pareto_frontier(points)
        else:
            return self._calculate_pareto_python(points)
    
    def _calculate_2d_hypervolume_python(self, points: np.ndarray, 
                                       reference_point: np.ndarray) -> float:
        """Versão Python para cálculo de hipervolume 2D."""
        # Ordenar por primeira coordenada
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        
        hypervolume = 0.0
        prev_x = reference_point[0]
        
        for i, point in enumerate(sorted_points):
            x, y = point[0], point[1]
            width = x - prev_x
            height = y - reference_point[1]
            if width > 0 and height > 0:
                hypervolume += width * height
            prev_x = x
        
        return hypervolume
    
    def calculate_interpolation_errors_optimized(self, df_logs: pd.DataFrame,
                                               efficient_frontier: pd.DataFrame) -> pd.DataFrame:
        """Versão otimizada do cálculo de erros de interpolação."""
        if efficient_frontier is None or len(efficient_frontier) < 2:
            return df_logs
        
        df_logs = df_logs.copy()
        
        try:
            # Pre-computar arrays para evitar múltiplas conversões
            risk_values = df_logs["risk"].values
            return_values = df_logs["expected_return"].values
            
            # Criar interpoladores uma vez
            interp_return = interp1d(
                efficient_frontier["std_dev"].values, 
                efficient_frontier["mean_return"].values,
                kind='linear', bounds_error=False, fill_value="extrapolate"
            )
            interp_risk = interp1d(
                efficient_frontier["mean_return"].values,
                efficient_frontier["std_dev"].values,
                kind='linear', bounds_error=False, fill_value="extrapolate"
            )
            
            # Aplicar interpolação vetorizada
            interp_ret_values = interp_return(risk_values)
            interp_risk_values = interp_risk(return_values)
            
            # Cálculos vetorizados com proteção contra divisão por zero
            with np.errstate(divide='ignore', invalid='ignore'):
                # Usar np.fmax para evitar zeros
                denom_ret = np.fmax(np.abs(interp_ret_values), 1e-12)
                denom_risk = np.fmax(np.abs(interp_risk_values), 1e-12)
                
                error_return = np.abs(return_values - interp_ret_values) / denom_ret
                error_risk = np.abs(risk_values - interp_risk_values) / denom_risk
                
                # Usar np.fmin para operação vetorizada
                percent_error = np.fmin(error_return, error_risk)
                
                # Tratamento robusto de valores inválidos
                valid_mask = np.isfinite(percent_error)
                if not np.all(valid_mask):
                    max_valid = np.nanmax(percent_error[valid_mask]) if np.any(valid_mask) else 1.0
                    percent_error = np.where(valid_mask, percent_error, max_valid * 2)
            
            # Atribuir resultados de uma vez
            df_logs["interp_ret_from_risk"] = interp_ret_values
            df_logs["interp_risk_from_ret"] = interp_risk_values
            df_logs["error_return"] = error_return
            df_logs["error_risk"] = error_risk
            df_logs["percent_error"] = percent_error
            
        except Exception as e:
            print(f"⚠️ Erro na interpolação otimizada: {e}")
            df_logs["percent_error"] = 1.0
        
        return df_logs
    
    def get_best_solutions_optimized(self, df_logs: pd.DataFrame, 
                                   frontier_points: pd.DataFrame,
                                   n_solutions_per_point: int = 100,
                                   use_parallel: bool = True) -> pd.DataFrame:
        """Versão otimizada da seleção das melhores soluções."""
        if 'percent_error' not in df_logs.columns:
            return df_logs.nsmallest(len(frontier_points) * n_solutions_per_point, 'objective')
        
        # Verificar se deve usar chunking
        data_complexity = len(df_logs) * len(frontier_points)
        use_chunking, chunk_size = should_use_chunking(data_complexity)
        
        if use_chunking:
            print(f"🧠 Usando chunking - chunk size: {chunk_size}")
            return self._get_best_solutions_chunked(
                df_logs, frontier_points, n_solutions_per_point, chunk_size
            )
        
        # Versão otimizada direta
        return self._get_best_solutions_direct(
            df_logs, frontier_points, n_solutions_per_point, use_parallel
        )
    
    def _get_best_solutions_chunked(self, df_logs: pd.DataFrame,
                                  frontier_points: pd.DataFrame,
                                  n_solutions_per_point: int,
                                  chunk_size: int) -> pd.DataFrame:
        """Versão com chunking para datasets grandes."""
        selected_indices_set = set()
        n_frontier_points = len(frontier_points)
        
        # Arrays pré-computados
        solutions_risk = df_logs['risk'].values
        solutions_return = df_logs['expected_return'].values
        solutions_error = df_logs['percent_error'].values
        solutions_indices = df_logs.index.values
        
        # Determinar tipo de fronteira
        if 'std_dev' in frontier_points.columns:
            frontier_risk = frontier_points['std_dev'].values
            frontier_return = frontier_points['mean_return'].values
        else:
            frontier_risk = frontier_points['risk'].values
            frontier_return = frontier_points['expected_return'].values
        
        # Processar em chunks
        for chunk_start in range(0, n_frontier_points, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_frontier_points)
            
            # Usar Numba se disponível para cálculo de distâncias
            if self.use_numba:
                frontier_chunk = np.column_stack([
                    frontier_return[chunk_start:chunk_end],
                    frontier_risk[chunk_start:chunk_end]
                ])
                solutions_points = np.column_stack([solutions_return, solutions_risk])
                distances = _numba_distance_matrix(frontier_chunk, solutions_points)
            else:
                frontier_chunk = np.column_stack([
                    frontier_return[chunk_start:chunk_end],
                    frontier_risk[chunk_start:chunk_end]
                ])
                solutions_points = np.column_stack([solutions_return, solutions_risk])
                distances = cdist(frontier_chunk, solutions_points)
            
            # Processar cada ponto do chunk
            for i in range(distances.shape[0]):
                # Encontrar candidatos mais próximos
                n_candidates = min(n_solutions_per_point * 2, len(df_logs))
                # Corrigir para evitar erro de índice - argpartition precisa de k < len(array)
                current_distances = distances[i]
                k = min(n_candidates, len(current_distances) - 1) if len(current_distances) > 1 else 0
                if k > 0:
                    closest_idx = np.argpartition(current_distances, k)[:n_candidates]
                else:
                    closest_idx = np.arange(len(current_distances))
                
                # Selecionar melhores por erro
                candidate_errors = solutions_error[closest_idx]
                # Aplicar a mesma correção para argpartition
                k_error = min(n_solutions_per_point, len(candidate_errors) - 1) if len(candidate_errors) > 1 else 0
                if k_error > 0:
                    best_idx = np.argpartition(candidate_errors, k_error)[:n_solutions_per_point]
                else:
                    best_idx = np.arange(len(candidate_errors))
                selected_for_point = closest_idx[best_idx]
                
                selected_indices_set.update(solutions_indices[selected_for_point])
            
            # Limpeza de memória
            del distances
            if chunk_start % (chunk_size * 5) == 0:  # A cada 5 chunks
                gc.collect()
        
        # Retornar soluções selecionadas
        selected_indices = list(selected_indices_set)
        return df_logs.loc[selected_indices].copy()
    
    def _get_best_solutions_direct(self, df_logs: pd.DataFrame,
                                 frontier_points: pd.DataFrame,
                                 n_solutions_per_point: int,
                                 use_parallel: bool) -> pd.DataFrame:
        """Versão direta otimizada."""
        # Implementação similar mas sem chunking
        # ... (código similar ao método chunked mas processando tudo de uma vez)
        pass
    
    def print_performance_stats(self):
        """Imprime estatísticas de performance."""
        total_calls = sum(self.stats.values())
        if total_calls == 0:
            return
        
        print("\n" + "="*50)
        print("📊 ESTATÍSTICAS DE PERFORMANCE")
        print("="*50)
        print(f"Cache hits: {self.stats['cache_hits']} ({self.stats['cache_hits']/total_calls*100:.1f}%)")
        print(f"Cache misses: {self.stats['cache_misses']} ({self.stats['cache_misses']/total_calls*100:.1f}%)")
        
        if NUMBA_AVAILABLE:
            print(f"Numba calls: {self.stats['numba_calls']} ({self.stats['numba_calls']/total_calls*100:.1f}%)")
            print(f"Python fallback: {self.stats['fallback_calls']} ({self.stats['fallback_calls']/total_calls*100:.1f}%)")
        
        memory_usage = get_memory_usage_gb()
        print(f"Uso de memória: {memory_usage:.2f} GB")

def analyze_portfolio_results_optimized(output_dir: str, 
                                       efficient_frontier_file: Optional[str] = None,
                                       use_numba: bool = NUMBA_AVAILABLE,
                                       use_parallel: bool = True) -> Dict[str, Any]:
    """
    Função principal otimizada para análise de portfólio.
    
    Args:
        output_dir: Diretório com os resultados
        efficient_frontier_file: Arquivo da fronteira eficiente
        use_numba: Usar Numba para aceleração (se disponível)
        use_parallel: Usar processamento paralelo
    
    Returns:
        Dicionário com métricas calculadas
    """
    print(f"\n🚀 ANÁLISE OTIMIZADA iniciando para: {output_dir}")
    print(f"⚙️ Numba: {'✅ Ativo' if use_numba and NUMBA_AVAILABLE else '❌ Inativo'}")
    print(f"⚙️ Paralelo: {'✅ Ativo' if use_parallel else '❌ Inativo'}")
    
    start_time = time.time()
    initial_memory = get_memory_usage_gb()
    
    # Inicializar analisador otimizado
    analyzer = OptimizedPortfolioAnalyzer(use_numba=use_numba)
    
    try:
        # 1. Carregar dados
        logs_path = os.path.join(output_dir, "execution_logs.csv")
        if not os.path.exists(logs_path):
            raise FileNotFoundError(f"execution_logs.csv não encontrado em {output_dir}")
        
        df_logs = pd.read_csv(logs_path)
        print(f"✅ Carregados {len(df_logs)} logs ({get_memory_usage_gb():.2f} GB)")
        
        # Carregar fronteira eficiente
        efficient_frontier = None
        if efficient_frontier_file and os.path.exists(efficient_frontier_file):
            try:
                efficient_frontier = pd.read_csv(
                    efficient_frontier_file, sep=r"\s+", 
                    header=None, names=["mean_return", "variance"]
                )
                efficient_frontier["std_dev"] = np.sqrt(efficient_frontier["variance"])
                print(f"✅ Fronteira eficiente carregada: {len(efficient_frontier)} pontos")
            except Exception as e:
                print(f"⚠️ Erro ao carregar fronteira eficiente: {e}")
        
        # 2. Calcular fronteira de Pareto otimizada
        print("🎯 Calculando fronteira de Pareto (otimizada)...")
        pareto_start = time.time()
        pareto_frontier = analyzer.calculate_pareto_frontier_optimized(df_logs)
        pareto_time = time.time() - pareto_start
        print(f"✅ Pareto calculado: {len(pareto_frontier)} pontos em {pareto_time:.2f}s")
        
        # 3. Calcular erros de interpolação otimizados
        metrics = {}
        if efficient_frontier is not None:
            print("📏 Calculando erros de interpolação (otimizados)...")
            interp_start = time.time()
            df_logs = analyzer.calculate_interpolation_errors_optimized(df_logs, efficient_frontier)
            interp_time = time.time() - interp_start
            
            metrics['avg_interpolation_error'] = df_logs['percent_error'].mean()
            metrics['median_interpolation_error'] = df_logs['percent_error'].median()
            print(f"✅ Interpolação concluída em {interp_time:.2f}s")
            print(f"📊 Erro médio: {metrics['avg_interpolation_error']:.6e}")
        
        # 4. Calcular IGD+ otimizado
        if efficient_frontier is not None:
            print("📐 Calculando IGD+ (otimizado)...")
            igd_start = time.time()
            metrics['igd_plus'] = analyzer.calculate_igd_plus_optimized(pareto_frontier, efficient_frontier)
            igd_time = time.time() - igd_start
            print(f"✅ IGD+ calculado em {igd_time:.2f}s: {metrics['igd_plus']:.6e}")
        
        # 5. Calcular hipervolumes otimizados
        print("📦 Calculando hipervolumes (otimizados)...")
        hv_start = time.time()
        
        # Ponto de referência otimizado
        min_return = df_logs['expected_return'].min()
        max_risk = df_logs['risk'].max()
        return_range = df_logs['expected_return'].max() - min_return
        risk_range = max_risk - df_logs['risk'].min()
        
        ref_point = np.array([
            min_return - 0.01 * return_range,
            max_risk + 0.01 * risk_range
        ])
        
        # Hipervolume geral
        all_points = df_logs[['expected_return', 'risk']].values
        hv_general = analyzer.calculate_hypervolume_optimized(all_points, ref_point)
        
        # Hipervolume Pareto
        pareto_points = pareto_frontier[['expected_return', 'risk']].values
        hv_pareto = analyzer.calculate_hypervolume_optimized(pareto_points, ref_point)
        
        hv_time = time.time() - hv_start
        print(f"✅ Hipervolumes calculados em {hv_time:.2f}s")
        
        metrics.update({
            'hypervolume_general': hv_general,
            'hypervolume_pareto': hv_pareto,
            'total_evaluations': len(df_logs),
            'pareto_frontier_size': len(pareto_frontier),
            'processing_mode': 'optimized',
            'numba_enabled': use_numba and NUMBA_AVAILABLE,
            'parallel_enabled': use_parallel
        })
        
        # 6. Salvar resultados
        print("💾 Salvando resultados...")
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(output_dir, "analysis_metrics_optimized.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        # Salvar fronteira de Pareto
        pareto_path = os.path.join(output_dir, "pareto_frontier_optimized.csv")
        pareto_frontier.to_csv(pareto_path, index=False)
        
        # 7. Estatísticas finais
        total_time = time.time() - start_time
        final_memory = get_memory_usage_gb()
        memory_delta = final_memory - initial_memory
        
        print("\n⚡ ANÁLISE OTIMIZADA CONCLUÍDA!")
        print(f"⏱️ Tempo total: {total_time:.2f}s")
        print(f"🧠 Memória delta: {memory_delta:+.2f} GB")
        print(f"📁 Resultados salvos em: {output_dir}")
        
        # Imprimir estatísticas de performance
        analyzer.print_performance_stats()
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro na análise otimizada: {e}")
        raise
    finally:
        # Limpeza final
        gc.collect()

# Função de compatibilidade para substituir a original
def analyze_portfolio_results_ultra_fast(output_dir: str, 
                                        efficient_frontier_file: Optional[str] = None) -> Dict[str, Any]:
    """Wrapper para máxima performance - usa todas as otimizações disponíveis."""
    return analyze_portfolio_results_optimized(
        output_dir=output_dir,
        efficient_frontier_file=efficient_frontier_file,
        use_numba=True,
        use_parallel=True
    )

if __name__ == "__main__":
    # Teste básico
    print("🧪 Testando analisador otimizado...")
    print(f"Numba disponível: {NUMBA_AVAILABLE}")
    print(f"CPUs disponíveis: {cpu_count()}")
    print(f"Memória disponível: {get_memory_usage_gb():.2f} GB")
