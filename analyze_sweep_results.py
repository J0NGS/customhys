#!/usr/bin/env python3
"""
⭐ ANÁLISE PARALELA DE RESULTADOS DE SWEEP (PORTA-ANÁLISE)
===========================================================

Executa portfolio_analyzer.py em cada lambda de um sweep em PARALELO.
Gera análises detalhadas nas pastas de cada lambda sem re-executar testes.

Suporta dois modos de paralelismo:
  1. Externa (nível sweep): Múltiplos lambdas processados em paralelo
  2. Interna (nível lambda): Cada lambda pode usar paralelismo em portfolio_analyzer

Uso:
    python analyze_sweep_results.py <sweep_dir> [--frontier-file FRONTIER] [--workers N] [--no-inner-parallel]

Exemplo:
    python analyze_sweep_results.py 20260109_193058_port2_random_lambda_sweep_50 \
        --frontier-file portef2.txt \
        --workers 4
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time

import pandas as pd
import numpy as np

# Importar funções auxiliares
sys.path.insert(0, os.path.dirname(__file__))
from portfolio_utils.config_utils import _save_json
from portfolio_utils.portfolio_analyzer import analyze_portfolio_results

warnings.filterwarnings('ignore')


def _analyze_lambda_worker(args: Tuple) -> Dict:
    """
    ⭐ WORKER para processar um lambda em paralelo
    
    Executado em processo separado (sem conflito de GIL).
    Cada worker chama portfolio_analyzer de forma SEQUENCIAL (n_processes=1).
    
    ⚠️ CRÍTICO: Paralelismo apenas no nível do sweep (entre lambdas).
    Paralelismo aninhado (dentro do lambda) é DESABILITADO para evitar explosão de processos.
    
    Args:
        args: Tupla (lambda_val, lambda_dir, frontier_file, risk_free_rate, use_parallel_inner)
        
    Returns:
        Dicionário com resultado da análise
    """
    lambda_val, lambda_dir, frontier_file, risk_free_rate, use_parallel_inner = args
    
    result = {
        'lambda': lambda_val,
        'status': 'pending',
        'message': '',
        'output_dir': str(lambda_dir),
        'worker_pid': os.getpid()
    }
    
    # Verificar se tem execution_logs.parquet
    log_file = Path(lambda_dir) / "execution_logs.parquet"
    if not log_file.exists():
        result['status'] = 'skipped'
        result['message'] = 'Sem execution_logs.parquet'
        return result
    
    try:
        print(f"[WORKER {result['worker_pid']}] λ={lambda_val:.4f}: Iniciando análise...", flush=True)
        
        # ⭐ CRÍTICO: FORÇAR SEQUENCIAL
        # Paralelismo DEVE ocorrer apenas no nível sweep (entre lambdas).
        # Paralelismo aninhado causa explosão de processos e OOM.
        # Sempre usa n_processes=1, ignore use_parallel_inner
        
        analyze_portfolio_results(
            output_dir=str(lambda_dir),
            efficient_frontier_file=frontier_file,
            use_parallel=False,  # ⭐ FORÇADO: Sem paralelismo interno
            n_processes=1,       # ⭐ FORÇADO: 1 processo sequencial
            risk_free_rate=risk_free_rate
        )
        
        result['status'] = 'success'
        result['message'] = 'Análise concluída'
        print(f"[WORKER {result['worker_pid']}] λ={lambda_val:.4f}: ✓ Sucesso", flush=True)
        
    except Exception as e:
        result['status'] = 'error'
        result['message'] = str(e)
        print(f"[WORKER {result['worker_pid']}] λ={lambda_val:.4f}: ✗ Erro - {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    finally:
        # ⭐ GARBAGE COLLECTION: Liberar memória deste worker
        gc.collect()
    
    return result


class SweepAnalyzer:
    """Executa análise paralela de múltiplos lambdas."""
    
    def __init__(self, sweep_dir: str, frontier_file: Optional[str] = None, 
                 risk_free_rate: float = 0.0057, n_workers: Optional[int] = None,
                 use_parallel_inner: bool = True):
        """
        Inicializa analisador de sweep paralelo.
        
        Args:
            sweep_dir: Diretório contendo resultados do sweep
            frontier_file: Arquivo com portfólio de fronteira (opcional)
            risk_free_rate: Taxa livre de risco
            n_workers: Número de workers paralelos (None = auto, cpu_count()//2)
            use_parallel_inner: Se True, cada lambda pode usar até 2 processos internos
                               (padrão: True, mas com limite para evitar overload)
        """
        self.sweep_dir = Path(sweep_dir)
        if not self.sweep_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {sweep_dir}")
        
        # ⭐ Configurar número de workers
        if n_workers is None:
            n_workers = max(1, cpu_count() // 2)  # Conservador: metade dos CPUs
        self.n_workers = min(n_workers, cpu_count())  # Limitar ao máximo disponível
        
        self.frontier_file = frontier_file
        self.risk_free_rate = risk_free_rate
        self.use_parallel_inner = use_parallel_inner
        self.lambda_dirs: Dict[float, Path] = {}
        self.analysis_results: Dict[float, Dict] = {}
        
        print(f"[INIT] Analisador de sweep paralelo inicializado")
        print(f"[INFO] Diretório: {self.sweep_dir.name}")
        print(f"[INFO] Workers (nível sweep): {self.n_workers}/{cpu_count()} CPUs")
        print(f"[INFO] Processos por lambda: {'até 2' if use_parallel_inner else '1 (sequencial)'}")
        print(f"[INFO] Fronteira: {frontier_file or 'Nenhuma'}")
        print(f"[INFO] Taxa livre de risco: {risk_free_rate}")
        
        self._discover_lambda_dirs()
    
    def _discover_lambda_dirs(self) -> None:
        """Descobre todos os diretórios lambda_*.* no sweep."""
        lambda_dirs = sorted(
            [d for d in self.sweep_dir.iterdir() if d.is_dir() and d.name.startswith('lambda_')],
            key=lambda p: float(p.name.split('_')[1])
        )
        
        for lambda_dir in lambda_dirs:
            try:
                lambda_val = float(lambda_dir.name.split('_')[1])
                self.lambda_dirs[lambda_val] = lambda_dir
            except (ValueError, IndexError):
                pass
        
        print(f"[INFO] Encontrados {len(self.lambda_dirs)} lambdas para análise")
    
    def analyze_all_parallel(self) -> None:
        """
        ⭐ Executa análise em PARALELO em todos os lambdas do sweep.
        
        Usa multiprocessing.Pool com n_workers processos independentes.
        Cada worker executa portfolio_analyzer com até 2 processos internos.
        """
        print(f"\n{'='*70}")
        print(f"[PROCESS] Analisando {len(self.lambda_dirs)} lambdas em PARALELO")
        print(f"[INFO] Workers: {self.n_workers}, Processos/lambda: {'até 2' if self.use_parallel_inner else '1'}")
        print(f"{'='*70}\n")
        
        # ⭐ Preparar argumentos para cada lambda
        worker_args = [
            (lambda_val, str(lambda_dir), self.frontier_file, self.risk_free_rate, self.use_parallel_inner)
            for lambda_val, lambda_dir in sorted(self.lambda_dirs.items())
        ]
        
        start_time = time.time()
        
        # ⭐ Executar Pool de workers
        try:
            with Pool(processes=self.n_workers, maxtasksperchild=1) as pool:
                # Usar map para processar todos os lambdas em paralelo
                # maxtasksperchild=1 força liberação de memória do SO após cada worker
                results = pool.map(_analyze_lambda_worker, worker_args)
            
            # Armazenar resultados
            for result in results:
                self.analysis_results[result['lambda']] = result
            
            # Exibir progresso
            for i, result in enumerate(sorted(results, key=lambda r: r['lambda']), 1):
                status_icon = {
                    'success': '✓',
                    'error': '✗',
                    'skipped': '-',
                    'pending': '?'
                }.get(result['status'], '?')
                
                print(f"  [{i}/{len(results)}] {status_icon} λ={result['lambda']:.4f} (PID:{result.get('worker_pid', 'N/A')}): {result['message']}")
        
        except KeyboardInterrupt:
            print("\n[WARNING] Análise interrompida pelo usuário")
            raise
        except Exception as e:
            print(f"\n[ERROR] Erro durante execução paralela: {e}")
            raise
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"[DONE] Análise paralela concluída em {elapsed_time:.2f}s")
        print(f"{'='*70}")

    
    def print_summary(self) -> None:
        """Imprime resumo da análise."""
        if not self.analysis_results:
            print("[INFO] Nenhuma análise executada")
            return
        
        success = sum(1 for r in self.analysis_results.values() if r['status'] == 'success')
        errors = sum(1 for r in self.analysis_results.values() if r['status'] == 'error')
        skipped = sum(1 for r in self.analysis_results.values() if r['status'] == 'skipped')
        
        print(f"\n[SUMMARY] Resumo da Análise")
        print(f"{'='*70}")
        print(f"Total de lambdas: {len(self.analysis_results)}")
        print(f"  ✓ Sucesso:  {success}")
        print(f"  ✗ Erros:    {errors}")
        print(f"  - Pulados:  {skipped}")
        print(f"{'='*70}")
        
        if errors > 0:
            print(f"\n[WARNING] Lambdas com erro:")
            for lambda_val, result in sorted(self.analysis_results.items()):
                if result['status'] == 'error':
                    print(f"  λ={lambda_val:.4f}: {result['message']}")
    
    def save_summary(self, output_file: Optional[str] = None) -> str:
        """
        Salva resumo da análise em JSON.
        
        Args:
            output_file: Arquivo de saída (padrão: sweep_dir/analysis_summary.json)
            
        Returns:
            Caminho do arquivo salvo
        """
        if output_file is None:
            output_file = self.sweep_dir / "analysis_summary.json"
        else:
            output_file = Path(output_file)
        
        # Contar resultados
        success = sum(1 for r in self.analysis_results.values() if r['status'] == 'success')
        errors = sum(1 for r in self.analysis_results.values() if r['status'] == 'error')
        skipped = sum(1 for r in self.analysis_results.values() if r['status'] == 'skipped')
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'sweep_dir': str(self.sweep_dir),
            'total_lambdas': len(self.analysis_results),
            'analysis_results': {
                'success': success,
                'errors': errors,
                'skipped': skipped
            },
            'parallelism': {
                'workers_used': self.n_workers,
                'total_cpus_available': cpu_count(),
                'use_inner_parallel': self.use_parallel_inner,
                'processes_per_lambda': '2 (limit)' if self.use_parallel_inner else '1 (sequential)'
            },
            'parameters': {
                'frontier_file': self.frontier_file,
                'risk_free_rate': self.risk_free_rate
            },
            'details': self.analysis_results
        }
        
        # Criar diretório se não existir
        output_file.parent.mkdir(parents=True, exist_ok=True)
        _save_json(summary, str(output_file), default=str)
        print(f"[SAVED] Resumo salvo em: {output_file}")
        
        return str(output_file)
    
    def run(self) -> None:
        """Executa análise paralela completa do sweep."""
        print(f"\n[START] Iniciando análise paralela de sweep")
        print(f"[INFO] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.analyze_all_parallel()
        self.print_summary()
        self.save_summary()
        
        print(f"\n[COMPLETE] Análise paralela finalizada!")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Executa análise paralela de portfólio em cada lambda de um sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Análise paralela básica (workers automático = CPU//2)
  python analyze_sweep_results.py 20260109_193058_port2_random_lambda_sweep_50
  
  # Com fronteira eficiente
  python analyze_sweep_results.py 20260109_193058_port2_random_lambda_sweep_50 \\
    --frontier-file portef2.txt
  
  # Com 8 workers (paralelismo máximo)
  python analyze_sweep_results.py 20260109_193058_port2_random_lambda_sweep_50 \\
    --frontier-file portef2.txt --workers 8
  
  # Sem paralelismo interno (sequencial por lambda, paralelo entre lambdas)
  python analyze_sweep_results.py 20260109_193058_port2_random_lambda_sweep_50 \\
    --no-inner-parallel
        """
    )
    
    parser.add_argument('sweep_dir', help='Diretório com resultados do sweep')
    parser.add_argument('--frontier-file', type=str, default=None,
                       help='Arquivo com portfólio de fronteira (ex: portef2.txt)')
    parser.add_argument('--workers', type=int, default=None,
                       help=f'Número de workers paralelos (padrão: CPU_count//2 = {max(1, cpu_count()//2)})')
    parser.add_argument('--risk-free-rate', type=float, default=0.0057,
                       help='Taxa livre de risco (padrão: 0.0057)')
    parser.add_argument('--no-inner-parallel', action='store_true',
                       help='Desabilitar paralelismo interno em portfolio_analyzer (cada lambda sequencial)')
    
    args = parser.parse_args()
    
    # Validar diretório
    if not os.path.isdir(args.sweep_dir):
        print(f"[ERROR] Diretório não encontrado: {args.sweep_dir}")
        sys.exit(1)
    
    # Validar frontier_file se fornecido
    if args.frontier_file and not os.path.exists(args.frontier_file):
        print(f"[WARNING] Arquivo de fronteira não encontrado: {args.frontier_file}")
    
    # Validar workers
    if args.workers is not None and args.workers <= 0:
        print(f"[ERROR] Número de workers deve ser > 0")
        sys.exit(1)
    
    # Executar análise
    try:
        analyzer = SweepAnalyzer(
            sweep_dir=args.sweep_dir,
            frontier_file=args.frontier_file,
            risk_free_rate=args.risk_free_rate,
            n_workers=args.workers,
            use_parallel_inner=not args.no_inner_parallel
        )
        analyzer.run()
        
        print(f"\n[SUCCESS] Análise paralela concluída com sucesso!")
        
    except Exception as e:
        print(f"[ERROR] Erro durante análise: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
