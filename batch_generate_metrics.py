#!/usr/bin/env python3
"""
Script para gerar métricas rápidas para todas as pastas de uma configuração.
"""

import os
import glob
import sys
import time
from generate_analysis_metrics_fast import generate_fast_metrics

def process_all_folders(base_pattern: str, frontier_file: str = "portef1.txt"):
    """
    Processa todas as pastas que seguem um padrão específico.
    
    Args:
        base_pattern: Padrão para buscar pastas (ex: "port*_config_*/*/")
        frontier_file: Arquivo da fronteira eficiente
    """
    print("🚀 GERAÇÃO EM LOTE DE MÉTRICAS RÁPIDAS")
    print("=" * 60)
    
    # Buscar todas as pastas que seguem o padrão
    folders = glob.glob(base_pattern)
    
    if not folders:
        print(f"❌ Nenhuma pasta encontrada com padrão: {base_pattern}")
        return
    
    print(f"📁 Encontradas {len(folders)} pastas para processar")
    print("-" * 60)
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, folder in enumerate(folders, 1):
        print(f"\n📦 [{i}/{len(folders)}] Processando: {folder}")
        
        try:
            # Verificar se já existe analysis_metrics.csv
            metrics_file = os.path.join(folder, "analysis_metrics.csv")
            if os.path.exists(metrics_file):
                print(f"⏭️ Pulando {folder} (métricas já existem)")
                continue
            
            # Gerar métricas
            metrics = generate_fast_metrics(folder, frontier_file)
            successful += 1
            
            print(f"✅ Concluído: {folder}")
            print(f"   • {metrics.get('total_evaluations', 0):,} avaliações")
            print(f"   • {metrics.get('pareto_frontier_size', 0):,} pontos Pareto")
            
        except Exception as e:
            print(f"❌ Erro em {folder}: {e}")
            failed += 1
        
        print("-" * 60)
    
    # Resumo final
    elapsed = time.time() - start_time
    print("\n🎉 PROCESSAMENTO CONCLUÍDO!")
    print(f"✅ Sucessos: {successful}")
    print(f"❌ Falhas: {failed}")
    print(f"⏱️ Tempo total: {elapsed:.1f}s")
    print(f"⚡ Tempo médio por pasta: {elapsed/len(folders):.1f}s")

def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gerar métricas rápidas em lote')
    parser.add_argument('-p', '--pattern', default='port*_config_*/*/',
                       help='Padrão de busca das pastas (padrão: port*_config_*/*/')
    parser.add_argument('-f', '--frontier', default='portef1.txt',
                       help='Arquivo da fronteira eficiente (padrão: portef1.txt)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Pular pastas que já têm analysis_metrics.csv')
    
    args = parser.parse_args()
    
    process_all_folders(args.pattern, args.frontier)

if __name__ == "__main__":
    # Configurações predefinidas para diferentes cenários
    scenarios = {
        "port1": "port1_config_alta/*/",
        "port3": "port3_config_alta/*/", 
        "port4": "port4_config_alta/*/",
        "all_alta": "port*_config_alta/*/",
        "default": "testes_config_default_lambda_0_5/*/",
        "fast": "testes_config_fast_lambda_0_5/*/"
    }
    
    print("📋 CENÁRIOS DISPONÍVEIS:")
    for name, pattern in scenarios.items():
        count = len(glob.glob(pattern))
        print(f"  {name}: {pattern} ({count} pastas)")
    
    print("\n" + "=" * 60)
    
    # Se executado sem argumentos, processar port1_config_alta
    if len(sys.argv) == 1:
        print("🎯 Executando cenário padrão: port1_config_alta")
        process_all_folders(scenarios["port1"], "portef1.txt")
    else:
        main()
