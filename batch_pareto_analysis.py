#!/usr/bin/env python3
"""
Script para executar análise de fronteira de Pareto em múltiplas pastas de sweep
Organiza os resultados com padrão: test_results_port_X_<tipo>
"""

import os
import subprocess
import glob
import re
from pathlib import Path
import argparse


def detect_instance_from_folder(folder_name):
    """
    Detecta a instância (port1, port2, etc.) a partir do nome da pasta.
    
    Args:
        folder_name: Nome da pasta de sweep
    
    Returns:
        str: Instância detectada (ex: 'port1') ou None se não detectar
    """
    # Buscar padrão port seguido de número
    match = re.search(r'port(\d+)', folder_name.lower())
    if match:
        return f"port{match.group(1)}"
    return None


def detect_sweep_type(folder_name):
    """
    Detecta o tipo de sweep (fast, default, etc.) a partir do nome da pasta.
    
    Args:
        folder_name: Nome da pasta de sweep
    
    Returns:
        str: Tipo detectado ou 'unknown'
    """
    folder_lower = folder_name.lower()
    
    if 'fast' in folder_lower:
        return 'fast'
    elif 'default' in folder_lower:
        return 'default'
    elif 'optimized' in folder_lower:
        return 'optimized'
    else:
        # Tentar extrair tipo dos componentes da pasta
        parts = folder_name.split('_')
        for part in parts:
            if part.lower() in ['fast', 'default', 'optimized', 'beta', 'exponetial', 
                               'halton', 'lhc', 'lognormal', 'normal', 'random', 
                               'rayleigh', 'sobol', 'weibull']:
                return part.lower()
        return 'unknown'


def get_efficient_frontier_file(instance):
    """
    Retorna o arquivo da fronteira eficiente para uma instância.
    
    Args:
        instance: Instância (ex: 'port1')
    
    Returns:
        str: Caminho para o arquivo da fronteira eficiente ou None se não existir
    """
    frontier_map = {
        'port1': 'portef1.txt',
        'port2': 'portef2.txt', 
        'port3': 'portef3.txt',
        'port4': 'portef4.txt',
        'port5': 'portef5.txt'
    }
    
    frontier_file = frontier_map.get(instance)
    if frontier_file and os.path.exists(frontier_file):
        return frontier_file
    return None


def find_sweep_folders(base_dir="."):
    """
    Encontra todas as pastas de sweep no diretório base.
    
    Args:
        base_dir: Diretório base para buscar (padrão: diretório atual)
    
    Returns:
        list: Lista de tuplas (pasta, instância, tipo)
    """
    # Padrão para pastas de sweep: YYYYMMDD_HHMMSS_portX_<tipo>_lambda_sweep_N
    pattern = os.path.join(base_dir, "*_port*_*lambda_sweep*")
    sweep_folders = glob.glob(pattern)
    
    results = []
    for folder in sweep_folders:
        if os.path.isdir(folder):
            folder_name = os.path.basename(folder)
            instance = detect_instance_from_folder(folder_name)
            sweep_type = detect_sweep_type(folder_name)
            
            if instance:
                results.append((folder, instance, sweep_type))
            else:
                print(f"⚠️ Não foi possível detectar instância para: {folder_name}")
    
    return results


def run_pareto_analysis(sweep_folders, efficient_frontier_file, output_dir):
    """
    Executa o script de análise de fronteira de Pareto.
    
    Args:
        sweep_folders: Lista de pastas de sweep para analisar
        efficient_frontier_file: Arquivo da fronteira eficiente
        output_dir: Diretório de saída
    
    Returns:
        bool: True se executou com sucesso, False caso contrário
    """
    try:
        # Comando para executar o script de análise
        cmd = [
            "python", 
            "pareto_frontier_analysis.py",
            efficient_frontier_file
        ]
        
        # Adicionar todas as pastas de sweep como argumentos individuais
        if isinstance(sweep_folders, list):
            cmd.extend(sweep_folders)
        else:
            cmd.append(sweep_folders)
        
        # Adicionar argumentos de saída
        cmd.extend(["--output", output_dir])
        
        print(f"🚀 Executando: {' '.join(cmd)}")
        
        # Executar o comando
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print(f"✅ Análise concluída com sucesso para: {len(sweep_folders) if isinstance(sweep_folders, list) else 1} pasta(s)")
            return True
        else:
            print("❌ Erro na execução:")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exceção durante execução: {e}")
        return False


def group_sweeps_by_instance_and_type(sweep_folders):
    """
    Agrupa as pastas de sweep por instância e tipo.
    
    Args:
        sweep_folders: Lista de tuplas (pasta, instância, tipo)
    
    Returns:
        dict: Estrutura {instância: {tipo: [lista_de_pastas]}}
    """
    grouped = {}
    
    for folder, instance, sweep_type in sweep_folders:
        if instance not in grouped:
            grouped[instance] = {}
        if sweep_type not in grouped[instance]:
            grouped[instance][sweep_type] = []
        grouped[instance][sweep_type].append(folder)
    
    return grouped


def run_batch_analysis(sweep_folders, dry_run=False):
    """
    Executa análise em lote para todas as pastas de sweep.
    
    Args:
        sweep_folders: Lista de tuplas (pasta, instância, tipo)
        dry_run: Se True, apenas mostra o que seria executado sem executar
    
    Returns:
        dict: Estatísticas da execução
    """
    if not sweep_folders:
        print("❌ Nenhuma pasta de sweep encontrada!")
        return {}
    
    # Agrupar por instância e tipo
    grouped = group_sweeps_by_instance_and_type(sweep_folders)
    
    stats = {
        'total_analyses': 0,
        'successful': 0,
        'failed': 0,
        'skipped': 0
    }
    
    print("\n📊 RESUMO DAS ANÁLISES A EXECUTAR:")
    print("=" * 60)
    
    for instance in sorted(grouped.keys()):
        print(f"\n🎯 Instância: {instance}")
        for sweep_type in sorted(grouped[instance].keys()):
            folders = grouped[instance][sweep_type]
            print(f"   📁 Tipo {sweep_type}: {len(folders)} pastas")
            
            # Definir nome do diretório de saída
            output_dir = f"test_results_{instance}_{sweep_type}"
            
            # Obter arquivo da fronteira eficiente
            efficient_frontier_file = get_efficient_frontier_file(instance)
            
            if not efficient_frontier_file:
                print(f"   ⚠️ Fronteira eficiente não encontrada para {instance}")
                stats['skipped'] += 1
                continue
            
            print(f"   📈 Fronteira eficiente: {efficient_frontier_file}")
            print(f"   📁 Saída: {output_dir}")
            
            if dry_run:
                print(f"   🔍 [DRY RUN] Executaria análise com {len(folders)} pastas")
                stats['total_analyses'] += 1
            else:
                # Executar análise agregada para todas as pastas deste tipo
                stats['total_analyses'] += 1
                success = run_pareto_analysis(folders, efficient_frontier_file, output_dir)
                
                if success:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
    
    return stats


def run_individual_analysis(sweep_folders, dry_run=False):
    """
    Executa análise individual para cada pasta de sweep.
    
    Args:
        sweep_folders: Lista de tuplas (pasta, instância, tipo)
        dry_run: Se True, apenas mostra o que seria executado sem executar
    
    Returns:
        dict: Estatísticas da execução
    """
    stats = {
        'total_analyses': 0,
        'successful': 0,
        'failed': 0,
        'skipped': 0
    }
    
    print("\n📊 ANÁLISES INDIVIDUAIS:")
    print("=" * 60)
    
    for folder, instance, sweep_type in sweep_folders:
        folder_name = os.path.basename(folder)
        output_dir = f"test_results_{folder_name}"
        
        # Obter arquivo da fronteira eficiente
        efficient_frontier_file = get_efficient_frontier_file(instance)
        
        if not efficient_frontier_file:
            print(f"⚠️ Pulando {folder_name}: fronteira eficiente não encontrada para {instance}")
            stats['skipped'] += 1
            continue
        
        print(f"\n🔍 Processando: {folder_name}")
        print(f"   📈 Fronteira: {efficient_frontier_file}")
        print(f"   📁 Saída: {output_dir}")
        
        if dry_run:
            print("   🔍 [DRY RUN] Executaria análise")
            stats['total_analyses'] += 1
        else:
            stats['total_analyses'] += 1
            success = run_pareto_analysis([folder], efficient_frontier_file, output_dir)
            
            if success:
                stats['successful'] += 1
            else:
                stats['failed'] += 1
    
    return stats


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Executa análise de fronteira de Pareto em múltiplas pastas de sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python batch_pareto_analysis.py                    # Analisa todas as pastas (modo agregado)
  python batch_pareto_analysis.py --individual       # Analisa cada pasta individualmente
  python batch_pareto_analysis.py --dry-run          # Mostra o que seria executado
  python batch_pareto_analysis.py --base-dir /path   # Busca em diretório específico
        """
    )
    
    parser.add_argument(
        '--base-dir', '-d',
        default='.',
        help='Diretório base para buscar pastas de sweep (padrão: diretório atual)'
    )
    
    parser.add_argument(
        '--individual', '-i',
        action='store_true',
        help='Executa análise individual para cada pasta (ao invés de agregar por tipo)'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Mostra o que seria executado sem executar de fato'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        help='Padrão específico para filtrar pastas (ex: "*port3*")'
    )
    
    args = parser.parse_args()
    
    print("🚀 SCRIPT DE ANÁLISE EM LOTE - FRONTEIRA DE PARETO")
    print("=" * 60)
    print(f"📁 Diretório base: {os.path.abspath(args.base_dir)}")
    print(f"🔧 Modo: {'Individual' if args.individual else 'Agregado por tipo'}")
    if args.dry_run:
        print("🔍 Modo DRY RUN - apenas simulação")
    if args.pattern:
        print(f"🔍 Filtro: {args.pattern}")
    
    # Verificar se o script de análise existe
    if not os.path.exists("pareto_frontier_analysis.py"):
        print("❌ Script 'pareto_frontier_analysis.py' não encontrado no diretório atual!")
        return
    
    # Encontrar todas as pastas de sweep
    print(f"\n🔍 Buscando pastas de sweep em: {args.base_dir}")
    sweep_folders = find_sweep_folders(args.base_dir)
    
    # Aplicar filtro se especificado
    if args.pattern:
        import fnmatch
        filtered_folders = []
        for folder, instance, sweep_type in sweep_folders:
            if fnmatch.fnmatch(os.path.basename(folder), args.pattern):
                filtered_folders.append((folder, instance, sweep_type))
        sweep_folders = filtered_folders
        print(f"🔍 Filtro aplicado, restaram {len(sweep_folders)} pastas")
    
    if not sweep_folders:
        print("❌ Nenhuma pasta de sweep encontrada!")
        print("💡 Verifique se há pastas com padrão: YYYYMMDD_HHMMSS_portX_<tipo>_lambda_sweep_N")
        return
    
    print(f"✅ Encontradas {len(sweep_folders)} pastas de sweep")
    
    # Verificar arquivos de fronteira eficiente
    instances_found = {instance for _, instance, _ in sweep_folders}
    print("\n📈 Verificando fronteiras eficientes:")
    missing_frontiers = []
    
    for instance in sorted(instances_found):
        frontier_file = get_efficient_frontier_file(instance)
        if frontier_file:
            print(f"   ✅ {instance}: {frontier_file}")
        else:
            print(f"   ❌ {instance}: arquivo não encontrado")
            missing_frontiers.append(instance)
    
    if missing_frontiers:
        print(f"\n⚠️ Fronteiras eficientes ausentes para: {', '.join(missing_frontiers)}")
        print("💡 Estas instâncias serão puladas na análise")
    
    # Executar análises
    if args.individual:
        stats = run_individual_analysis(sweep_folders, args.dry_run)
    else:
        stats = run_batch_analysis(sweep_folders, args.dry_run)
    
    # Relatório final
    print("\n📊 RELATÓRIO FINAL:")
    print("=" * 40)
    print(f"📈 Total de análises: {stats['total_analyses']}")
    print(f"✅ Sucessos: {stats['successful']}")
    print(f"❌ Falhas: {stats['failed']}")
    print(f"⚠️ Puladas: {stats['skipped']}")
    
    if stats['total_analyses'] > 0:
        success_rate = (stats['successful'] / stats['total_analyses']) * 100
        print(f"📊 Taxa de sucesso: {success_rate:.1f}%")
    
    if not args.dry_run and stats['successful'] > 0:
        print("\n🎉 Análises concluídas!")
        print("📁 Resultados salvos em diretórios: test_results_*")
        print("💡 Verifique os arquivos pareto_statistics.txt e gráficos gerados")
    elif args.dry_run:
        print("\n💡 Execute sem --dry-run para executar de fato as análises")


if __name__ == "__main__":
    main()
