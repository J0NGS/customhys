#!/usr/bin/env python3
"""
Análise Comparativa dos Métodos de Inicialização Populacional
Extrai métricas de todas as execuções e gera estatísticas agregadas
"""

import os
import pandas as pd
import numpy as np
import glob
from collections import defaultdict
import json
import argparse
from scipy import interpolate

def load_efficient_frontier(filepath):
    """
    Carrega a fronteira eficiente de um arquivo txt
    Formato esperado: retorno risco (separados por espaços)
    """
    try:
        data = np.loadtxt(filepath, dtype=float)
        if data.shape[1] != 2:
            raise ValueError(f"Arquivo deve ter 2 colunas (retorno, risco), encontrado {data.shape[1]}")
        
        returns = data[:, 0]
        risks = data[:, 1]
        
        # Ordena por risco para interpolação
        sorted_indices = np.argsort(risks)
        returns_sorted = returns[sorted_indices]
        risks_sorted = risks[sorted_indices]
        
        return returns_sorted, risks_sorted
    except Exception as e:
        print(f"❌ Erro ao carregar fronteira eficiente de {filepath}: {e}")
        return None, None

def calculate_interpolation_error(portfolio_data, efficient_returns, efficient_risks):
    """
    Calcula o erro de interpolação entre pontos do portfólio e a fronteira eficiente
    Retorna o menor erro entre interpolação nos dois eixos
    """
    if efficient_returns is None or efficient_risks is None:
        return None
    
    try:
        # Remove pontos duplicados e ordena para interpolação
        unique_indices = np.unique(efficient_risks, return_index=True)[1]
        eff_risks_unique = efficient_risks[unique_indices]
        eff_returns_unique = efficient_returns[unique_indices]
        
        # Interpola retorno a partir do risco
        f_ret_from_risk = interpolate.interp1d(
            eff_risks_unique, eff_returns_unique, 
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        # Interpola risco a partir do retorno
        f_risk_from_ret = interpolate.interp1d(
            eff_returns_unique, eff_risks_unique,
            kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        
        errors = []
        
        for _, row in portfolio_data.iterrows():
            portfolio_return = row['expected_return']
            portfolio_risk = row['risk']
            
            # Interpolação 1: Encontra retorno esperado para o risco do portfólio
            interpolated_return = f_ret_from_risk(portfolio_risk)
            error_return = abs(portfolio_return - interpolated_return)
            
            # Interpolação 2: Encontra risco esperado para o retorno do portfólio
            interpolated_risk = f_risk_from_ret(portfolio_return)
            error_risk = abs(portfolio_risk - interpolated_risk)
            
            # Usa o menor erro entre as duas interpolações
            min_error = min(error_return, error_risk)
            errors.append(min_error)
        
        return np.array(errors)
    
    except Exception as e:
        print(f"❌ Erro no cálculo de interpolação: {e}")
        return None

def calculate_interpolation_metrics(execution_folder, efficient_returns, efficient_risks):
    """
    Calcula métricas de interpolação para uma execução específica
    """
    metrics = {}
    
    # Arquivo da fronteira de Pareto
    pareto_file = os.path.join(execution_folder, 'population_pareto_frontier.csv')
    if os.path.exists(pareto_file):
        try:
            pareto_data = pd.read_csv(pareto_file)
            pareto_errors = calculate_interpolation_error(pareto_data, efficient_returns, efficient_risks)
            if pareto_errors is not None:
                metrics['pareto_interp_mean'] = np.mean(pareto_errors)
                metrics['pareto_interp_std'] = np.std(pareto_errors)
                metrics['pareto_interp_min'] = np.min(pareto_errors)
                metrics['pareto_interp_max'] = np.max(pareto_errors)
        except Exception as e:
            print(f"⚠️  Erro ao processar fronteira de Pareto em {execution_folder}: {e}")
    
    # Arquivo do best Pareto  
    best_pareto_file = os.path.join(execution_folder, 'population_best_pareto.csv')
    if os.path.exists(best_pareto_file):
        try:
            best_data = pd.read_csv(best_pareto_file)
            best_errors = calculate_interpolation_error(best_data, efficient_returns, efficient_risks)
            if best_errors is not None:
                metrics['best_pareto_interp_mean'] = np.mean(best_errors)
                metrics['best_pareto_interp_std'] = np.std(best_errors)
                metrics['best_pareto_interp_min'] = np.min(best_errors)
                metrics['best_pareto_interp_max'] = np.max(best_errors)
        except Exception as e:
            print(f"⚠️  Erro ao processar best Pareto em {execution_folder}: {e}")
    
    return metrics

def find_execution_folders(base_dir):
    """
    Encontra todas as pastas de execução seguindo o padrão: data_hora_instancia_esquema
    
    Args:
        base_dir: Diretório base para buscar
    
    Returns:
        Dict com estrutura: {instancia: {esquema: [lista_de_pastas]}}
    """
    execution_folders = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(base_dir):
        print(f"⚠️ Diretório não encontrado: {base_dir}")
        return execution_folders
        
    print(f"🔍 Buscando execuções em: {base_dir}")
    
    # Buscar todas as pastas que seguem o padrão
    pattern = os.path.join(base_dir, "*_*_port*_*")
    folders = glob.glob(pattern)
    
    for folder in folders:
        folder_name = os.path.basename(folder)
        parts = folder_name.split('_')
        
        if len(parts) >= 4:
            # Formato: data_hora_instancia_esquema
            # Não precisamos usar data e hora, só extrair instancia e esquema
            instancia = parts[2]
            esquema = '_'.join(parts[3:])  # Caso o esquema tenha underscores
            
            execution_folders[instancia][esquema].append(folder)
    
    print(f"   - Encontradas {len(folders)} execuções")
    
    return execution_folders

def load_metrics_from_folder(folder_path):
    """
    Carrega métricas de analysis_metrics.csv de uma pasta de execução
    
    Args:
        folder_path: Caminho para a pasta da execução
    
    Returns:
        Dict com as métricas ou None se erro
    """
    metrics_file = os.path.join(folder_path, "analysis_metrics.csv")
    
    if not os.path.exists(metrics_file):
        print(f"⚠️ Arquivo não encontrado: {metrics_file}")
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        if len(df) == 0:
            print(f"⚠️ Arquivo vazio: {metrics_file}")
            return None
        
        # Pegar a primeira (e única) linha
        metrics = df.iloc[0].to_dict()
        
        # Remover métricas não desejadas
        metrics_to_remove = ['processing_mode', 'n_processes_used', 'igd_plus']
        for metric in metrics_to_remove:
            metrics.pop(metric, None)
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro ao ler {metrics_file}: {e}")
        return None

def collect_all_metrics(execution_folders, efficient_frontier_file=None):
    """
    Coleta todas as métricas de todas as execuções
    
    Args:
        execution_folders: Dict com estrutura {instancia: {esquema: [pastas]}}
        efficient_frontier_file: Caminho para o arquivo da fronteira eficiente
    
    Returns:
        Dict com estrutura {instancia: {esquema: [lista_de_metricas]}}
    """
    all_metrics = defaultdict(lambda: defaultdict(list))
    
    # Carrega fronteira eficiente se fornecida
    efficient_returns, efficient_risks = None, None
    if efficient_frontier_file and os.path.exists(efficient_frontier_file):
        efficient_returns, efficient_risks = load_efficient_frontier(efficient_frontier_file)
        if efficient_returns is not None:
            print(f"✅ Fronteira eficiente carregada: {len(efficient_returns)} pontos")
        else:
            print(f"❌ Falha ao carregar fronteira eficiente de {efficient_frontier_file}")
    
    total_executions = 0
    successful_loads = 0
    
    for instancia, esquemas in execution_folders.items():
        print(f"\n📊 Processando instância: {instancia}")
        
        for esquema, folders in esquemas.items():
            print(f"   🔹 Esquema: {esquema} ({len(folders)} execuções)")
            
            for folder in folders:
                total_executions += 1
                metrics = load_metrics_from_folder(folder)
                
                if metrics:
                    # Adiciona métricas de interpolação se fronteira eficiente disponível
                    if efficient_returns is not None:
                        interp_metrics = calculate_interpolation_metrics(folder, efficient_returns, efficient_risks)
                        metrics.update(interp_metrics)
                    
                    all_metrics[instancia][esquema].append(metrics)
                    successful_loads += 1
                    print(f"      ✅ {os.path.basename(folder)}")
                else:
                    print(f"      ❌ {os.path.basename(folder)}")
    
    print(f"\n📈 Resumo da coleta:")
    print(f"   - Total de execuções encontradas: {total_executions}")
    print(f"   - Métricas carregadas com sucesso: {successful_loads}")
    print(f"   - Taxa de sucesso: {successful_loads/total_executions*100:.1f}%")
    
    return all_metrics

def calculate_statistics(metrics_list):
    """
    Calcula estatísticas para uma lista de métricas de múltiplas execuções
    
    Args:
        metrics_list: Lista de dicionários com métricas
    
    Returns:
        Dict com estatísticas agregadas
    """
    if not metrics_list:
        return {}
    
    # Converter para DataFrame para facilitar cálculos
    df = pd.DataFrame(metrics_list)
    
    stats = {}
    
    # Métricas de qualidade (lower is better para errors)
    quality_metrics = {
        'avg_interpolation_error': {'mean', 'std', 'min'},
        'median_interpolation_error': {'median', 'min'},
        'hypervolume_general': {'mean', 'max', 'std'},
        'hypervolume_best_pareto': {'mean', 'max', 'std'},
        'hypervolume_best_efficient': {'mean', 'max', 'std'},
        'best_sharpe': {'mean', 'max', 'std'},
        'pareto_frontier_size': {'mean', 'max', 'std'},
        'total_evaluations': {'mean', 'std'},
        # NOVAS MÉTRICAS DE INTERPOLAÇÃO
        'pareto_interp_mean': {'mean', 'std', 'min', 'max'},
        'pareto_interp_std': {'mean', 'std'},
        'pareto_interp_min': {'mean', 'min'},
        'pareto_interp_max': {'mean', 'max'},
        'best_pareto_interp_mean': {'mean', 'std', 'min', 'max'},
        'best_pareto_interp_std': {'mean', 'std'},
        'best_pareto_interp_min': {'mean', 'min'},
        'best_pareto_interp_max': {'mean', 'max'}
    }
    
    for metric, stat_types in quality_metrics.items():
        if metric in df.columns:
            metric_stats = {}
            
            if 'mean' in stat_types:
                metric_stats['mean'] = df[metric].mean()
            if 'median' in stat_types:
                metric_stats['median'] = df[metric].median()
            if 'std' in stat_types:
                metric_stats['std'] = df[metric].std()
            if 'min' in stat_types:
                metric_stats['min'] = df[metric].min()
            if 'max' in stat_types:
                metric_stats['max'] = df[metric].max()
            
            metric_stats['count'] = len(df[metric].dropna())
            stats[metric] = metric_stats
    
    return stats

def generate_comparison_table(aggregated_stats):
    """
    Gera tabela comparativa formatada dos métodos de inicialização
    
    Args:
        aggregated_stats: Dict com {instancia: {esquema: stats}}
    
    Returns:
        DataFrame com comparação formatada
    """
    comparison_data = []
    
    for instancia, esquemas in aggregated_stats.items():
        for esquema, stats in esquemas.items():
            row = {
                'Instância': instancia,
                'Método': esquema,
                'N_Execuções': stats.get('avg_interpolation_error', {}).get('count', 0)
            }
            
            # Erro de interpolação (menor é melhor)
            if 'avg_interpolation_error' in stats:
                avg_err = stats['avg_interpolation_error']
                row['Avg_Interp_Error'] = f"{avg_err.get('mean', 0):.4f} ± {avg_err.get('std', 0):.4f}"
                row['Min_Interp_Error'] = f"{avg_err.get('min', 0):.4f}"
            
            # Erro mediano de interpolação
            if 'median_interpolation_error' in stats:
                med_err = stats['median_interpolation_error']
                row['Median_Interp_Error'] = f"{med_err.get('median', 0):.4f}"
                row['Min_Median_Error'] = f"{med_err.get('min', 0):.4f}"
            
            # Hipervolume geral (maior é melhor)
            if 'hypervolume_general' in stats:
                hv_gen = stats['hypervolume_general']
                row['HV_General'] = f"{hv_gen.get('mean', 0):.2e}"
                row['HV_General_Max'] = f"{hv_gen.get('max', 0):.2e}"
            
            # Hipervolume best pareto (maior é melhor)
            if 'hypervolume_best_pareto' in stats:
                hv_pareto = stats['hypervolume_best_pareto']
                row['HV_Best_Pareto'] = f"{hv_pareto.get('mean', 0):.2e}"
                row['HV_Best_Pareto_Max'] = f"{hv_pareto.get('max', 0):.2e}"
            
            # Hipervolume best efficient (maior é melhor)
            if 'hypervolume_best_efficient' in stats:
                hv_eff = stats['hypervolume_best_efficient']
                row['HV_Best_Efficient'] = f"{hv_eff.get('mean', 0):.2e}"
                row['HV_Best_Efficient_Max'] = f"{hv_eff.get('max', 0):.2e}"
            
            # Best Sharpe (maior é melhor, mas pode ser negativo)
            if 'best_sharpe' in stats:
                sharpe = stats['best_sharpe']
                row['Best_Sharpe'] = f"{sharpe.get('mean', 0):.4f} / {sharpe.get('max', 0):.4f}"
            
            # NOVAS MÉTRICAS DE INTERPOLAÇÃO
            # Interpolação da Fronteira de Pareto (menor é melhor)
            if 'pareto_interp_mean' in stats:
                pareto_interp = stats['pareto_interp_mean']
                row['Pareto_Interp_Error'] = f"{pareto_interp.get('mean', 0):.6f} ± {pareto_interp.get('std', 0):.6f}"
                row['Pareto_Interp_Min'] = f"{pareto_interp.get('min', 0):.6f}"
            
            # Interpolação do Best Pareto (menor é melhor)
            if 'best_pareto_interp_mean' in stats:
                best_interp = stats['best_pareto_interp_mean']
                row['Best_Pareto_Interp_Error'] = f"{best_interp.get('mean', 0):.6f} ± {best_interp.get('std', 0):.6f}"
                row['Best_Pareto_Interp_Min'] = f"{best_interp.get('min', 0):.6f}"
            
            # Tamanho da fronteira de Pareto
            if 'pareto_frontier_size' in stats:
                pf_size = stats['pareto_frontier_size']
                row['Pareto_Size'] = f"{pf_size.get('mean', 0):.1f}"
                row['Pareto_Size_Max'] = f"{pf_size.get('max', 0):.0f}"
            
            # Total de avaliações
            if 'total_evaluations' in stats:
                total_eval = stats['total_evaluations']
                row['Total_Evaluations'] = f"{total_eval.get('mean', 0):.0f}"
            
            comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def save_detailed_results(all_metrics, aggregated_stats, output_dir):
    """
    Salva resultados detalhados em múltiplos formatos
    
    Args:
        all_metrics: Dados brutos coletados
        aggregated_stats: Estatísticas agregadas
        output_dir: Diretório de saída
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Salvar dados brutos em JSON
    raw_data_file = os.path.join(output_dir, "raw_metrics_data.json")
    
    # Converter para formato serializável
    serializable_data = {}
    for instancia, esquemas in all_metrics.items():
        serializable_data[instancia] = {}
        for esquema, metrics_list in esquemas.items():
            serializable_data[instancia][esquema] = metrics_list
    
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Dados brutos salvos: {raw_data_file}")
    
    # 2. Salvar estatísticas agregadas em JSON
    stats_file = os.path.join(output_dir, "aggregated_statistics.json")
    
    # Converter numpy types para tipos nativos Python
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    serializable_stats = convert_numpy_types(aggregated_stats)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Estatísticas agregadas salvas: {stats_file}")
    
    # 3. Gerar e salvar tabela comparativa
    comparison_df = generate_comparison_table(aggregated_stats)
    
    comparison_csv = os.path.join(output_dir, "comparison_table.csv")
    comparison_df.to_csv(comparison_csv, index=False, encoding='utf-8')
    print(f"✅ Tabela comparativa salva: {comparison_csv}")
    
    return comparison_df

def print_summary_report(comparison_df):
    """
    Imprime relatório resumido no console
    
    Args:
        comparison_df: DataFrame com dados comparativos
    """
    print("\n" + "="*80)
    print("📊 RELATÓRIO COMPARATIVO - MÉTODOS DE INICIALIZAÇÃO POPULACIONAL")
    print("="*80)
    
    if comparison_df.empty:
        print("❌ Nenhum dado encontrado para gerar relatório")
        return
    
    # Agrupar por instância
    for instancia in comparison_df['Instância'].unique():
        df_inst = comparison_df[comparison_df['Instância'] == instancia]
        
        print(f"\n🎯 INSTÂNCIA: {instancia}")
        print("-" * 50)
        
        # Tabela resumida com novas métricas
        print(f"{'Método':<15} {'N_Exec':<6} {'Avg_Error':<12} {'HV_General':<12} {'Best_Sharpe':<15} {'Pareto_Interp':<12} {'Best_Interp':<12}")
        print("-" * 100)
        
        for _, row in df_inst.iterrows():
            metodo = row['Método'][:14]  # Truncar se muito longo
            n_exec = row['N_Execuções']
            avg_error = row.get('Avg_Interp_Error', 'N/A')[:11]
            hv_general = row.get('HV_General', 'N/A')[:11]
            best_sharpe = row.get('Best_Sharpe', 'N/A')[:14]
            pareto_interp = row.get('Pareto_Interp_Error', 'N/A')[:11]
            best_interp = row.get('Best_Pareto_Interp_Error', 'N/A')[:11]
            
            print(f"{metodo:<15} {n_exec:<6} {avg_error:<12} {hv_general:<12} {best_sharpe:<15} {pareto_interp:<12} {best_interp:<12}")
    
    print("\n" + "="*80)
    print("📋 LEGENDA:")
    print("  • Avg_Error: Erro médio de interpolação (menor = melhor)")
    print("  • HV_General: Hipervolume geral (maior = melhor)")
    print("  • Best_Sharpe: Média/Máximo do Sharpe ratio (maior = melhor)")
    print("  • Pareto_Interp: Erro de interpolação da fronteira de Pareto vs fronteira eficiente (menor = melhor)")
    print("  • Best_Interp: Erro de interpolação do best Pareto vs fronteira eficiente (menor = melhor)")
    print("="*80)

def main():
    """Função principal"""
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(
        description='Análise Comparativa dos Métodos de Inicialização Populacional',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python analyze_initialization_methods.py testes_config_fast_lambda_0_5
  python analyze_initialization_methods.py testes_config_default_lambda_0_5
  python analyze_initialization_methods.py port1_config_alta
        """
    )
    
    parser.add_argument(
        'pasta_analise',
        help='Nome da pasta que contém as execuções para análise'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Diretório de saída para os resultados (padrão: analysis_<pasta_analise>)'
    )
    
    parser.add_argument(
        '-f', '--frontier',
        help='Arquivo da fronteira eficiente (ex: portef1.txt). Se não fornecido, tentará detectar automaticamente'
    )
    
    args = parser.parse_args()
    
    # Configurar diretório de saída
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"analysis_{args.pasta_analise}"
    
    print("🚀 ANÁLISE COMPARATIVA DOS MÉTODOS DE INICIALIZAÇÃO POPULACIONAL")
    print("="*70)
    print(f"📁 Pasta de análise: {args.pasta_analise}")
    print(f"📁 Diretório de saída: {output_dir}")
    
    # Determinar arquivo da fronteira eficiente
    efficient_frontier_file = None
    if args.frontier:
        efficient_frontier_file = args.frontier
        if not os.path.exists(efficient_frontier_file):
            print(f"❌ Arquivo da fronteira eficiente não encontrado: {efficient_frontier_file}")
            return
    else:
        # Tentar detectar automaticamente baseado no nome da pasta ou execuções encontradas
        instance_detected = None
        
        # Primeiro, tentar pelo nome da pasta
        if 'port1' in args.pasta_analise.lower():
            instance_detected = 'port1'
        elif 'port2' in args.pasta_analise.lower():
            instance_detected = 'port2'
        elif 'port3' in args.pasta_analise.lower():
            instance_detected = 'port3'
        elif 'port4' in args.pasta_analise.lower():
            instance_detected = 'port4'
        elif 'port5' in args.pasta_analise.lower():
            instance_detected = 'port5'
        
        # Se não detectou pela pasta, buscar pela primeira execução encontrada
        if not instance_detected:
            pattern = os.path.join(args.pasta_analise, "*_*_port*_*")
            sample_folders = glob.glob(pattern)
            if sample_folders:
                folder_name = os.path.basename(sample_folders[0])
                parts = folder_name.split('_')
                if len(parts) >= 4:
                    for part in parts:
                        if part.startswith('port'):
                            instance_detected = part
                            break
        
        # Agora mapear para o arquivo correto
        if instance_detected == 'port1':
            efficient_frontier_file = 'portef1.txt'
        elif instance_detected == 'port2':
            efficient_frontier_file = 'portef2.txt'
        elif instance_detected == 'port3':
            efficient_frontier_file = 'portef3.txt'
        elif instance_detected == 'port4':
            efficient_frontier_file = 'portef4.txt'
        elif instance_detected == 'port5':
            efficient_frontier_file = 'portef5.txt'
        
        if efficient_frontier_file and os.path.exists(efficient_frontier_file):
            print(f"✅ Fronteira eficiente detectada automaticamente: {efficient_frontier_file} (instância: {instance_detected})")
        else:
            print("⚠️  Fronteira eficiente não fornecida. Análise de interpolação será ignorada.")
            efficient_frontier_file = None
    
    # 1. Encontrar todas as pastas de execução
    execution_folders = find_execution_folders(args.pasta_analise)
    
    if not execution_folders:
        print("❌ Nenhuma execução encontrada!")
        print("💡 Verifique se:")
        print(f"   - A pasta '{args.pasta_analise}' existe")
        print("   - Há subpastas com padrão: YYYYMMDD_HHMMSS_portX_esquema")
        print("   - As pastas contêm arquivos analysis_metrics.csv")
        return
    
    # 2. Coletar todas as métricas
    all_metrics = collect_all_metrics(execution_folders, efficient_frontier_file)
    
    if not all_metrics:
        print("❌ Nenhuma métrica coletada!")
        return
    
    # 3. Calcular estatísticas agregadas
    print("\n🧮 Calculando estatísticas agregadas...")
    aggregated_stats = {}
    
    for instancia, esquemas in all_metrics.items():
        aggregated_stats[instancia] = {}
        for esquema, metrics_list in esquemas.items():
            if metrics_list:  # Só processar se há dados
                aggregated_stats[instancia][esquema] = calculate_statistics(metrics_list)
                print(f"   ✅ {instancia} - {esquema}: {len(metrics_list)} execuções")
    
    # 4. Salvar resultados detalhados
    print(f"\n💾 Salvando resultados em: {output_dir}")
    
    comparison_df = save_detailed_results(all_metrics, aggregated_stats, output_dir)
    
    # 5. Imprimir relatório resumido
    print_summary_report(comparison_df)
    
    print("\n🎉 Análise concluída!")
    print(f"📁 Resultados salvos em: {os.path.abspath(output_dir)}")
    
    # 6. Dicas adicionais
    print("\n💡 PRÓXIMOS PASSOS:")
    print(f"   - Verifique a tabela: {output_dir}/comparison_table.csv")
    print(f"   - Dados detalhados: {output_dir}/aggregated_statistics.json")
    print(f"   - Dados brutos: {output_dir}/raw_metrics_data.json")

if __name__ == "__main__":
    main()
