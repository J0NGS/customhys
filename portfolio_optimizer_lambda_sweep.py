#!/usr/bin/env python3
"""
Portfolio Optimization Lambda Sweep usando CustomHyS
Executa múltiplas otimizações com diferentes valores de lambda
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from functools import partial
import time
from datetime import datetime

from customhys.hyperheuristic import Hyperheuristic
from portfolio_utils.instance_reader import read_or_library_instance
from portfolio_utils.portfolio_logger import PortfolioLogger 
from portfolio_utils.portfolio_evaluator import portfolio_evaluation, configure_problem
from portfolio_utils.config_utils import load_hh_config, create_output_directory, save_logs
from portfolio_utils.portfolio_analyzer import analyze_portfolio_results

def main():
    parser = argparse.ArgumentParser(description='Portfolio Optimization Lambda Sweep usando CustomHyS')
    parser.add_argument('instance_file', help='Arquivo da instância (ex: port1.txt)')
    parser.add_argument('cardinality', type=int, help='Restrição de cardinalidade (0 = sem restrição)')
    parser.add_argument('config_file', help='Arquivo de configuração da hyper-heurística')
    parser.add_argument('frontier_file', nargs='?', default=None, 
                        help='Arquivo da fronteira eficiente (opcional)')
    parser.add_argument('lambda_intervals', type=int, 
                        help='Número de intervalos de lambda entre 0 e 1 (ex: 50)')
    args = parser.parse_args()

    # Carregamento de dados e configurações
    instance_data = read_or_library_instance(args.instance_file)
    k = None if args.cardinality == 0 else args.cardinality
    hh_config = load_hh_config(args.config_file)
    
    # Gerar valores de lambda
    lambda_values = np.linspace(0.01, 0.99, args.lambda_intervals)  # Evitar 0 e 1 exatos
    
    print(f"🔄 Lambda Sweep configurado:")
    print(f"   - Número de intervalos: {args.lambda_intervals}")
    print(f"   - Valores de lambda: {lambda_values[0]:.3f} até {lambda_values[-1]:.3f}")
    print(f"   - Total de execuções: {len(lambda_values)}")
    
    # Validar arquivo da fronteira eficiente se fornecido
    if args.frontier_file and not os.path.exists(args.frontier_file):
        print(f"⚠️ Arquivo da fronteira eficiente não encontrado: {args.frontier_file}")
        print("   A análise será executada sem a fronteira eficiente.")
        args.frontier_file = None
    elif args.frontier_file:
        print(f"✅ Arquivo da fronteira eficiente encontrado: {args.frontier_file}")
    else:
        print("ℹ️ Nenhum arquivo de fronteira eficiente fornecido.")
    
    # Criar diretório principal para todos os resultados
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_name = os.path.splitext(args.instance_file)[0]
    scheme_name = hh_config.get('initial_scheme', 'unknown')
    main_output_dir = f"{base_timestamp}_{instance_name}_{scheme_name}_lambda_sweep_{args.lambda_intervals}"
    
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"📁 Diretório principal: {main_output_dir}")
    
    # Lista para armazenar resultados de todas as execuções
    all_results = []
    start_time = time.time()
    
    # Executar para cada valor de lambda
    for i, lambda_param in enumerate(lambda_values, 1):
        print(f"\n{'='*60}")
        print(f"🚀 EXECUÇÃO {i}/{len(lambda_values)} - Lambda: {lambda_param:.4f}")
        print(f"{'='*60}")
        
        # Criar subdiretório para esta execução
        sub_output_dir = os.path.join(main_output_dir, f"lambda_{lambda_param:.4f}")
        os.makedirs(sub_output_dir, exist_ok=True)
        
        try:
            # Configuração do Logger
            log_file_path = os.path.join(sub_output_dir, "execution_logs.csv")
            logger = PortfolioLogger(log_file_path=log_file_path, buffer_size=100000)

            # Configuração do problema para a HH
            print("⚙️  Configurando problema...")
            problem_config = configure_problem(instance_data, k=k, risk_free_rate=0.03, lambda_=lambda_param)

            evaluation_func_with_logger = partial(
                portfolio_evaluation, 
                instance_data=instance_data, 
                k=k, 
                risk_free_rate=0.03,
                logger=logger,
                lambda_=lambda_param,
                epsilon=0.01,  # Valor padrão de epsilon
                delta=1.0
            )
            
            problem_config["function"] = lambda weights: evaluation_func_with_logger(weights)[0]
            print("✅ Problema configurado!")

            # Execução da Hyper-Heurística
            print("🚀 Iniciando execução da Hyper-Heurística...")
            execution_start = time.time()
            
            hh = Hyperheuristic(
                heuristic_space='default_portfolio.txt', 
                problem=problem_config, 
                parameters=hh_config,
                file_label=f"lambda_{lambda_param:.4f}"
            )
            
            result_from_hh = hh.solve()
            execution_time = time.time() - execution_start
            
            print(f"✅ Execução concluída em {execution_time:.1f}s!")
            
        except Exception as e:
            print(f"❌ Erro durante execução: {e}")
            result_from_hh = {"error": str(e)}
            execution_time = 0
            
        finally:
            print("💾 Salvando logs...")
            logger.close()
            print("✅ Logs salvos.")

        # Análise e salvamento dos resultados
        try:
            # Ler logs gerados
            all_logs_from_file = []
            if os.path.exists(log_file_path):
                df = pd.read_csv(log_file_path)
                all_logs_from_file = df.to_dict('records')

            # Encontrar melhor solução
            best_solution = None
            if all_logs_from_file:
                best_solution = min(all_logs_from_file, key=lambda x: x.get('objective', float('inf')))
                
                print(f"📊 Resultados:")
                print(f"   - Total de avaliações: {len(all_logs_from_file)}")
                print(f"   - Melhor objetivo: {best_solution.get('objective', 'N/A'):.6f}")
                print(f"   - Melhor Sharpe: {best_solution.get('sharpe', 'N/A'):.4f}")
                print(f"   - Retorno: {best_solution.get('expected_return', 'N/A'):.4f}")
                print(f"   - Risco: {best_solution.get('risk', 'N/A'):.4f}")
                
                # Armazenar resultado para análise posterior
                result_summary = {
                    'lambda': lambda_param,
                    'execution_order': i,
                    'execution_time': execution_time,
                    'total_evaluations': len(all_logs_from_file),
                    'best_objective': best_solution.get('objective', None),
                    'best_sharpe': best_solution.get('sharpe', None),
                    'best_return': best_solution.get('expected_return', None),
                    'best_risk': best_solution.get('risk', None),
                    'output_dir': sub_output_dir
                }
                all_results.append(result_summary)
            
            # Salvar logs individuais
            save_logs(sub_output_dir, all_logs_from_file, instance_data, hh_config, result_from_hh)
            
            # Análise avançada (opcional, pode ser demorada)
            if args.frontier_file and len(all_logs_from_file) > 10:  # Só se houver dados suficientes
                try:
                    print("📊 Iniciando análise avançada...")
                    analyze_portfolio_results(sub_output_dir, args.frontier_file)
                    print("✅ Análise avançada concluída!")
                except Exception as e:
                    print(f"⚠️ Erro na análise avançada: {e}")

        except Exception as e:
            print(f"❌ Erro ao processar resultados: {e}")
            # Adicionar resultado com erro
            result_summary = {
                'lambda': lambda_param,
                'execution_order': i,
                'execution_time': execution_time,
                'total_evaluations': 0,
                'best_objective': None,
                'best_sharpe': None,
                'best_return': None,
                'best_risk': None,
                'output_dir': sub_output_dir,
                'error': str(e)
            }
            all_results.append(result_summary)
        
        # Progresso
        elapsed_time = time.time() - start_time
        estimated_total = (elapsed_time / i) * len(lambda_values)
        remaining_time = estimated_total - elapsed_time
        
        print(f"⏱️  Progresso: {i}/{len(lambda_values)} ({100*i/len(lambda_values):.1f}%)")
        print(f"   Tempo decorrido: {elapsed_time/60:.1f}min")
        print(f"   Tempo estimado restante: {remaining_time/60:.1f}min")

    # Análise final consolidada
    print(f"\n{'='*60}")
    print("📊 ANÁLISE CONSOLIDADA DOS RESULTADOS")
    print(f"{'='*60}")
    
    # Salvar resultados consolidados
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(main_output_dir, "lambda_sweep_results.csv")
    results_df.to_csv(results_file, index=False)
    
    # Estatísticas gerais
    valid_results = results_df[results_df['best_objective'].notna()]
    
    if len(valid_results) > 0:
        print(f"✅ Execuções bem-sucedidas: {len(valid_results)}/{len(lambda_values)}")
        print(f"📈 Estatísticas dos objetivos:")
        print(f"   - Melhor: {valid_results['best_objective'].min():.6f}")
        print(f"   - Pior: {valid_results['best_objective'].max():.6f}")
        print(f"   - Média: {valid_results['best_objective'].mean():.6f}")
        print(f"   - Desvio padrão: {valid_results['best_objective'].std():.6f}")
        
        # Melhor lambda encontrado
        best_row = valid_results.loc[valid_results['best_objective'].idxmin()]
        print(f"🏆 Melhor resultado:")
        print(f"   - Lambda: {best_row['lambda']:.4f}")
        print(f"   - Objetivo: {best_row['best_objective']:.6f}")
        print(f"   - Sharpe: {best_row['best_sharpe']:.4f}")
        print(f"   - Retorno: {best_row['best_return']:.4f}")
        print(f"   - Risco: {best_row['best_risk']:.4f}")
    else:
        print("❌ Nenhuma execução foi bem-sucedida!")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Tempo total de execução: {total_time/60:.1f}min")
    print(f"📁 Todos os resultados salvos em: {main_output_dir}")
    print(f"📊 Resultados consolidados em: {results_file}")
    
    print("\n🎉 Lambda Sweep finalizado!")

if __name__ == "__main__":
    main()
