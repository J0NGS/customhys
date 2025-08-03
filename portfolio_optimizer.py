#!/usr/bin/env python3
"""
Portfolio Optimization usando CustomHyS
"""

import os
import sys
import argparse
import pandas as pd
from functools import partial

from customhys.hyperheuristic import Hyperheuristic
from portfolio_utils.instance_reader import read_or_library_instance
from portfolio_utils.portfolio_logger import PortfolioLogger 
from portfolio_utils.portfolio_evaluator import portfolio_evaluation, configure_problem
from portfolio_utils.config_utils import load_hh_config, create_output_directory, save_logs
from portfolio_utils.portfolio_analyzer import analyze_portfolio_results

def main():
    parser = argparse.ArgumentParser(description='Portfolio Optimization usando CustomHyS')
    parser.add_argument('instance_file', help='Arquivo da instância (ex: port1.txt)')
    parser.add_argument('cardinality', type=int, help='Restrição de cardinalidade (0 = sem restrição)')
    parser.add_argument('config_file', help='Arquivo de configuração da hyper-heurística')
    parser.add_argument('frontier_file', nargs='?', default=None, 
                        help='Arquivo da fronteira eficiente (opcional)')
    parser.add_argument('lambda_param', nargs='?', type=float, default=0.5,
                        help='Parâmetro lambda para trade-off risco-retorno (padrão: 0.5)')
    args = parser.parse_args()

    # Carregamento de dados e configurações
    instance_data = read_or_library_instance(args.instance_file)
    k = None if args.cardinality == 0 else args.cardinality
    hh_config = load_hh_config(args.config_file)
    lambda_param = args.lambda_param
    
    print(f"📊 Parâmetro lambda configurado: {lambda_param}")
    
    # Validar arquivo da fronteira eficiente se fornecido
    if args.frontier_file and not os.path.exists(args.frontier_file):
        print(f"⚠️ Arquivo da fronteira eficiente não encontrado: {args.frontier_file}")
        print("   A análise será executada sem a fronteira eficiente.")
        args.frontier_file = None
    elif args.frontier_file:
        print(f"✅ Arquivo da fronteira eficiente encontrado: {args.frontier_file}")
    else:
        print("ℹ️ Nenhum arquivo de fronteira eficiente fornecido.")
    
    # Configuração do Logger
    output_dir = create_output_directory(args.instance_file, hh_config.get('initial_scheme', 'unknown'))
    print(f"📁 Diretório de saída: {output_dir}")
    
    log_file_path = os.path.join(output_dir, "execution_logs.csv")
    logger = PortfolioLogger(log_file_path=log_file_path, buffer_size=100000)

    # Configuração do problema para a HH
    print("\n⚙️  Configurando problema...")
    problem_config = configure_problem(instance_data, k=k, risk_free_rate=0.03, lambda_=lambda_param)

    evaluation_func_with_logger = partial(
        portfolio_evaluation, 
        instance_data=instance_data, 
        k=k, 
        risk_free_rate=0.03,
        logger=logger,
        lambda_=lambda_param,
    )
    
    problem_config["function"] = lambda weights: evaluation_func_with_logger(weights)[0]
    print("✅ Problema configurado!")

    # --- EXECUÇÃO SIMPLIFICADA ---
    print(f"\n🚀 Iniciando execução da Hyper-Heurística...")
    result_from_hh = None
    try:
        # Usar o mesmo nome do output_dir como file_label para a HH
        hh = Hyperheuristic(
            heuristic_space='default_portfolio.txt', 
            problem=problem_config, 
            parameters=hh_config,
            file_label=output_dir  # Mesmo identificador do diretório de saída
        )
        # Apenas executamos. Vamos ignorar o que hh.solve() retorna.
        result_from_hh = hh.solve()
        print(f"✅ Execução concluída!")
        print(f"📁 Data files da HH salvos em: data_files/raw/{output_dir}/")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        result_from_hh = {"error": str(e)}
        
    finally:
        print("\n💾 Salvando logs restantes...")
        logger.close()
        print("✅ Logs restantes salvos.")

    # --- ANÁLISE E SALVAMENTO A PARTIR DO ARQUIVO DE LOG ---
    print(f"\n💾 Analisando e salvando resultados...")
    try:
        # 1. Ler o arquivo CSV que foi gerado
        all_logs_from_file = []
        if os.path.exists(log_file_path):
            df = pd.read_csv(log_file_path)
            # Converte o DataFrame de volta para uma lista de dicionários, igual ao 'execution_logs' original
            all_logs_from_file = df.to_dict('records')

        # 2. Aplicar a sua lógica original para encontrar a melhor solução
        if all_logs_from_file:
            print(f"   - Total de avaliações: {len(all_logs_from_file)}")
            best_solution = min(all_logs_from_file, key=lambda x: x.get('objective', float('inf')))
            print(f"   - Melhor objetivo: {best_solution.get('objective', 'N/A'):.6f}")
            print(f"   - Melhor Sharpe: {best_solution.get('sharpe', 'N/A'):.4f}")
            print(f"   - Retorno: {best_solution.get('expected_return', 'N/A'):.4f}")
            print(f"   - Risco: {best_solution.get('risk', 'N/A'):.4f}")
            # O 'selected_assets' no CSV pode ser lido como uma string, precisamos converter
            selected_assets_str = best_solution.get('selected_assets', '[]')
            print(f"   - Ativos selecionados: {len(eval(selected_assets_str))}")
        else:
            print("   - Nenhum log de execução encontrado para analisar.")
        
        # 3. Chamar a função save_logs com os dados corretos
        # Passamos a lista de logs lida do arquivo para gerar as estatísticas
        save_logs(output_dir, all_logs_from_file, instance_data, hh_config, result_from_hh)
        
        print(f"✅ Resultados salvos em: {output_dir}")
        print(f"   - execution_logs.csv: Log de todas as avaliações")
        print(f"   - hh_config.json: Configuração da hyper-heurística")
        print(f"   - instance_data.json: Dados da instância")
        print(f"   - hh_result.json: Resultado final da HH (bruto)")
        print(f"   - summary_stats.json: Estatísticas resumidas")

    except Exception as e:
        print(f"❌ Erro ao salvar ou analisar resultados: {e}")
    
    # --- ANÁLISE AVANÇADA DOS RESULTADOS ---
    try:
        print("\n📊 Iniciando análise avançada dos resultados...")
        analyze_portfolio_results(output_dir, args.frontier_file)
    except Exception as e:
        print(f"❌ Erro durante análise avançada: {e}")
        
    print(f"\n🎉 Execução finalizada!")

if __name__ == "__main__":
    main()