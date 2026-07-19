#!/usr/bin/env python3
"""
Portfolio Optimization usando CustomHyS

Salva todas as avaliações em formato Parquet otimizado para melhor compressão.
"""

import os
import sys
import argparse
import time
import pandas as pd
import shutil

from customhys.hyperheuristic import Hyperheuristic
from portfolio_utils.instance_reader import read_or_library_instance
from portfolio_utils.parquet_handler import ParquetBufferWriter
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
    parser.add_argument('--no-logger', action='store_true', default=False,
                        help='Desabilita a criação do logger')
    parser.add_argument('--no-analysis', action='store_true', default=False,
                        help='Pula a análise avançada (analyze_portfolio_results)')
    parser.add_argument('--risk-free-rate', type=float, default=0.00057,
                        help='Taxa livre de risco para cálculo do índice de Sharpe (padrão: 0.00057 = 3%% anual convertido para semanal)')
    args = parser.parse_args()
    
    # --------- LEITURA E INSTANCIAMENTO DOS ARGUMENTOS PASSADOS ---------
    instance_data = read_or_library_instance(args.instance_file)
    k = None if args.cardinality == 0 else args.cardinality
    hh_config = load_hh_config(args.config_file)
    lambda_param = args.lambda_param
    
    print("------------------- Parâmetros Configurados -----------------------")
    print("[INFO] Parâmetro configurados:")
    print("- Instância: {}".format(args.instance_file))
    print("- Cardinalidade: {}".format("Sem restrição" if k is None else k))
    print("- Configuração HH: {}".format(args.config_file))
    print("- Lambda: {:.2f}".format(lambda_param))
    time.sleep(1.5)
    print("-------------------------------------------------------------------")

    # --------- VALIDAÇÕES INICIAIS DOS ARQUIVOS ---------
    if args.frontier_file and not os.path.exists(args.frontier_file):
        print(f"[WARNING] Arquivo da fronteira eficiente não encontrado: {args.frontier_file}")
        args.frontier_file = None
    elif args.frontier_file:
        print(f"[INFO] Arquivo da fronteira eficiente encontrado: {args.frontier_file}")
    else:
        print("[INFO] Nenhum arquivo de fronteira eficiente fornecido.")
    
    # --------- Configuração do Logger em Parquet ---------
    output_dir = create_output_directory(args.instance_file, hh_config.get('initial_scheme', 'unknown'))
    print(f"[INFO] Diretório de saída: {output_dir}")
    
    log_file_path = os.path.join(output_dir, "execution_logs.parquet")
    if args.no_logger:
        logger = None
        print("[INFO] Logger desabilitado por parâmetro CLI (--no-logger).")
    else:
        # ⭐ OTIMIZAÇÃO NUMPY: Passar n_assets para pré-alocar matriz 2D para weights
        n_assets = instance_data.get("n_assets", None)
        logger = ParquetBufferWriter(
            file_path=log_file_path, 
            buffer_size=5000, 
            n_assets=n_assets,  # ⭐ Ativa matriz 2D pré-alocada
            compression='snappy'
        )
        print(f"[INFO] Logger Parquet configurado: {log_file_path} (n_assets={n_assets})")
        print("[INFO] Buffer size: 5000 registros | Compressão: snappy")

    # --------- Configuração do problema com Stateful Evaluator ---------
    print("\n[PROCESS] Configurando problema com Stateful Evaluator...")
    problem_config = configure_problem(
        instance_data, 
        k=k, 
        risk_free_rate=0.03, 
        lambda_=lambda_param,
        logger=logger
    )
    
    evaluator = problem_config.get("evaluator")
    print("[INFO] Problema configurado com Stateful Evaluator!")

    # --- EXECUÇÃO ---
    print(f"\n[PROCESS] Iniciando execução da Hyper-Heurística...")
    result_from_hh = None
    try:
        hh = Hyperheuristic(
            heuristic_space='default_portfolio.txt', 
            problem=problem_config, 
            parameters=hh_config,
            file_label=output_dir
        )
        result_from_hh = hh.solve()
        print(f"[INFO] Execução concluída!")
        
        # Copiar data_files da HH para output
        raw_dd = os.path.join('data_files', 'raw', str(output_dir))
        dest_raw = os.path.join(output_dir, 'data_files_raw')
        try:
            if os.path.exists(raw_dd):
                shutil.copytree(raw_dd, dest_raw, dirs_exist_ok=True)
                print(f"[INFO] Data files da HH copiados para: {dest_raw}")
                # Converter JSONs em data_files_raw para Parquet
                _convert_data_files_to_parquet(dest_raw)
            else:
                print(f"[INFO] Nenhum data_files/raw/{output_dir} encontrado para copiar.")
        except Exception as e:
            print(f"[WARNING] Falha ao copiar/converter data_files: {e}")
    except Exception as e:
        print(f"[ERROR] Erro durante execução: {e}")
        result_from_hh = {"error": str(e)}
    finally:
        print("\n[PROCESS] Finalizando logs...")
        if evaluator is not None:
            evaluator.finalize()
            stats = evaluator.get_stats()
            print(f"[STATS] Total de avaliações: {stats['total_evaluations']}")
            print(f"[STATS] Melhor objetivo: {stats['best_objective']:.6e}")
            if stats['log_file_size_mb']:
                print(f"[STATS] Tamanho do arquivo de log: {stats['log_file_size_mb']:.2f} MB")
        
        if logger is not None:
            logger.close()
            print("[INFO] Logger Parquet finalizado.")

    # --- ANALISE E SALVAMENTO A PARTIR DO ARQUIVO DE LOG ---
    print(f"\n[PROCESS] Analisando e salvando resultados...")
    try:
        all_logs_from_file = []
        if os.path.exists(log_file_path):
            df = pd.read_parquet(log_file_path)
            all_logs_from_file = df.to_dict('records')
            
            # ⭐ OTIMIZAÇÃO: Não precisa de eval() - arrays já são nativos!
            # PyArrow carrega automaticamente como arrays/listas

        if all_logs_from_file:
            print(f"   - Total de avaliações: {len(all_logs_from_file)}")
            best_solution = min(all_logs_from_file, key=lambda x: x.get('objective', float('inf')))
            print(f"   - Melhor objetivo: {best_solution.get('objective', 'N/A'):.6f}")
            print(f"   - Retorno: {best_solution.get('expected_return', 'N/A'):.4f}")
            print(f"   - Risco: {best_solution.get('risk', 'N/A'):.4f}")
            selected_assets = best_solution.get('selected_assets', [])
            print(f"   - Ativos selecionados: {len(selected_assets)}")
        else:
            print("   - Nenhum log de execução encontrado para analisar.")
        
        save_logs(output_dir, all_logs_from_file, instance_data, hh_config, result_from_hh)
        print(f"   Resultados salvos em: {output_dir}")
        print(f"   - execution_logs.parquet: Log de todas as avaliações (otimizado)")
        print(f"   - hh_config.json: Configuração da hyper-heurística")
        print(f"   - instance_data.json: Dados da instância")
        print(f"   - hh_result.json: Resultado final da HH")
        print(f"   - summary_stats.json: Estatísticas resumidas")

    except Exception as e:
        print(f"[ERROR] Erro ao salvar ou analisar resultados: {e}")
    
    # --- ANALISE AVANÇADA DOS RESULTADOS ---
    try:
        if args.no_analysis:
            print("\n[INFO] Análise avançada desabilitada por parâmetro CLI (--no-analysis).")
        else:
            print("\n[PROCESS] Iniciando análise avançada dos resultados...")
            analyze_portfolio_results(output_dir, args.frontier_file, risk_free_rate=args.risk_free_rate)
    except Exception as e:
        print(f"[ERROR] Erro durante análise avançada: {e}")
        
    print(f"\n[INFO] Execução finalizada!")


def _convert_data_files_to_parquet(directory):
    """Converte JSON files em data_files_raw para Parquet."""
    from portfolio_utils.parquet_handler import JSONToParquetConverter
    import glob
    
    json_files = glob.glob(os.path.join(directory, "*.json"))
    if not json_files:
        return
    
    print(f"[PARQUET] Convertendo {len(json_files)} JSON files em {directory}...")
    for json_file in json_files:
        try:
            JSONToParquetConverter.json_to_parquet(json_file, compression='snappy')
            os.remove(json_file)  # Remove JSON após conversão
        except Exception as e:
            print(f"[WARNING] Falha ao converter {json_file}: {e}")


if __name__ == "__main__":
    main()