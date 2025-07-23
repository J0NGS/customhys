#!/usr/bin/env python3
"""
Gera gráficos informativos a partir do execution_logs.csv de uma pasta de resultados do CustomHyS.

Uso:
    python plot_portfolio_results.py <nome_da_pasta>

Exemplo:
    python plot_portfolio_results.py 20250723_150351_port1_random
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Uso: python plot_portfolio_results.py <nome_da_pasta> <arquivo_fronteira>")
        sys.exit(1)
    pasta = sys.argv[1]
    arquivo_fronteira = sys.argv[2]
    caminho_csv = os.path.join(pasta, "execution_logs.csv")
    if not os.path.exists(caminho_csv):
        print(f"Arquivo execution_logs.csv não encontrado em {pasta}")
        sys.exit(1)
    if not os.path.exists(arquivo_fronteira):
        print(f"Arquivo da fronteira eficiente não encontrado: {arquivo_fronteira}")
        sys.exit(1)
    df_logs = pd.read_csv(caminho_csv)
    df_portef = pd.read_csv(arquivo_fronteira, sep="\s+", header=None, names=["mean_return", "variance"])
    df_portef["std_dev"] = np.sqrt(df_portef["variance"])
    print(f"Dados carregados: {len(df_logs)} avaliações, {len(df_portef)} pontos da fronteira eficiente")

    # Interpolação
    from scipy.interpolate import interp1d
    interp_return = interp1d(df_portef["std_dev"], df_portef["mean_return"], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_risk = interp1d(df_portef["mean_return"], df_portef["std_dev"], kind='linear', bounds_error=False, fill_value="extrapolate")
    df_logs["interp_ret_from_risk"] = interp_return(df_logs["risk"])
    df_logs["interp_risk_from_ret"] = interp_risk(df_logs["expected_return"])
    df_logs["error_return"] = abs(df_logs["expected_return"] - df_logs["interp_ret_from_risk"]) / df_logs["interp_ret_from_risk"]
    df_logs["error_risk"] = abs(df_logs["risk"] - df_logs["interp_risk_from_ret"]) / df_logs["interp_risk_from_ret"]
    df_logs["percent_error"] = df_logs[["error_return", "error_risk"]].min(axis=1)

    # Estatísticas dos erros
    print(f"\n📉 Erro percentual médio: {df_logs['percent_error'].mean():.4f}")
    print(f"📈 Erro percentual mediano: {df_logs['percent_error'].median():.4f}")
    print("\n🚨 Top 5 maiores erros percentuais:")
    print(df_logs.sort_values(by="percent_error", ascending=False).head(5)[[
        "expected_return", "risk", "interp_ret_from_risk", "interp_risk_from_ret", "percent_error"
    ]])

    # Seleção dos destaques
    melhor_sharpe = df_logs.loc[df_logs["sharpe"].idxmax()]
    maior_retorno = df_logs.loc[df_logs["expected_return"].idxmax()]
    menor_risco = df_logs.loc[df_logs["risk"].idxmin()]
    melhor_objetivo = df_logs.loc[df_logs["objective"].idxmin()]

    # Gráfico: Soluções vs Fronteira eficiente
    plt.figure(figsize=(10, 6))
    plt.scatter(df_portef["std_dev"], df_portef["mean_return"], s=10, color='blue', label="Fronteira Eficiente")
    plt.scatter(df_logs["risk"], df_logs["expected_return"], s=20, color='red', alpha=0.5, label="Soluções CustomHyS")
    plt.scatter(melhor_sharpe["risk"], melhor_sharpe["expected_return"], s=100, color='gold', edgecolor='black', label="Melhor Sharpe", zorder=5)
    plt.scatter(maior_retorno["risk"], maior_retorno["expected_return"], s=100, color='green', edgecolor='black', label="Maior Retorno", zorder=5)
    plt.scatter(menor_risco["risk"], menor_risco["expected_return"], s=100, color='purple', edgecolor='black', label="Menor Risco", zorder=5)
    plt.scatter(melhor_objetivo["risk"], melhor_objetivo["expected_return"], s=100, color='cyan', edgecolor='black', label="Melhor Objetivo", zorder=5)
    plt.text(melhor_sharpe["risk"], melhor_sharpe["expected_return"] + 0.0005, f"Sharpe: {melhor_sharpe['sharpe']:.2f}", ha='center', fontsize=9)
    plt.text(maior_retorno["risk"], maior_retorno["expected_return"] + 0.0005, f"Ret: {maior_retorno['expected_return']:.2f}", ha='center', fontsize=9)
    plt.text(menor_risco["risk"], menor_risco["expected_return"] + 0.0005, f"Risk: {menor_risco['risk']:.2f}", ha='center', fontsize=9)
    plt.text(melhor_objetivo["risk"], melhor_objetivo["expected_return"] + 0.0005, f"Obj: {melhor_objetivo['objective']:.2f}", ha='center', fontsize=9)
    plt.title("Fronteira Eficiente vs Soluções CustomHyS (Destaques)")
    plt.xlabel("Risco (Desvio Padrão)")
    plt.ylabel("Retorno Esperado")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, "grafico_fronteira_vs_solucoes.png"))
    plt.close()

    # Histograma da cardinalidade
    df_logs["num_selected"] = df_logs["selected_assets"].apply(lambda x: len(eval(str(x))) if isinstance(x, str) else len(x))
    plt.figure(figsize=(8, 4))
    plt.hist(df_logs["num_selected"], bins=range(1, df_logs["num_selected"].max()+2), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title("Distribuição da Cardinalidade (Número de Ativos Selecionados)")
    plt.xlabel("Número de Ativos")
    plt.ylabel("Frequência")
    plt.grid(True)
    plt.xticks(range(1, df_logs["num_selected"].max()+1))
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, "histograma_cardinalidade.png"))
    plt.close()

    print("Gráficos salvos na pasta:", pasta)
    print("- grafico_fronteira_vs_solucoes.png")
    print("- histograma_cardinalidade.png")

if __name__ == "__main__":
    main()
