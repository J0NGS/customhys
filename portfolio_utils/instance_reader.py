import os
import numpy as np
import pandas as pd

def _read_returns_and_stddevs(lines, n_assets):
    returns = []
    std_devs = []
    for i in range(1, n_assets + 1):
        mu, sigma = map(float, lines[i].strip().split())
        returns.append(mu)
        std_devs.append(sigma)
    return returns, std_devs

def _build_corr_matrix(lines, n_assets):
    corr_matrix = np.eye(n_assets)
    for line in lines[n_assets + 1:]:
        if line.strip():
            i, j, rho = line.strip().split()
            i, j = int(i) - 1, int(j) - 1
            rho = float(rho)
            corr_matrix[i][j] = rho
            corr_matrix[j][i] = rho
    return corr_matrix

def _build_cov_matrix_and_df(std_devs, corr_matrix, n_assets):
    sigma_array = np.array(std_devs)
    cov_matrix = np.outer(sigma_array, sigma_array) * corr_matrix
    df_cov = pd.DataFrame(
        cov_matrix,
        columns=[f"Asset {i+1}" for i in range(n_assets)],
        index=[f"Asset {i+1}" for i in range(n_assets)]
    )
    return cov_matrix, df_cov

def read_or_library_instance(filepath):
    """
    Lê um arquivo de instância da OR-Library (ex: port1.txt) e retorna:
    - número de ativos
    - retornos esperados (μ_i)
    - desvios padrão (σ_i)
    - matriz de correlação (ρ_{i,j})
    - matriz de covariância (Σ)
    - dataframe da covariância para visualização
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo {filepath} não encontrado")
    with open(filepath, 'r') as file:
        lines = file.readlines()
    n_assets = int(lines[0].strip())
    returns, std_devs = _read_returns_and_stddevs(lines, n_assets)
    corr_matrix = _build_corr_matrix(lines, n_assets)
    cov_matrix, df_cov = _build_cov_matrix_and_df(std_devs, corr_matrix, n_assets)
    return {
        "n_assets": n_assets,
        "returns": np.array(returns),
        "std_devs": np.array(std_devs),
        "corr_matrix": corr_matrix,
        "cov_matrix": cov_matrix,
        "df_cov": df_cov
    }
