# Análise do Sistema de Portfolio Analyzer

## 📋 Visão Geral

O projeto possui **2 versões distintas** do analisador de portfólio, cada uma com propósitos e características diferentes:

1. **`portfolio_analyzer.py`** - Versão principal (1101 linhas)
2. **`portfolio_analyzer_optimized.py`** - Versão otimizada com Numba (656 linhas)

Ambos seguem o mesmo padrão de pipeline, mas com diferentes estratégias de otimização.

---

## 🏗️ Arquitetura da Pipeline Principal

### Função Entry Point: `analyze_portfolio_results()`

```
output_dir + efficient_frontier_file (opcional) 
    ↓
[1] _load_and_validate_data() 
    └─→ Carrega execution_logs.csv + fronteira eficiente (se existir)
    └─→ Valida existência de arquivos
    
[2] calculate_pareto_frontier() [paralelo opcional]
    └─→ Ordena por retorno (descendente)
    └─→ Filtra pontos não dominados (menor risco = melhor)
    └─→ Output: DataFrame com fronteira de Pareto
    
[3] _calculate_metrics()
    ├─→ calculate_interpolation_errors() [se fronteira eficiente disponível]
    │   └─→ Compara pontos Pareto com fronteira ideal via interpolação
    │   └─→ Calcula erro percentual para cada ponto
    │   └─→ Output: avg_interpolation_error, median_interpolation_error
    │
    └─→ calculate_igd_plus() [se fronteira eficiente disponível]
        └─→ Calcula distância euclidiana entre Pareto e fronteira ideal
        └─→ Inverted Generational Distance Plus
        └─→ Output: igd_plus
    
[4] get_best_solutions_per_frontier_point() [paralelo opcional]
    ├─→ Para cada ponto na fronteira eficiente:
    │   └─→ Encontra N=100 soluções mais próximas (menor erro)
    │
    └─→ Para cada ponto na fronteira de Pareto:
        └─→ Encontra N=100 melhores soluções
        └─→ Output: best_efficient_solutions + best_pareto_solutions
    
[5] _calculate_hypervolumes()
    ├─→ Define ponto de referência (1% pior que piores valores)
    ├─→ hypervolume_general: hipervolume de TODAS as soluções
    ├─→ hypervolume_best_pareto: hipervolume das 100 melhores (Pareto)
    └─→ hypervolume_best_efficient: hipervolume das 100 melhores (fronteira eficiente)
    
[6] _save_analysis_results()
    ├─→ Salva métricas em: analise_metricas.csv
    ├─→ Salva pareto_frontier.csv
    ├─→ Salva best_efficient_solutions.csv
    └─→ Salva best_pareto_solutions.csv
    
[7] plot_frontiers_comparison()
    ├─→ fronteira_eficiente_vs_todas.png
    ├─→ fronteira_eficiente_vs_pareto.png
    ├─→ fronteira_eficiente_vs_melhores_interpolacao.png
    ├─→ fronteira_eficiente_vs_melhores_pareto.png
    └─→ histograma_cardinalidade.png [se houver dados de ativos]
    
[8] _display_metrics_summary()
    └─→ Exibe resumo formatado no console
```

---

## 🔧 Funções Auxiliares Principais

### Cálculo de Fronteira de Pareto

**`calculate_pareto_frontier(df_logs)`** (sequencial)
- Entrada: DataFrame com colunas `expected_return` e `risk`
- Algoritmo: Ordena por retorno DESC, depois filtra riscos monótonos
- Saída: DataFrame sorted by risk com pontos não dominados
- Complexidade: O(n log n)

**`calculate_pareto_frontier_parallel(df_logs, n_processes)`** (paralelo)
- Para n > 10000 soluções usa multiprocessing
- Divide dados em chunks, processa em paralelo, combina resultados
- Usa `_find_pareto_in_chunk()` em cada worker process
- Fallback para versão sequencial se falhar

---

### Cálculo de Erros de Interpolação

**`calculate_interpolation_errors(df_logs, efficient_frontier)`**
- Para cada solução calculada:
  1. Localiza o intervalo na fronteira eficiente onde ela se encaixa
  2. Interpola o valor ESPERADO de retorno para aquele risco
  3. Calcula erro: `percent_error = |expected - actual| / |expected|`
- Adiciona coluna `percent_error` ao DataFrame
- Usa `scipy.interpolate.interp1d` com extrapolação

**Uso:** Mede quanto as soluções encontradas se afastam da fronteira teórica ideal

---

### Cálculo de IGD+ (Inverted Generational Distance Plus)

**`calculate_igd_plus(pareto_front, reference_front)`**
- Para cada ponto na fronteira eficiente:
  1. Calcula distância euclidiana até TODOS os pontos Pareto
  2. Pega a distância MÍNIMA
  3. Media todas as distâncias mínimas
- Quanto MENOR o IGD+, melhor (significa Pareto mais perto do ideal)

**`calculate_igd_plus_parallel()`** (versão paralelo para n*m > 50000)
- Divide fronteira eficiente em chunks
- Cada worker calcula IGD+ para seu chunk
- Combina resultados com média

---

### Seleção de Melhores Soluções

**`get_best_solutions_per_frontier_point(df_logs, frontier_points, n_solutions_per_point=100)`**
- Para cada ponto na fronteira (eficiente ou Pareto):
  1. Encontra N soluções mais próximas (menor erro de interpolação)
  2. Retorna up to 100 soluções únicas
- Usa `scipy.spatial.distance.cdist` para calcular distâncias

**`get_best_solutions_per_frontier_point_parallel()`** (versão paralelo)
- Divide pontos frontier em batches
- Cada worker processa seu batch
- Usa `_process_frontier_batch_parallel()`

**Saída:** DataFrame com as melhores soluções encontradas

---

### Cálculo de Hipervolume

**`calculate_hypervolume_2d(points, reference_point)`**
- Calcula área dominada por um conjunto de soluções 2D
- Pontos precisam ser não-dominados
- Usa `_get_non_dominated_points()` + `_calculate_2d_hypervolume()`
- Ponto de referência = (retorno 1% pior que mínimo, risco 1% pior que máximo)

---

## 📊 Estrutura de Dados - Fluxo

```
INPUT: execution_logs.csv
├─ Colunas esperadas:
│  ├─ objective (ou sharpe, fitness)
│  ├─ expected_return
│  ├─ risk
│  ├─ sharpe
│  └─ selected_assets [opcional, para cardinalidade]
│
├─ Processamento P1: Load + Validate
│  └─→ df_logs (DataFrame com todas as avaliações)
│
├─ Processamento P2: Pareto Frontier
│  └─→ pareto_frontier (DataFrame com N_pareto pontos não-dominados)
│
├─ Processamento P3: Métricas
│  └─→ df_logs.percent_error (adicionado, N linhas)
│  └─→ metrics dict: avg_error, median_error, igd_plus
│
├─ Processamento P4: Melhores Soluções
│  └─→ best_efficient_solutions (DataFrame com até 100*len(frontier_ef))
│  └─→ best_pareto_solutions (DataFrame com até 100*len(pareto))
│
└─ OUTPUT: Arquivos salvos
   ├─ analise_metricas.csv (1 linha com todas as métricas)
   ├─ pareto_frontier.csv (N_pareto linhas)
   ├─ best_efficient_solutions.csv 
   ├─ best_pareto_solutions.csv
   ├─ 5 PNG files (gráficos)
   └─ metrics dict retornado para caller
```

---

## 🎯 Funções de Entrada (Wrappers)

### Em `portfolio_analyzer.py`:

1. **`analyze_portfolio_results(output_dir, efficient_frontier_file=None, use_parallel=True, n_processes=None)`**
   - Função principal PADRÃO
   - ✅ Usado por: `portfolio_optimizer.py` + `portfolio_optimizer_lambda_sweep.py`
   - Paraleliza automaticamente se n > 10000 ou se flag `use_parallel=True`

2. **`analyze_portfolio_results_fast(output_dir, efficient_frontier_file=None, n_processes=None)`**
   - Alias para `analyze_portfolio_results(..., use_parallel=True)`
   - Garante processamento paralelo
   - Para datasets grandes (>10k soluções)

3. **`analyze_portfolio_results_sequential(output_dir, efficient_frontier_file=None)`**
   - Alias para `analyze_portfolio_results(..., use_parallel=False)`
   - Versão estritamente sequencial
   - Para compatibilidade/debugging

### Em `portfolio_analyzer_optimized.py`:

4. **`analyze_portfolio_results_optimized(output_dir, efficient_frontier_file=None, ...)`**
   - Versão otimizada com Numba (JIT compilation)
   - Usa monitoramento de memória
   - Mais rápida para datasets MUITO grandes (>100k)
   - ⚠️ Não é usado atualmente (portfolio_optimizer.py usa a versão padrão)

5. **`analyze_portfolio_results_ultra_fast(output_dir, efficient_frontier_file=None)`**
   - Alias para otimized com parâmetros agressivos
   - Máxima velocidade, pode sacrificar alguns detalhes
   - Experimental

---

## 🔌 Integração com Scripts Existentes

### `portfolio_optimizer.py`

```python
# Linha 19
from portfolio_utils.portfolio_analyzer import analyze_portfolio_results

# Linha 168 (após execução da HH)
if not args.no_analysis:
    analyze_portfolio_results(output_dir, args.frontier_file)
```

**Status:** ✅ ATIVO - Chama versão principal com paralelização automática

---

### `portfolio_optimizer_lambda_sweep.py`

```python
# Linha 22
from portfolio_utils.portfolio_analyzer import analyze_portfolio_results

# Linha 209 (dentro do loop de lambda)
if not args.no_analysis:
    analyze_portfolio_results(sub_output_dir, args.frontier_file)
```

**Status:** ✅ ATIVO - Chama versão principal para CADA lambda

**⚠️ OBSERVAÇÃO:** Rodando análise N vezes (uma por lambda) pode ser lento
- Potencial melhoria: Fazer análise agregada após loop

---

### `portfolio_metaheuristic_executor.py`

```python
# NÃO IMPORTA portfolio_analyzer
# Portanto, NÃO roda análise automática
```

**Status:** ❌ NÃO INTEGRADO - Precisaria de implementação

**Proposta:** Adicionar `--analysis` flag + chamar `analyze_portfolio_results()`

---

## 📈 Características das Versões

### `portfolio_analyzer.py` (PADRÃO)

| Aspecto | Detalhe |
|---------|---------|
| **Tamanho** | 1101 linhas |
| **Estratégia** | Multiprocessing Pool |
| **Threshold Paralelo** | n > 10000 |
| **Aceleração** | Divisão em chunks |
| **Numba** | Não usa |
| **Psutil** | Não requer |
| **Status** | ✅ Produção |
| **Usado por** | portfolio_optimizer[_lambda_sweep].py |

**Vantagens:**
- Simples de entender
- Sem dependências extras
- Bem testado em produção
- Com fallback para sequencial

**Desvantagens:**
- Overhead de multiprocessing para datasets pequenos
- Sem monitoramento de memória

---

### `portfolio_analyzer_optimized.py` (OTIMIZADA)

| Aspecto | Detalhe |
|---------|---------|
| **Tamanho** | 656 linhas |
| **Estratégia** | Numba JIT + Chunking |
| **Threshold Paralelo** | Inteligente (baseado em memória) |
| **Aceleração** | JIT compilation + Broadcasting |
| **Numba** | Usa (com fallback Python) |
| **Psutil** | Usa (com fallback) |
| **Status** | 🧪 Experimental |
| **Usado por** | Ninguém atualmente |

**Vantagens:**
- Mais rápida para datasets MUITO grandes (>100k)
- Monitoramento de memória
- Compilação JIT (10-100x mais rápido em loops críticos)
- Melhor manejo de memória

**Desvantagens:**
- Mais complexa
- Dependências extras (psutil, numba)
- Menos testada
- Não integrada em scripts existentes

---

## 🚀 Fluxo Paralelo vs Sequencial

### Quando `use_parallel=True` (Automático)

```
Se n_soluções > 10000:
    ├─→ Pareto: Usa calculate_pareto_frontier_parallel()
    ├─→ IGD+: Usa calculate_igd_plus_parallel()
    └─→ Seleção: Usa get_best_solutions_per_frontier_point_parallel()
Senão:
    └─→ Usa versões sequenciais
```

**Número de processos:** `min(cpu_count(), 8)` (até 8 cores)

---

### Quando `use_parallel=False`

```
Usa SEMPRE versões sequenciais:
├─→ calculate_pareto_frontier()
├─→ calculate_igd_plus()
└─→ get_best_solutions_per_frontier_point()
```

---

## 📁 Estrutura de Saída

Após executar `analyze_portfolio_results(output_dir)`:

```
output_dir/
├─ execution_logs.csv (original, não modificado)
├─ analise_metricas.csv (📊 NOVA - métricas calculadas)
│  └─ Colunas: total_evaluations, pareto_frontier_size, 
│             avg_interpolation_error, median_interpolation_error,
│             igd_plus, hypervolume_general, hypervolume_best_pareto,
│             hypervolume_best_efficient, best_sharpe, processing_mode, etc
│
├─ pareto_frontier.csv (📊 NOVA)
│  └─ N_pareto linhas, colunas: expected_return, risk, objective, sharpe, etc
│
├─ best_efficient_solutions.csv (📊 NOVA, se fronteira fornecida)
│  └─ Até 100*N_ef linhas
│
├─ best_pareto_solutions.csv (📊 NOVA)
│  └─ Até 100*N_pareto linhas
│
├─ fronteira_eficiente_vs_todas.png
├─ fronteira_eficiente_vs_pareto.png
├─ fronteira_eficiente_vs_melhores_interpolacao.png [se fronteira fornecida]
├─ fronteira_eficiente_vs_melhores_pareto.png [se fronteira fornecida]
└─ histograma_cardinalidade.png [se dados de ativos disponíveis]
```

---

## 🎯 Decisões que Você Pode Tomar

### 1️⃣ **Integrar análise no `portfolio_metaheuristic_executor.py`?**

**Opções:**
- A) Adicionar flag `--analysis` que chama `analyze_portfolio_results()`
- B) Sempre rodar análise (como faz `portfolio_optimizer.py`)
- C) Deixar sem análise (não integrar)

**Recomendação:** ✅ Opção A (flag opcional, compatível com `portfolio_optimizer.py`)

---

### 2️⃣ **Usar versão otimizada?**

**Cenário 1 - Datasets pequenos/médios (<50k avaliações):**
- Continue usando `portfolio_analyzer.py` (versão padrão)
- Já é rápido o suficiente

**Cenário 2 - Datasets gigantes (>100k avaliações):**
- Considere usar `portfolio_analyzer_optimized.py`
- Ganha 5-10x de velocidade com Numba

---

### 3️⃣ **Paralelização em `portfolio_optimizer_lambda_sweep.py`?**

**Problema atual:** Roda análise N vezes (uma por lambda)

**Potenciais soluções:**
- A) Status quo (análise individual, mais detalhada)
- B) Análise agregada depois do loop (mais rápido)
- C) Usar `analyze_portfolio_results_fast(..., n_processes=4)` (default paraleliza)

**Recomendação:** ✅ Status quo é bom (detalhamento fino), mas C é mais rápido

---

### 4️⃣ **Adicionar novas métricas de análise?**

Opções:
- Adicionar função auxiliar em `_calculate_metrics()` 
- Seguir padrão: `if condition: metrics['new_metric'] = calculate_new_metric(...)`
- Exemplos: Coeficiente de Gini, Sortino ratio, Max drawdown, etc

---

## 📝 Resumo Executivo

| Métrica | Valor |
|---------|-------|
| **Linhas de código (analyzer.py)** | 1,101 |
| **Funções análise** | 20+ |
| **Funções gráficos** | 6+ |
| **Métricas calculadas** | 10+ |
| **Gráficos gerados** | 5 |
| **Integração atual** | 2 scripts (portfolio_optimizer.py + _lambda_sweep.py) |
| **Versão otimizada** | Experimental (não integrada) |
| **Paralelização** | Automática (>10k soluções) |
| **Tempo típico** | 2-30s (depende do dataset) |

---

## 🔗 Referências Cruzadas

```
portfolio_analyzer.py ─────→ Importado por:
   ├─ portfolio_optimizer.py (linha 19)
   ├─ portfolio_optimizer_lambda_sweep.py (linha 22)
   └─ (portfolio_metaheuristic_executor.py não importa)

portfolio_analyzer_optimized.py ─────→ Não importado por ninguém
   └─ Pode ser ativado manualmente para datasets grandes
```

---

**Última atualização:** 17 de novembro de 2025
