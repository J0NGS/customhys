# 🔍 ANÁLISE DE ERROS CRÍTICOS - CONSUMO EXCESSIVO DE RAM

## Revisão do código: `analyze_sweep_results.py` e `portfolio_analyzer.py`

**Data**: 13 de janeiro de 2026  
**Objetivo**: Identificar erros que causam consumo excessivo de RAM durante análise

---

## ⚠️ ERROS CRÍTICOS ENCONTRADOS

### **ERRO 1: Carregamento completo do Parquet em memória (CRÍTICO)**

**Arquivo**: `portfolio_utils/portfolio_analyzer.py` - Função `_load_and_validate_data()` (linha 880)

```python
df_logs = _load_dataframe_flexible(output_dir, "execution_logs")  # CARREGA TUDO NA RAM!
```

**Problema**:
- `pd.read_parquet()` carrega **100% do arquivo na memória** de uma vez
- Se `execution_logs.parquet` tem 1M+ linhas, vai consumir **gigabytes**
- Exemplo: 1M linhas × 20 colunas × 8 bytes ≈ **160 MB por coluna** = **3.2 GB+ facilmente**
- Não usa lazy loading mesmo quando disponível

**Impacto**:
- Multiplicado por **N lambdas processados em paralelo**
- Exemplo: 50 lambdas × 4 workers × 3.2 GB = **640 GB de pico de RAM** (!)
- Mesmo que cada lambda tenha dados pequenos, **multiplicação é catastrófica**

**Comparação com código**:
- Existe `_load_dataframe_lazy()` (linha 54) que **não é usada** em `_load_and_validate_data()`
- A função lazy já existe mas está ignorada

---

### **ERRO 2: Múltiplas cópias de DataFrame durante processamento**

**Arquivo**: `portfolio_utils/portfolio_analyzer.py` - Funções várias

```python
# Linha 254: Retorna cópia vazia
return df_logs.iloc[0:0].copy()

# Linha 257: Cópia desnecessária
return df_logs.copy()

# Linha 300: Outra cópia
pareto_df = df_logs.iloc[candidate_indices].copy().sort_values('risk')

# Linha 419: Cópia em calculate_interpolation_errors
df_logs = df_logs.copy()
```

**Problema**:
- Cada `.copy()` duplica **todo o DataFrame na memória**
- Se df_logs tem 1M linhas e 20 colunas:
  - Original: 3.2 GB
  - Após `.copy()`: 6.4 GB (mesmo processo)
  - Após segunda cópia: 9.6 GB
  - Garbage collection lento → pico de 10+ GB simultaneamente

**Impacto em `analyze_portfolio_results()`**:
1. `_load_and_validate_data()` → carrega df_logs (3.2 GB)
2. `calculate_pareto_frontier()` → faz `.copy()` (6.4 GB)
3. `calculate_interpolation_errors()` → faz `df_logs.copy()` (9.6 GB)
4. `_calculate_metrics()` → mais cópias (12+ GB)
5. Isso acontece **para cada lambda**, **multiplicado por workers paralelos**

**Multiplicação de workers**:
- 4 workers × 4 lambdas simultâneos × 12 GB = **192 GB de pico**

---

### **ERRO 3: Concatenação de DataFrames sem limite (O(n²) overhead)**

**Arquivo**: `portfolio_utils/portfolio_analyzer.py` - Linha 126

```python
def calculate_pareto_frontier_lazy():
    pareto_frontier = None
    for chunk in _load_dataframe_lazy(file_path, chunk_size):
        if pareto_frontier is None:
            pareto_frontier = calculate_pareto_frontier(chunk)
        else:
            # ⚠️ PROBLEMA AQUI:
            combined = pd.concat([pareto_frontier, chunk], ignore_index=True)  # Cópia
            pareto_frontier = calculate_pareto_frontier(combined)  # Outra cópia
            del combined  # Garbage collection atrasado
```

**Problema**:
- `pd.concat()` cria **cópia intermediária**
- `calculate_pareto_frontier()` faz **outra cópia**
- Para 50 chunks:
  - Chunk 1: 100k rows → 2 cópias
  - Chunk 2: 100k + 100k → 4 cópias
  - Chunk 3: 300k + 100k → 8 cópias
  - ...padrão exponencial

**Impacto O(n²)**:
- Se processa 5M linhas em chunks de 100k:
  - Total de cópias: ~50² = 2500+ operações de cópia
  - Pico de RAM: soma de todas as cópias intermediárias
  - Exemplo: **(50 + 49 + 48 + ... + 1) × chunk_size = 1275 × 100k = 127.5M linhas copiadas**

---

### **ERRO 4: Paralelização aninhada SEM limite de processos**

**Arquivo**: `analyze_sweep_results.py` - Linha 75

```python
n_processes_inner = 2 if use_parallel_inner else 1
analyze_portfolio_results(
    output_dir=str(lambda_dir),
    use_parallel=use_parallel_inner,
    n_processes=n_processes_inner,
    risk_free_rate=risk_free_rate
)
```

**Problema**:
- Cada worker do Pool externo roda `analyze_portfolio_results()` com `n_processes=2`
- Se tem 4 workers externos × 2 processos internos = **8 processos paralelos**
- Cada processo carrega df_logs completo → **8 × 3.2 GB = 25.6 GB**
- Soma com overhead de cópias = **40+ GB de pico**

**Falta de verdadeira paralelização aninhada**:
- multiprocessing.Pool não coordena recursos entre níveis
- Não existe synchronization de memória compartilhada
- Cada processo é COMPLETAMENTE independente (sem compartilhamento)

---

### **ERRO 5: Armazenamento de resultados intermediários em `analysis_results`**

**Arquivo**: `analyze_sweep_results.py` - Linha 163

```python
# Armazenar resultados
for result in results:
    self.analysis_results[result['lambda']] = result  # Acumula tudo na RAM!
```

**Problema**:
- Acumula **todas as respostas de análise** na memória do processo principal
- Exemplo: 50 lambdas × 10 MB de metadata = 500 MB
- Parece pequeno, mas é**memória que não é necessária** enquanto análise está rodando
- Deveria ser escrito direto em JSON/arquivo

**Multiplicação de workers**:
- Cada worker retorna `result` que é recolhido no processo pai
- Processo pai aguarda **todos os workers terminarem** antes fazer garbage collection
- Se worker demorar 5 minutos, seus DataFrames ficam na memória 5 minutos extras

---

### **ERRO 6: Sem limpeza de cache entre lambdas**

**Arquivo**: `analyze_sweep_results.py` - Função `_analyze_lambda_worker()` (linha 47)

```python
def _analyze_lambda_worker(args):
    # ... processa lambda ...
    analyze_portfolio_results(...)
    # NÃO LIMPA:
    # - Cache de matplotlib
    # - DataFrames temporários
    # - Memória de interpolação
    # - Estruturas internas do pandas
```

**Problema**:
- Python não libera memória alocada até:
  1. Variável sair do escopo
  2. `del` ser chamado
  3. Garbage collection ser acionado (automático, mas lento)

**Impacto**:
- Após processar lambda_1, memória NÃO é liberada
- Lambda_2 aloca NOVO bloco de memória
- Ao fim de 50 lambdas: **50 blocos não liberados**
- Pico de RAM cresce linearmente: **baseline + 50 × 3.2 GB**

---

### **ERRO 7: Matplotlib gera gráficos em memória antes de salvar**

**Arquivo**: `portfolio_utils/portfolio_analyzer.py` - Função `plot_frontiers_comparison()` (linha 857+)

```python
def plot_frontiers_comparison(output_dir, df_logs, efficient_frontier=None, pareto_frontier=None):
    # Cria matplotlib figure (~500 MB para gráficos complexos)
    # Renderiza tudo em memória
    # Depois salva em arquivo
    # Figura não é deletada explicitamente
```

**Problema**:
- Matplotlib cria **buffer em memória** para cada gráfico
- Se gera 10+ gráficos por lambda × 50 lambdas = 500 gráficos
- Cada um ocupando 50-100 MB × 500 = **25-50 GB de picos**
- `plt.close()` não garante limpeza imediata de memória

**Sem cleanup**:
```python
# Falta isso:
plt.close('all')
gc.collect()  # Force garbage collection
```

---

### **ERRO 8: Lazy loading existe mas NÃO é usado na função principal**

**Arquivo**: `portfolio_utils/portfolio_analyzer.py`

```python
# Função lazy EXISTE:
def _load_dataframe_lazy(file_path, chunk_size=50000):
    # ...

# MAS É IGNORADA em _load_and_validate_data():
def _load_and_validate_data(output_dir, efficient_frontier_file):
    df_logs = _load_dataframe_flexible(...)  # ❌ Carrega TUDO
    # Não usa _load_dataframe_lazy()
```

**Problema**:
- Código otimizado foi escrito mas **não é chamado**
- Desenvolvimento técnico sem integração
- Cada nova versão provavelmente esquece dessa otimização

---

## 📊 ESTIMATIVA DE CONSUMO DE RAM

### Cenário atual (COM ERROS):

```
Configuração: 50 lambdas, 4 workers, 1M linhas por lambda, 20 colunas

Por lambda (sequencialmente):
  - df_logs carregado:                    3.2 GB
  - .copy() em calculate_pareto:          +3.2 GB (total: 6.4 GB)
  - .copy() em interpolation:             +3.2 GB (total: 9.6 GB)
  - Mais cópias em _calculate_metrics:    +3.2 GB (total: 12.8 GB)
  - Matplotlib gráficos:                  +2.0 GB (total: 14.8 GB)
  - Pico por lambda:                      ~15 GB

Com 4 workers paralelos:
  4 × 15 GB = 60 GB de pico RAM

Com multiplicação de cache não liberado:
  Baseline (SO + Python): 1 GB
  Worker 1-4 (simultâneos): 60 GB
  Worker 5-8 (próximas 4 lambdas, cache anterior não liberado): +15 GB
  ...
  PICO TOTAL: 80-100 GB 💥
```

### Cenário otimizado (SEM ERROS):

```
Com lazy loading + sem cópias desnecessárias:
  - Por lambda: 3.2 GB (carrega 1 chunk por vez)
  - Mantém apenas 1 chunk + resultado Pareto em RAM
  - Pico por lambda: ~1.5 GB
  
Com 4 workers:
  4 × 1.5 GB = 6 GB de pico RAM

Economia: **60-100 GB reduzido para ~6 GB** 🎯
```

---

## 📋 RESUMO DOS 8 ERROS CRÍTICOS

| # | Erro | Tipo | Impacto RAM | Localização |
|---|------|------|-------------|-------------|
| 1 | Carregamento completo de Parquet | CRÍTICO | +3.2 GB/lambda | `_load_and_validate_data()` |
| 2 | Múltiplas `.copy()` | CRÍTICO | +10-15 GB/lambda | `calculate_pareto_frontier()`, `calculate_interpolation_errors()` |
| 3 | Concatenação O(n²) de chunks | ALTO | +20-30 GB em lazy | `calculate_pareto_frontier_lazy()` |
| 4 | Paralelização aninhada descontrolada | ALTO | ×4 na multiplicação | `_analyze_lambda_worker()` |
| 5 | Armazenamento de resultados | MÉDIO | +500 MB | `SweepAnalyzer.analysis_results` |
| 6 | Sem limpeza entre lambdas | ALTO | +150 GB acumulado | Toda função que aloca |
| 7 | Matplotlib em memória | MÉDIO | +25-50 GB | `plot_frontiers_comparison()` |
| 8 | Lazy loading desintegrado | CRÍTICO | Não usado | `_load_dataframe_lazy()` vs `_load_and_validate_data()` |

---

## 🎯 Próximos Passos

Essas são as áreas que DEVEM ser refatoradas para reduzir consumo de RAM de **60-100 GB para ~6 GB**.

**Prioridade**:
1. ✅ Erro 1: Integrar lazy loading na função principal
2. ✅ Erro 2: Eliminar `.copy()` desnecessários
3. ✅ Erro 3: Refatorar concatenação de chunks
4. ✅ Erro 6: Adicionar garbage collection entre lambdas
5. ✅ Erro 7: Limpar matplotlib figures explicitamente
6. ✅ Erro 4: Considerar paralelização single-level ao invés de aninhada
7. ⚠️ Erro 5: Salvar resultados em streaming (JSON incremental)
8. ⚠️ Erro 8: Documentar que lazy loading deve ser usado para datasets > 500k linhas

