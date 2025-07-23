# Portfolio Optimizer usando CustomHyS

Este sistema implementa otimização de portfólio com restrição de cardinalidade usando a biblioteca CustomHyS, baseado no artigo de Chang et al. (2000).

## Arquivos Principais

- `portfolio_optimizer.py`: Script principal de otimização
- `config_hh_default.txt`: Arquivo de configuração exemplo da hyper-heurística
- `run_example.py`: Script de exemplo de uso

## Como Usar

### Sintaxe Básica
```bash
python portfolio_optimizer.py <arquivo_instancia> <cardinalidade> <arquivo_config>
```

### Parâmetros

1. **arquivo_instancia**: Nome do arquivo de instância da OR-Library (ex: port1.txt, port2.txt, etc.)
2. **cardinalidade**: Número inteiro para restrição de cardinalidade
   - `0`: Sem restrição de cardinalidade (k=None)
   - `1 até N`: Restrição para exatamente K ativos
3. **arquivo_config**: Arquivo JSON com configuração da hyper-heurística

### Exemplos de Uso

```bash
# Sem restrição de cardinalidade
python portfolio_optimizer.py port1.txt 0 config_hh_default.txt

# Com restrição de 5 ativos
python portfolio_optimizer.py port1.txt 5 config_hh_default.txt

# Usando arquivo port5 com 10 ativos
python portfolio_optimizer.py port5.txt 10 config_hh_default.txt
```

## Formato do Arquivo de Configuração

O arquivo de configuração deve ser um JSON válido com os seguintes parâmetros:

```json
{
    "cardinality": 10,           # Máximo de operadores na sequência
    "cardinality_min": 2,        # Mínimo de operadores na sequência
    "num_iterations": 100,       # Número de iterações
    "num_agents": 30,            # Tamanho da população
    "as_mh": true,              # Usa a sequência de HH como uma metaheurística completa?
    "allow_weight_matrix": true, # Permite matriz de pesos
    "num_replicas": 10,         # Número de réplicas por MH
    "num_steps": 20,            # Número de tentativas por passo da HH
    "stagnation_percentage": 0.40, # Percentual de estagnação para controle
    "max_temperature": 1,        # Temperatura inicial (Simulated Annealing)
    "min_temperature": 1e-6,     # Temperatura mínima (Simulated Annealing)
    "cooling_rate": 0.01,        # Taxa de resfriamento (Simulated Annealing)
    "temperature_scheme": "fast", # Esquema de temperatura
    "acceptance_scheme": "exponential", # Critério de aceitação de soluções
    "trial_overflow": false,     # Política de overflow de tentativas
    "repeat_operators": true,    # Permite repetição de operadores na sequência
    "verbose": true,             # Exibir logs e progresso
    "learning_portion": 0.3,     # Percentual de aprendizado das sequências
    "solver": "static",          # Tipo de solver
    "initial_scheme": "random"   # Esquema inicial de seleção de operadores
}
```

## Saída do Programa

O programa cria um diretório com nome no formato:
```
YYYYMMDD_HHMMSS_<nome_instancia>_<metodo_inicial>
```

Exemplo: `20250716_143052_port1_random`

### Arquivos Gerados

1. **execution_logs.csv**: Log detalhado de todas as avaliações da função objetivo
2. **hh_config.json**: Configuração da hyper-heurística utilizada
3. **instance_data.json**: Dados da instância (retornos, matriz de covariância, etc.)
4. **hh_result.json**: Resultado final da hyper-heurística
5. **summary_stats.json**: Estatísticas resumidas da melhor solução

## Exemplo de Execution Log

Cada avaliação da função objetivo gera um registro com:

```json
{
    "execution_number": 1,
    "weights": [0.25, 0.35, 0.40, 0.0],
    "selected_assets": [0, 1, 2],
    "expected_return": 0.1150,
    "risk": 0.1820,
    "sharpe": 0.4670,
    "variance": 0.0331,
    "objective": 0.0016,
    "timestamp": "2025-07-16T14:30:52.123456"
}
```

## Funcionalidades

### Demonstração da Instância
O programa exibe os primeiros 5 e últimos 5 ativos da instância carregada, mostrando:
- Retorno esperado (μ)
- Desvio padrão (σ)

### Restrição de Cardinalidade
- Quando k=0: Otimização sem restrição de cardinalidade
- Quando k>0: Seleciona exatamente k ativos com maiores pesos
- Normalização automática dos pesos para somar 1

### Métricas Calculadas
- **Retorno Esperado**: E[R] = Σ(wi × μi)
- **Risco**: σ = √(wᵀ × Σ × w)
- **Índice de Sharpe**: (E[R] - Rf) / σ
- **Função Objetivo**: λ × σ² - (1-λ) × E[R]

## Executando os Exemplos

### Exemplo Básico
```bash
python run_example.py
```

Este comando executa automaticamente:
```bash
python portfolio_optimizer.py port1.txt 3 config_hh_default.txt
```

### Bateria de Testes
Para executar múltiplos cenários de teste:
```bash
python test_portfolio_optimizer.py
```

Este script testa:
- port1.txt sem restrição de cardinalidade
- port1.txt com 3 ativos
- port2.txt com 5 ativos

### Configurações Disponíveis

- **config_hh_default.txt**: Configuração completa (100 iterações, 30 agentes)
- **config_hh_fast.txt**: Configuração rápida para testes (20 iterações, 10 agentes)

## Dependências

- numpy
- pandas
- customhys
- scipy (para funcionalidades futuras)

## Notas

- O programa utiliza taxa livre de risco padrão de 3% (0.03)
- O parâmetro λ (lambda) é fixado em 0.5 para balancear risco e retorno
- Os logs são salvos com timestamp para análise posterior
- O programa valida a existência dos arquivos antes de executar
