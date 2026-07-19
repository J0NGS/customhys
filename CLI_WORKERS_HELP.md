# 🎯 Novo Argumento CLI: `--workers`

## Sumário

Adicionado argumento `--workers` ao script `portfolio_optimizer_lambda_sweep.py` para permitir especificar o número de workers paralelos.

## Uso

### Comando Básico (Padrão)
```bash
python portfolio_optimizer_lambda_sweep.py port2.txt 3 config.txt portef2.txt 50
```
- Usa **cpu_count // 2** workers automaticamente (metade dos cores disponíveis)

### Com Número Específico de Workers

#### 4 workers:
```bash
python portfolio_optimizer_lambda_sweep.py port2.txt 3 config.txt portef2.txt 50 --workers 4
```

#### 8 workers:
```bash
python portfolio_optimizer_lambda_sweep.py port2.txt 3 config.txt portef2.txt 50 --workers 8
```

#### 1 worker (quase sequencial, mas com overhead de Pool):
```bash
python portfolio_optimizer_lambda_sweep.py port2.txt 3 config.txt portef2.txt 50 --workers 1
```

#### Usar TODOS os cores:
```bash
python portfolio_optimizer_lambda_sweep.py port2.txt 3 config.txt portef2.txt 50 --workers <max_cores>
# Ex: Se tem 16 cores, fazer --workers 16
```

## Comportamento

### Modo Padrão (sem `--workers`)
```
[INFO] Usando 8 workers (metade dos 16 cores disponíveis)
```
- Automático: `cpu_count() // 2`
- Balanceado: deixa metade dos cores livres para SO e outras tarefas

### Com `--workers N`
```
[INFO] Usando 4 workers (especificado via --workers)
```
- Usa exatamente N workers
- Se N > cpu_count(): vai usar anyway (pode ficar lento)
- Se N = 0 ou negativo: convertido para 1

### Com `--sequential`
```bash
python portfolio_optimizer_lambda_sweep.py port2.txt 3 config.txt portef2.txt 50 --sequential
```
- Ignora `--workers` completamente
- Executa lambdas um por um, sem paralelismo

## Exemplos Práticos

### Teste Rápido (1 worker, overhead mínimo)
```bash
python portfolio_optimizer_lambda_sweep.py port2.txt 3 config.txt portef2.txt 3 --workers 1
```

### Uso Agressivo (todos os cores)
```bash
python portfolio_optimizer_lambda_sweep.py port2.txt 3 config.txt portef2.txt 50 --workers 16
```
Útil quando a máquina é dedicada ao sweep.

### Uso Conservador (1/4 dos cores)
```bash
python portfolio_optimizer_lambda_sweep.py port2.txt 3 config.txt portef2.txt 50 --workers 4
```
Mantém sistema responsivo para outras tarefas.

### Debug (sequencial + verbose)
```bash
python portfolio_optimizer_lambda_sweep.py port2.txt 3 config.txt portef2.txt 3 --sequential --no-analysis
```

## Help Completo

```bash
python portfolio_optimizer_lambda_sweep.py --help
```

Output mostra:
```
options:
  --workers WORKERS     Número de workers paralelos (padrão: cpu_count // 2).
                       Ignorado com --sequential
```

## Implementação

### Adicionado ao parser:
```python
parser.add_argument('--workers', type=int, default=None,
                    help='Número de workers paralelos (padrão: cpu_count // 2). Ignorado com --sequential')
```

### Lógica de decisão:
```python
if args.workers is not None:
    n_workers = max(1, args.workers)
    print(f"[INFO] Usando {n_workers} workers (especificado via --workers)")
else:
    n_workers = max(1, cpu_count() // 2)
    print(f"[INFO] Usando {n_workers} workers (metade dos {cpu_count()} cores)")
```

## Notas

- **`--workers` é ignorado quando usar `--sequential`** - O modo sequencial sempre executa com 1 processo
- **Validação:** O número é validado com `max(1, args.workers)` - nunca usa 0 workers
- **CPU Count:** Detectado automaticamente via `multiprocessing.cpu_count()`
- **Padrão balanceado:** `cpu_count() // 2` mantém o sistema responsivo

## Casos de Uso Comuns

| Situação | Comando |
|----------|---------|
| Máquina com 16 cores, uso balanceado | `--workers 8` |
| Máquina com 16 cores, dedicada | `--workers 16` |
| Máquina com 8 cores, conservador | `--workers 4` |
| Teste rápido | `--workers 1` |
| Debug | `--sequential` |

