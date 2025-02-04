# Importar os módulos necessários
import customhys.population as Population
import customhys.hyperheuristic as Hyperheuristic
import operator as Operators
import numpy as np
import os

# Caminho do arquivo enviado
file_path = '/content/drive/MyDrive/instances/newInstances/small/knapPI_1_50_1000.csv'

# Função para ler as instâncias
def read_knapsack_instances(file_path):
    instances = []

    # Abrir o arquivo e ler seu conteúdo
    with open(file_path, 'r') as file:
        content = file.read()

    # Dividir o conteúdo em instâncias com base na linha de separação
    parts = content.split('-----')

    # Processar cada parte
    for part in parts:
        # Limpar e verificar se a parte não está vazia
        part = part.strip()
        if not part:
            continue

        # Dividir a parte em linhas
        lines = part.split('\n')

        # Pegar o nome da instância
        instance_name = lines[0].strip()

        # Pegar o número de itens, capacidade, melhor solução e tempo
        n = int(lines[1].split()[1])
        c = int(lines[2].split()[1])
        z = int(lines[3].split()[1])
        time = float(lines[4].split()[1])

        # Pegar os dados dos itens
        items = []
        for line in lines[5:]:
            if line.strip():
                # Dividir a linha em colunas usando vírgulas como delimitadores
                cols = line.split(',')
                if len(cols) == 4:
                    index = int(cols[0])
                    value = int(cols[1])
                    weight = int(cols[2])
                    in_solution = int(cols[3])
                    items.append({
                        'index': index,
                        'value': value,
                        'weight': weight,
                        'in_solution': in_solution
                    })

        # Adicionar a instância à lista
        instances.append({
            'name': instance_name,
            'n': n,
            'capacity': c,
            'optimal_value': z,
            'time': time,
            'items': items
        })

    return instances


import customhys.hyperheuristic as hh

## Função para calcular o valor da solução
def knapsack_objective(solution, capacity):
    print("--solução recebida--")
    total_weight = 0
    total_value = 0
    for i in solution:
        total_weight += i['weight']
        total_value += i['value']
    if total_weight > capacity:
        return -1  # Penalizar soluções que excedem a capacidade
    print(f"--valor da  solução: {total_value}--")
    return total_value

# Função para avaliar soluções
def evaluate_knapsack(solution, instance): #Debug
    solution_items = [item for item in instance['items'] if item['index'] in solution]
    return knapsack_objective(solution_items, instance['capacity'])

# Função principal para executar a hyperheurística
def run_hyperheuristic_with_instances(file_path):
    """
    Função para executar a hyperheurística utilizando as instâncias carregadas.

    Args:
        file_path (str): Caminho do arquivo com as instâncias.
    """
    instances = read_knapsack_instances(file_path)
    instance = instances[0]  # Usar apenas a primeira instância para teste
    print(f"Resolvendo instância: {instance['name']}")

    # Configurar o problema para o formato aceito
    problem = {
        "dimension": instance['n'],  # Dimensão baseada no número de itens
        "domain": [item['index'] for item in instance['items']],  # Domínio corresponde aos índices dos itens
        "function": lambda solution: evaluate_knapsack(solution, instance),
        "boundaries": [ [item['index'] for item in instance['items']]] * instance['n'],  # Limites binários para seleção
        "is_constrained": True
    }



        # Adicionar o parâmetro 'learning_portion'
    parameters = {
        "max_evaluations": 100,
        "time_limit": 60,
        "as_mh": True,
        "cardinality": 5,
        "cardinality_min": 2,
        "num_iterations": 50,
        "solver": "dynamic",
        "num_replicas": 1,
        "num_agents": 3,
        "num_steps": 5,
        "stagnation_percentage": 0.25,
        "cooling_rate": 0.9,
        "max_temperature": 100,
        "trial_overflow": True,
        "allow_weight_matrix": False,
        "repeat_operators": True,
        "verbose": True,
        "learning_portion": 0.3  # Define a proporção para aprendizado
    }

    # Inicializar a hyperheurística
    hyper = hh.Hyperheuristic(
        heuristic_space="default.txt",  # Arquivo com heurísticas pré-definidas
        problem=problem,
        parameters=parameters,
        file_label=instance['name']
    )
    # Executar a busca
    result = hyper.solve()  # Capturar o retorno
    best_solution = result[0]
    best_value = result[1]
    # Mostrar resultados
    print(f"Solução encontrada (itens selecionados): {best_solution}")
    print(f"Valor da solução: {best_value}")
    print(f"Valor ótimo esperado: {instance['optimal_value']}\n")


# Exemplo de execução
file_path = "C:/Users/João Neto/Desktop/small/knapPI_1_50_1000.csv"
run_hyperheuristic_with_instances(file_path)