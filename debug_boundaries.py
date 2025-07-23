#!/usr/bin/env python3
"""
Debug dos boundaries originais
"""

from customhys.population import Population

# Teste para ver o que está sendo armazenado
boundaries = [[0.00, 0.00], [1.00, 1.00]]
print(f"Boundaries: {boundaries}")
print(f"Tipo dos elementos: {[type(x) for x in boundaries[0]]}")

pop = Population(boundaries, num_agents=1, auto_decimal_precision=True)
print(f"Original boundaries: {pop._original_boundaries}")
print(f"Tipo dos elementos originais: {[type(x) for x in pop._original_boundaries[0]]}")

# Teste direto da função
print(f"Decimal places inferido: {pop.decimal_places}")

# Teste com strings explícitas
print("\n--- Teste com strings ---")
boundaries_str = [["0.00", "0.00"], ["1.00", "1.00"]]
print(f"Boundaries string: {boundaries_str}")

# Simular o que deveria acontecer
max_decimal_places = 0
all_boundaries = list(boundaries_str[0]) + list(boundaries_str[1])
for boundary_value in all_boundaries:
    str_value = str(boundary_value)
    if '.' in str_value:
        decimal_part = str_value.split('.')[1]
        decimal_places = len(decimal_part)
        max_decimal_places = max(max_decimal_places, decimal_places)
        print(f"  {boundary_value} -> {decimal_places} casas decimais")

print(f"Máximo: {max_decimal_places} casas decimais")
