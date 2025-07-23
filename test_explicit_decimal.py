#!/usr/bin/env python3
"""
Teste da precisão decimal com parâmetro explícito
"""

from customhys.population import Population

def test_explicit_decimal_places():
    """Testa a especificação explícita de casas decimais"""
    
    print("🔧 Teste com Decimal Places Explícito\n")
    
    # Teste 1: Especificar 2 casas decimais explicitamente
    print("Teste 1: boundaries=[[0.0, 0.0], [1.0, 1.0]], decimal_places=2")
    boundaries = [[0.0, 0.0], [1.0, 1.0]]
    pop = Population(boundaries, num_agents=5, decimal_places=2, sum_constraint={'indices': [0, 1], 'target_sum': 1.0})
    pop.initialise_positions('random')
    
    positions = pop.get_positions()
    print(f"   Precisão configurada: {pop.decimal_places} casas decimais")
    print("   Posições geradas:")
    for i, pos in enumerate(positions):
        formatted_pos = [f"{x:.{pop.decimal_places}f}" for x in pos]
        soma = sum(pos)
        print(f"     Agente {i}: [{', '.join(formatted_pos)}] (soma = {soma:.6f})")
    
    print()
    
    # Teste 2: Usando strings para manter precisão
    print("Teste 2: boundaries=[['0.00', '0.00'], ['1.00', '1.00']] com auto_decimal_precision")
    boundaries = [['0.00', '0.00'], ['1.00', '1.00']]
    pop = Population(boundaries, num_agents=3, auto_decimal_precision=True, sum_constraint={'indices': [0, 1], 'target_sum': 1.0})
    pop.initialise_positions('random')
    
    positions = pop.get_positions()
    print(f"   Precisão inferida: {pop.decimal_places} casas decimais")
    print("   Posições geradas:")
    for i, pos in enumerate(positions):
        formatted_pos = [f"{x:.{pop.decimal_places if pop.decimal_places else 2}f}" for x in pos]
        soma = sum(pos)
        print(f"     Agente {i}: [{', '.join(formatted_pos)}] (soma = {soma:.6f})")

if __name__ == "__main__":
    test_explicit_decimal_places()
