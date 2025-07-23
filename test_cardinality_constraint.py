#!/usr/bin/env python3
"""
Test script for cardinality constraint functionality in population module.
This script demonstrates the cardinality constraint feature for portfolio optimization.
"""

import numpy as np
import sys
import os

# Add the customhys directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'customhys'))

from population import Population

def test_cardinality_constraint():
    """Test cardinality constraint functionality"""
    print("=" * 60)
    print("TESTING CARDINALITY CONSTRAINT")
    print("=" * 60)
    
    # Portfolio with 10 assets, boundaries [0.0, 1.0] for each asset
    # Correct format: (lower_bounds_list, upper_bounds_list)
    lower_bounds = [0.0] * 10
    upper_bounds = [1.0] * 10
    boundaries = (lower_bounds, upper_bounds)
    
    # Test 1: Cardinality constraint only (select 3 out of 10 assets)
    print("\nTest 1: Cardinality constraint only (3 out of 10 assets)")
    print("-" * 50)
    
    pop1 = Population(
        boundaries=boundaries,
        num_agents=5,
        cardinality_constraint=3,
        auto_decimal_precision=False
    )
    
    # Initialize positions
    pop1.initialise_positions(scheme='random')
    
    positions = pop1.get_positions()
    print(f"Population shape: {positions.shape}")
    
    for i, agent in enumerate(positions):
        non_zero_count = np.count_nonzero(agent)
        non_zero_indices = np.where(agent != 0.0)[0]
        print(f"Agent {i}: Non-zero count = {non_zero_count}, Indices = {non_zero_indices}")
        print(f"  Values: {agent}")
    
    # Test 2: Cardinality + Sum constraint (portfolio weights sum to 1.0)
    print("\nTest 2: Cardinality + Sum constraint (3 assets, sum = 1.0)")
    print("-" * 50)
    
    sum_constraint = {
        'indices': list(range(10)),  # All assets
        'target_sum': 1.0
    }
    
    pop2 = Population(
        boundaries=boundaries,
        num_agents=5,
        cardinality_constraint=3,
        sum_constraint=sum_constraint,
        auto_decimal_precision=False
    )
    
    # Initialize positions
    pop2.initialise_positions(scheme='random')
    
    positions2 = pop2.get_positions()
    
    for i, agent in enumerate(positions2):
        non_zero_count = np.count_nonzero(agent)
        non_zero_indices = np.where(agent != 0.0)[0]
        total_sum = np.sum(agent)
        print(f"Agent {i}: Non-zero count = {non_zero_count}, Sum = {total_sum:.6f}")
        print(f"  Non-zero indices: {non_zero_indices}")
        print(f"  Values: {agent}")
    
    # Test 3: Cardinality + Sum + Decimal precision
    print("\nTest 3: Cardinality + Sum + Decimal precision (3 assets, sum = 1.0, 2 decimal places)")
    print("-" * 50)
    
    pop3 = Population(
        boundaries=boundaries,
        num_agents=5,
        cardinality_constraint=3,
        sum_constraint=sum_constraint,
        decimal_places=2
    )
    
    # Initialize positions
    pop3.initialise_positions(scheme='random')
    
    positions3 = pop3.get_positions()
    
    for i, agent in enumerate(positions3):
        non_zero_count = np.count_nonzero(agent)
        non_zero_indices = np.where(agent != 0.0)[0]
        total_sum = np.sum(agent)
        print(f"Agent {i}: Non-zero count = {non_zero_count}, Sum = {total_sum:.6f}")
        print(f"  Non-zero indices: {non_zero_indices}")
        print(f"  Values: {agent}")
    
    # Test 4: Different cardinality values
    print("\nTest 4: Different cardinality values")
    print("-" * 50)
    
    for cardinality in [1, 2, 5, 10]:
        print(f"\nCardinality = {cardinality}:")
        
        pop = Population(
            boundaries=boundaries,
            num_agents=1,
            cardinality_constraint=cardinality,
            auto_decimal_precision=False
        )
        
        # Initialize positions
        pop.initialise_positions(scheme='random')
        
        agent = pop.get_positions()[0]
        non_zero_count = np.count_nonzero(agent)
        non_zero_indices = np.where(agent != 0.0)[0]
        print(f"  Non-zero count: {non_zero_count} (expected: {cardinality})")
        print(f"  Non-zero indices: {non_zero_indices}")
        print(f"  Values: {agent}")

def test_error_conditions():
    """Test error conditions for cardinality constraint"""
    print("\n" + "=" * 60)
    print("TESTING ERROR CONDITIONS")
    print("=" * 60)
    
    # Portfolio with 5 assets
    lower_bounds = [0.0] * 5
    upper_bounds = [1.0] * 5
    boundaries = (lower_bounds, upper_bounds)
    
    # Test invalid cardinality values
    invalid_cardinalities = [0, -1, 6, 'invalid']
    
    for invalid_card in invalid_cardinalities:
        print(f"\nTesting invalid cardinality: {invalid_card}")
        try:
            pop = Population(
                boundaries=boundaries,
                cardinality_constraint=invalid_card
            )
            print(f"  ERROR: Should have raised exception for cardinality {invalid_card}")
        except Exception as e:
            print(f"  OK: Correctly raised exception: {e}")

if __name__ == "__main__":
    test_cardinality_constraint()
    test_error_conditions()
    print("\n" + "=" * 60)
    print("CARDINALITY CONSTRAINT TESTS COMPLETED")
    print("=" * 60)
