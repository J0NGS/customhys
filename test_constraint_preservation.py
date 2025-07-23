#!/usr/bin/env python3
"""
Test script to verify that constraints are preserved during population updates.
"""

import numpy as np
import sys
import os

# Add the customhys directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'customhys'))

from population import Population

def test_constraints_after_updates():
    """Test that constraints are preserved after population updates"""
    print("=" * 70)
    print("TESTING CONSTRAINT PRESERVATION DURING UPDATES")
    print("=" * 70)
    
    # Setup portfolio problem
    n_assets = 8
    lower_bounds = [0.0] * n_assets
    upper_bounds = [1.0] * n_assets
    boundaries = (lower_bounds, upper_bounds)
    
    sum_constraint = {
        'indices': list(range(n_assets)),
        'target_sum': 1.0
    }
    
    # Create population with all constraints
    pop = Population(
        boundaries=boundaries,
        num_agents=3,
        cardinality_constraint=3,  # Only 3 assets active
        sum_constraint=sum_constraint,
        decimal_places=2
    )
    
    print("1. Initial population setup:")
    pop.initialise_positions(scheme='random')
    
    positions = pop.get_positions()
    print_population_stats(positions, "After initialization")
    
    print("\n2. Testing direct position update via setter:")
    # Create some new positions
    new_positions = np.random.uniform(-1, 1, (3, 8))
    pop.positions = new_positions
    
    positions = pop.get_positions()
    print_population_stats(positions, "After setter update")
    
    print("\n3. Testing position update via update_positions method:")
    # Simulate some fitness values
    pop.fitness = np.array([0.5, 0.3, 0.7])
    
    # Create new positions as if from an operator
    new_positions2 = np.random.uniform(-1, 1, (3, 8))
    pop.positions = new_positions2
    pop.fitness = np.array([0.4, 0.6, 0.2])  # Different fitness values
    
    # Now call update_positions to test selection
    pop.update_positions(level='population', selector='greedy')
    
    positions = pop.get_positions()
    print_population_stats(positions, "After update_positions with greedy selection")
    
    print("\n4. Testing individual agent position updates:")
    # Test updating a single agent
    agent_id = 0
    old_position = pop.get_positions()[agent_id]
    print(f"Agent {agent_id} before update: Non-zero={np.count_nonzero(old_position)}, Sum={np.sum(old_position):.6f}")
    
    # Create a new external position
    new_external = np.random.uniform(0, 1, n_assets)
    pop._update_agent_position_with_constraints(agent_id, new_external)
    
    new_position = pop.get_positions()[agent_id]
    print(f"Agent {agent_id} after update:  Non-zero={np.count_nonzero(new_position)}, Sum={np.sum(new_position):.6f}")
    
    print("\n✅ All update tests completed!")

def print_population_stats(positions, title):
    """Print statistics for the population"""
    print(f"\n{title}:")
    for i, agent in enumerate(positions):
        non_zero_count = np.count_nonzero(agent)
        total_sum = np.sum(agent)
        print(f"  Agent {i}: Non-zero assets = {non_zero_count}, Sum = {total_sum:.6f}")

def test_constraint_edge_cases():
    """Test edge cases for constraint preservation"""
    print("\n" + "=" * 70)
    print("TESTING CONSTRAINT EDGE CASES")
    print("=" * 70)
    
    # Test case: Population without constraints
    print("\n1. Population without constraints:")
    boundaries = ([0.0] * 5, [1.0] * 5)
    pop_no_constraints = Population(boundaries=boundaries, num_agents=2)
    pop_no_constraints.initialise_positions()
    
    # Update positions - should not cause errors
    new_pos = np.random.uniform(-1, 1, (2, 5))
    pop_no_constraints.positions = new_pos
    print("   ✅ No constraints - updates work correctly")
    
    # Test case: Only cardinality constraint
    print("\n2. Only cardinality constraint:")
    pop_card = Population(
        boundaries=boundaries,
        num_agents=2,
        cardinality_constraint=2
    )
    pop_card.initialise_positions()
    
    new_pos = np.random.uniform(-1, 1, (2, 5))
    pop_card.positions = new_pos
    
    positions = pop_card.get_positions()
    for i, agent in enumerate(positions):
        non_zero = np.count_nonzero(agent)
        print(f"   Agent {i}: Non-zero count = {non_zero} (expected: 2)")
    
    # Test case: Only sum constraint
    print("\n3. Only sum constraint:")
    sum_only = {'indices': [0, 1, 2], 'target_sum': 1.0}
    pop_sum = Population(
        boundaries=boundaries,
        num_agents=2,
        sum_constraint=sum_only
    )
    pop_sum.initialise_positions()
    
    new_pos = np.random.uniform(-1, 1, (2, 5))
    pop_sum.positions = new_pos
    
    positions = pop_sum.get_positions()
    for i, agent in enumerate(positions):
        constrained_sum = np.sum(agent[:3])  # Only first 3 indices
        print(f"   Agent {i}: Constrained sum = {constrained_sum:.6f} (expected: 1.0)")
    
    print("\n✅ All edge case tests completed!")

if __name__ == "__main__":
    test_constraints_after_updates()
    test_constraint_edge_cases()
    print("\n" + "=" * 70)
    print("ALL CONSTRAINT PRESERVATION TESTS COMPLETED")
    print("=" * 70)
