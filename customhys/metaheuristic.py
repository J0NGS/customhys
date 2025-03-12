# -*- coding: utf-8 -*-
"""
This module contains the Metaheuristic class.

Created on Thu Sep 26 16:56:01 2019

@author: Jorge Mario Cruz-Duarte (jcrvz.github.io), e-mail: jorge.cruz@tec.mx
"""

import numpy as np
from . import operators as Operators
from .population import Population

__all__ = ['Metaheuristic', 'Population', 'Operators']
__operators__ = Operators.__all__
__selectors__ = ['greedy', 'probabilistic', 'metropolis', 'all', 'none']


class Metaheuristic:
    """
        This is the Metaheuristic class, each object corresponds to a metaheuristic implemented with a sequence of
        search operators from op, and it is based on a population from Population.
    """
    def __init__(self, problem, search_operators=None, num_agents: int = 30, num_iterations: int = 100,
                 initial_scheme: str = 'random', verbose: bool = False):
        """
        Create a population-based metaheuristic by employing different simple search operators.
        """
        print("--- LOG: Iniciando Metaheuristic ---")
        print(f"--- LOG: Parâmetros recebidos ---")
        print(f"Num_agents: {num_agents}, Num_iterations: {num_iterations}, Initial_scheme: {initial_scheme}")
        print(f"Verbose: {verbose}")
        
        # Read the problem function
        self.finalisation_conditions = None
        self._problem_function = problem['function']
        
        print("--- LOG: Criando população ---")
        # Create population
        self.pop = Population(problem['boundaries'], num_agents, problem['is_constrained'])

        # Check and read the search_operators
        if search_operators:
            print(f"--- LOG: Verificando search_operators (total: {len(search_operators)}) ---")
            if not isinstance(search_operators, list):
                search_operators = [search_operators]
            print("--- LOG: Estrutura de search_operators ---")
            for i, op in enumerate(search_operators):
                print(f"Operador {i}: {op}")
            
            self.perturbators, self.selectors = Operators.process_operators(search_operators)
            print("--- LOG: process_operators executado com sucesso ---")

        # Define the maximum number of iterations
        self.num_iterations = num_iterations
        print("--- LOG: Número máximo de iterações definido ---")

        # Read the number of dimensions
        self.num_dimensions = self.pop.num_dimensions
        print(f"--- LOG: Número de dimensões: {self.num_dimensions} ---")

        # Read the number of agents
        self.num_agents = num_agents
        print(f"--- LOG: Número de agentes: {self.num_agents} ---")

        # Initialise historical variables
        self.historical = dict()
        print("--- LOG: Variáveis históricas inicializadas ---")

        # Set additional variables
        self.verbose = verbose
        print(f"--- LOG: Verbose setado como: {self.verbose} ---")

        # Set the initial scheme
        self.initial_scheme = initial_scheme
        print(f"--- LOG: Initial_scheme setado como: {self.initial_scheme} ---")
        print("--- LOG: Metaheuristic inicializada com sucesso ---")

    def apply_initialiser(self):
        # Set initial iteration
        self.pop.iteration = 0

        # Initialise the population
        self.pop.initialise_positions(self.initial_scheme)  # Default: random

        # Evaluate fitness values
        self.pop.evaluate_fitness(self._problem_function)

        # Update population, particular, and global
        self.pop.update_positions('population', 'all')  # Default
        self.pop.update_positions('particular', 'all')
        self.pop.update_positions('global', 'greedy')

    def apply_search_operator(self, perturbator, selector):
        # Split operator
        operator_name, operator_params = perturbator.split('(')

        # Apply an operator
        exec('Operators.' + operator_name + '(self.pop,' + operator_params)

        # Evaluate fitness values
        self.pop.evaluate_fitness(self._problem_function)

        # Update population
        if selector in __selectors__:
            self.pop.update_positions('population', selector)
        else:
            self.pop.update_positions()

        # Update global position
        self.pop.update_positions('global', 'greedy')

    def run(self):
        """
        Run the metaheuristic for solving the defined problem.
        :return: None.
        """
        if (not self.perturbators) or (not self.selectors):
            raise Operators.OperatorsError("There are not perturbator or selector!")

        # Apply initialiser / Random Sampling
        self.apply_initialiser()

        # Initialise and update historical variables
        self.reset_historicals()
        self.update_historicals()

        # Report which operators are going to use
        self._verbose('\nSearch operators to employ:')
        for perturbator, selector in zip(self.perturbators, self.selectors):
            self._verbose("{} with {}".format(perturbator, selector))
        self._verbose("{}".format('-' * 50))

        # Start optimisaton procedure
        while not self.finaliser():
            # Update the current iteration
            self.pop.iteration += 1

            # Implement the sequence of operators and selectors
            for perturbator, selector in zip(self.perturbators, self.selectors):

                # Apply the corresponding search operator
                self.apply_search_operator(perturbator, selector)

                # Update historical variables
                self.update_historicals()

            # Verbose (if so) some information
            self._verbose('{}\npop. radius: {}'.format(self.pop.iteration, self.historical['radius'][-1]))
            self._verbose(self.pop.get_state())

    def set_finalisation_conditions(self, conditions):
        # TODO: Check that it works for budgets <=
        if not isinstance(conditions, list):
            conditions = list(conditions)

        self.finalisation_conditions = conditions

    def finaliser(self):
        criteria = self.pop.iteration >= self.num_iterations
        if self.finalisation_conditions is not None:
            for condition in self.finalisation_conditions:
                criteria |= condition()

        return criteria

    def get_solution(self):
        """
        Deliver the last position and fitness value obtained after ``run`` the metaheuristic procedure.
        :returns: ndarray, float
        """
        return self.historical['position'][-1], self.historical['fitness'][-1]

    def reset_historicals(self):
        """
        Reset the ``historical`` variables.
        :return: None.
        """
        self.historical = dict(fitness=list(), position=list(), centroid=list(), radius=list())

    def update_historicals(self):
        """
        Update the ``historical`` variables.
        :return: None.
        """
        # Update historical variables
        self.historical['fitness'].append(np.copy(self.pop.global_best_fitness))
        self.historical['position'].append(np.copy(self.pop.global_best_position))

        # Update population centroid and radius
        current_centroid = np.array(self.pop.positions).mean(0)
        self.historical['centroid'].append(np.copy(current_centroid))
        self.historical['radius'].append(np.max(np.linalg.norm(self.pop.positions - np.tile(
            current_centroid, (self.num_agents, 1)), 2, 1)))

    def _verbose(self, text_to_print):
        """
        Print each step performed during the solution procedure. It only works if ``verbose`` flag is True.
        :param str text_to_print:
            Explanation about what the metaheuristic is doing.
        :return: None.
        """
        if self.verbose:
            print(text_to_print)
