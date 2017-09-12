# -*- encoding: utf-8 -*-

"""
Contains pheromone memory implementations for the use in ACO
algorithms.
"""

from misc import array_double


class PheromoneMatrix(object):
    """
    Impl. of a basic (or standard) pheromone memory in the ACO
    in a form of a 2D matrix.
    Instances of PheromoneMatrix can be accessed like a 2d matrix, i.e.
    obj[a][b] = ...
    """
    def __init__(self, dimension, initial_value):
        self.initial_value = initial_value
        initial_row = [initial_value for _ in range(dimension)]
        # We use array.array as it provides more memory efficient storage
        # of the same type values
        self.pheromone = [array_double(initial_row)
                          for _ in range(dimension)]

    def __getitem__(self, index):
        """
        Overloads [] operator so that we can access the pheromone memory
        similarly to a 2d matrix.
        """
        return self.pheromone[index]

    def get_trail(self, beg, end):
        return self.pheromone[beg][end]

    def update(self, node_a, node_b,
               evap_rate, deposit=0.0,
               symmetric=True):
        """
        Updates pheromone value for an edge (node_a, node_b)
        using the formula: old_value * (1 - evap_rate) + evap_rate * deposit
        """
        pheromone_a = self.pheromone[node_a]
        updated = pheromone_a[node_b] * (1. - evap_rate) + evap_rate * deposit
        pheromone_a[node_b] = updated
        if symmetric:
            self.pheromone[node_b][node_a] = updated

    def print_debug_info(self):
        """
        This is for debugging purposes.
        """
        pass
