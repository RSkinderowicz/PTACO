# -*- encoding: utf-8 -*-

"""
Impl. of the Ant Colony System hybridized with the Simulated Annealing (SA)
to improve the convergence.
"""

from random import random
from math import exp

from acs import AntColonySystem


class ACST(AntColonySystem):
    """
    Implementation of the ACST algorithm, i.e. the ACS with a temperature
    parameter. The main difference to the ACS is the introduction of an active
    solution which is used when the global pheromone update is performed. The
    active solution is chosen among ants' solutions based on the quality of a
    candidate solution and the value of the temperature paramter.  Generally,
    it is equivalent to the selection criterion used in the Simulated Annealing
    algorithm.

    The active solution does not necessarily have to be the global best
    solution thus the search process becomes less exploitative and prone to
    getting stuck in local minima.
    """
    def __init__(self, problem,
                 initial_temperature=None,
                 **kwargs):
        AntColonySystem.__init__(self, problem, **kwargs)
        self.initial_temperature = initial_temperature
        self.temperature = 0.0
        self.active_solution = None

    def _init_temperature(self):
        """
        This is responsible for the calculation of an initial SA temperature
        value based on a sample of randomly generated instances.
        """
        self.stats['temperatures'] = []
        self.stats['initial_temperature'] = self.initial_temperature
        self.set_temperature(self.initial_temperature)

    def set_temperature(self, temp):
        """
        This overrides current SA temperature.
        """
        self.temperature = temp
        self.stats['temperatures'].append((temp, self.iteration))

    def run_init(self):
        AntColonySystem.run_init(self)
        self._init_temperature()
        self.active_solution = None
        self.stats['worse_accepted'] = 0
        self.stats['worse_rejected'] = 0

    def run(self, iterations, run_from_start=True):
        """
        Runs the algorithm for a given number of iterations.
        Returns the global best ant, i.e. the best solution found.
        """
        if run_from_start:
            self.run_init()
        for _ in range(iterations):
            self._build_ant_solutions()
            self._update_iteration_best()
            self._update_global_best()
            self._update_active_solution()
            self.global_pheromone_update(self.active_solution)
            self.iteration += 1
        return self.global_best

    def _update_active_solution(self):
        """
        Tries to update active_solution based on the ants solutions and
        the temperature value.
        """
        if not self.active_solution:
            self.active_solution = self.ants[0]
        for ant in self.ants:
            delta = (ant.value -
                     self.active_solution.value)/self.global_best.value
            if self.try_accept_solution_by_delta(delta):
                self.active_solution = ant

    def try_accept_solution_by_delta(self, delta):
        """
        Returns true if the solution for which the difference between its value
        and the current solution is delta should be accepted.
        A better solution (delta < 0) is always accepted,
        a worse one according to the Metropolis criterion,
        i.e. r < exp(-delta/T)
        """
        accept = False
        stats = self.stats
        if delta < 0.0:  # Always accept a better solution
            accept = True
        elif self.temperature > 0:  # Try to accept a worse solution
            prob = exp(-delta / self.temperature)
            if random() < prob:
                accept = True
                stats['worse_accepted'] += 1
            else:
                stats['worse_rejected'] += 1
        return accept
