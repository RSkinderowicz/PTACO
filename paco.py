# -*- encoding: utf-8 -*-

"""
Implementation of the Population-based ACO
"""


from random import randint, random, choice
from collections import defaultdict
from misc import array_double, array_int

from pheromone import PheromoneMatrix, SelectivePheromoneMemory
from acs import AntColonySystem
from sa import SimulatedAnnealing


class PACOPheromoneMemory(object):
    """
    Impl. of the pheromone memory for the Population-based ACO.
    The memory contains a number of complete solutions which are used to
    compute the value of a pheromone trail for each edge.
    """
    def __init__(self, dimension, trail_min, capacity=8):
        self.trail_min = trail_min
        self.trail_max = 1.0
        self.solutions = []  # Solutions in form of sequences of visited nodes
        self.solution_values = []
        self.capacity = capacity  # Max number of solutions to store
        self.buckets = [[] for _ in range(dimension)]

    def get(self, beg, end):
        """
        This returns a trail value for edge (beg, end)
        """
        n = self.buckets[beg].count(end)
        delta = (self.trail_max - self.trail_min) / self.capacity
        return self.trail_min + delta * n

    def add_solution(self, visited, value):
        """
        Sets all pheromone values from the given vector.
        """
        # Is there a place for the new solution
        solutions = self.solutions
        if len(solutions) > self.capacity:
            self.remove_solution(0)
        self.solutions.append(visited)
        self.solution_values.append(value)
        buckets = self.buckets
        beg = visited[-1]
        is_symmetric = True
        for end in visited:
            buckets[beg].append(end)
            if is_symmetric:
                buckets[end].append(beg)
            beg = end

    def remove_solution(self, index):
        visited = self.solutions[index]
        del self.solutions[index]
        del self.solution_values[index]
        beg = visited[-1]
        is_symmetric = True
        buckets = self.buckets
        for end in visited:
            buckets[beg].remove(end)
            if is_symmetric:
                buckets[end].remove(beg)
            beg = end

    def print_debug_info(self):
        pass


class PACO(AntColonySystem):
    """
    Implementation of the Population-based ACO
    """
    def __init__(self, problem):
        AntColonySystem.__init__(self, problem)
        dim = self.problem.dimension
        self.q0 = max(0.0, (dim - 20.0) / dim)

    def _init_pheromone(self):
        dim = self.problem.dimension
        trail_min = 1.0 / (dim - 1)
        self.pheromone = PACOPheromoneMemory(dim, trail_min, capacity=8)
        product = [array_double(trail_min*h for h in heur)
                   for heur in self.heuristic]
        self.cached_product = product
        self.initial_pheromone = trail_min
        self.stats['initial_pheromone'] = self.initial_pheromone

    def run(self, iterations):
        """
        Runs the algorithm for a given number of iterations.
        Returns the global best ant, i.e. the best solution found.
        """
        stats = self.stats = defaultdict(list)
        self._init_pheromone()
        global_best = None
        dim = self.problem.dimension
        for it in range(iterations):
            ants = [self.problem.create_ant() for _ in range(self.ants_count)]
            for ant in ants:
                # Places an ant at the first node (possibly random)
                ant.goto_initial_node()
            for _ in range(1, dim):
                for ant in ants:
                    self._move_ant_q0_optimized(ant)
            for ant in ants:
                assert not any(ant.unvisited_mask)
                ant.value = self.problem.evaluate_solution(ant.visited)
            iteration_best = min(ants, key=lambda ant: ant.value)
            if not global_best or (global_best.value > iteration_best.value):
                global_best = iteration_best
                stats['global_best_value'].append((it, global_best.value))
            self.global_pheromone_update(iteration_best)
        self.pheromone.print_debug_info()
        return global_best

    def local_pheromone_update(self, beg, end):
        """
        Local pheromone update diminishes pheromone level on a given edge with
        the aim of making it less frequently selected by subsequent ants.
        """
        pass

    def global_pheromone_update(self, ant):
        """
        Global pheromone update aims at reinforcing pheromone on the edges
        beloning to a given solution (ant). Typically it is the best so far
        solution (ant).
        """
        self.pheromone.add_solution(ant.visited, ant.value)
        beg = ant.visited[-1]
        for end in ant.visited:
            self._update_cached_product(beg, end)
            beg = end

    def _update_cached_product(self, beg, end):
        pheromone = self.pheromone
        heuristic = self.heuristic
        updated = pheromone.get(beg, end) * heuristic[beg][end]
        self.cached_product[beg][end] = updated
        if self.problem.is_symmetric:
            self.cached_product[end][beg] = updated


class PACO_SA(AntColonySystem):
    """
    An experimental merge of the Population-based ACO with the SA
    """
    def __init__(self, problem,
                 initial_temperature,
                 **kwargs):
        AntColonySystem.__init__(self, problem, **kwargs)
        self.sa = None
        self.initial_temperature = initial_temperature

    def _init_pheromone(self):
        dim = self.problem.dimension
        trail_min = 1.0 / (dim - 1)
        self.pheromone = PACOPheromoneMemory(dim, trail_min, capacity=8)
        product = [array_double(trail_min*h for h in heur)
                   for heur in self.heuristic]
        self.cached_product = product
        self.initial_pheromone = trail_min
        self.stats['initial_pheromone'] = self.initial_pheromone

    def _init_temperature(self):
        """
        This is responsible for the calculation of an initial SA temperature
        value based on a sample of randomly generated instances.
        """
        self.sa = SimulatedAnnealing(cooling_ratio=0.99)
        self.set_temperature(self.initial_temperature)
        self.stats['initial_temperature'] = self.initial_temperature
        self.stats['temperatures'] = [(self.initial_temperature,
                                       self.iteration)]

    def set_temperature(self, t):
        """
        This overrides current SA temperature.
        """
        self.sa.temperature = t
        self.stats['temperatures'].append((t, self.iteration))

    def run_init(self):
        AntColonySystem.run_init(self)
        self._init_temperature()
        self.active_solution = None
        self.global_best = None
        self.iteration_best = None
        self.ants = []

    def run(self, iterations, run_from_start=True):
        """
        Runs the algorithm for a given number of iterations.
        Returns the global best ant, i.e. the best solution found.
        """
        if run_from_start:
            self.run_init()
        stats = self.stats
        dim = self.problem.dimension
        phmem = self.pheromone
        for it in range(iterations*self.ants_count):
            ant = self.problem.create_ant()
            self.ants = [ant]
            # Places an ant at the first node (possibly random)
            ant.goto_initial_node()
            for _ in range(1, dim):
                self._move_ant_q0_optimized(ant)

            ant.value = self.problem.evaluate_solution(ant.visited)
            self.iteration_best = ant

            if not self.global_best or\
                    (self.global_best.value > self.iteration_best.value):
                self.global_best = self.iteration_best
                stats['global_best_value'].append((self.iteration,
                                                   self.global_best.value))
                print('New global_best:', self.global_best.value)

            k = len(phmem.solutions)
            if k == 0:
                self.global_pheromone_update(self.global_best)
            else:
                index = randint(0, k-1)
                delta = ant.value - phmem.solution_values[index]

                if self.sa.try_accept_solution_by_delta(delta):
                    if k == phmem.capacity:
                        phmem.remove_solution(index)

                    self.global_pheromone_update(ant)
            self.iteration += 1
        return self.global_best

    def local_pheromone_update(self, beg, end):
        """
        Local pheromone update diminishes pheromone level on a given edge with
        the aim of making it less frequently selected by subsequent ants.
        """
        pass

    def global_pheromone_update(self, ant):
        """
        Global pheromone update aims at reinforcing pheromone on the edges
        beloning to a given solution (ant). Typically it is the best so far
        solution (ant).
        """
        self.pheromone.add_solution(ant.visited, ant.value)
        beg = ant.visited[-1]
        for end in ant.visited:
            self._update_cached_product(beg, end)
            beg = end

    def _update_cached_product(self, beg, end):
        pheromone = self.pheromone
        heuristic = self.heuristic
        updated = pheromone.get(beg, end) * heuristic[beg][end]
        self.cached_product[beg][end] = updated
        if self.problem.is_symmetric:
            self.cached_product[end][beg] = updated
