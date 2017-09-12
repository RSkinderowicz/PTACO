# -*- encoding: utf-8 -*-

"""
Implementation of the Ant Colony System by Dorigo et. al
"""


from random import random
from collections import defaultdict
from misc import array_double, array_int

from pheromone import PheromoneMatrix


class AntColonySystem(object):
    """
    An impl. of the Ant Colony System.
    """
    def __init__(self, problem,
                 q0=10, beta=2.0, ants=10,
                 global_evaporation_rate=0.1,
                 local_evaporation_rate=0.01,
                 cand_list_size=25,
                 pheromone_factory=PheromoneMatrix):
        self.problem = problem
        self.ants_count = ants
        self.set_q0(q0)
        self.beta = beta
        self.global_evaporation_rate = global_evaporation_rate
        self.local_evaporation_rate = local_evaporation_rate
        self.cand_list_size = cand_list_size
        self._init_cand_lists()
        self.initial_pheromone = None
        self.heuristic = None
        self.cached_product = None
        self._init_heuristic()
        self.stats = None
        self.pheromone = None
        self.pheromone_factory = pheromone_factory
        self.iteration = None
        self.iteration_best = None
        self.global_best = None
        self.ants = []

    def _init_heuristic(self):
        """
        Initializes a matrix of heuristic values for every edge of the problem
        representation graph. The current impl. is intended for TSP-like
        problems.
        """
        def calc(val):
            " This is valid for the TSP-like problems "
            return 1.0 / val ** self.beta if val > 0 else 0.0
        self.heuristic = [array_double(calc(d) for d in row)
                          for row in self.problem.dist_matrix]

    def _calc_initial_pheromone(self):
        greedy_sol = self.problem.build_greedy_solution()
        greedy_sol_value = self.problem.evaluate_solution(greedy_sol.visited)
        return 1./(self.problem.dimension * greedy_sol_value)

    def _init_pheromone(self):
        if not self.initial_pheromone:
            self.initial_pheromone = self._calc_initial_pheromone()
        self.pheromone = self.pheromone_factory(self.problem.dimension,
                                                self.initial_pheromone)
        dim = self.problem.dimension
        product = [array_double(0 for _ in range(dim)) for _ in range(dim)]
        for i, heuristic_i in enumerate(self.heuristic):
            for j, heur in enumerate(heuristic_i):
                product[i][j] = self.pheromone.get_trail(i, j) * heur
        self.cached_product = product
        self.stats['initial_pheromone'] = self.initial_pheromone

    def _init_cand_lists(self):
        """
        For each node creates a list of its closest neighbours.  This lists are
        used to focus the solution construction process on the attractive
        edges, i.e. short in the case of the TSP.
        """
        size = self.cand_list_size
        nn_lists = []  # List of closest neighbours
        dim = self.problem.dimension
        size = min(size, dim-1)
        for node in range(dim):
            neighbors = range(dim)
            neighbors.remove(node)  # City is not a neighbour of itself
            dist = self.problem.dist_matrix[node]
            neighbors.sort(key=lambda neighbor: dist[neighbor])
            closest = array_int(neighbors[:size])
            nn_lists.append(closest)
        self.cand_lists = nn_lists

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
            self.global_pheromone_update(self.global_best)
            self.iteration += 1
        return self.global_best

    def _build_ant_solutions(self):
        """
        Initializes ants (self.ants) and builds their complete solutions to the
        problem. Solutions are also evaluated.
        """
        self.ants = [self.problem.create_ant() for _ in range(self.ants_count)]
        for ant in self.ants:
            # Places an ant at the first node (possibly random)
            ant.goto_initial_node()
        for _ in range(1, self.problem.dimension):
            for ant in self.ants:
                self._move_ant_q0_optimized(ant)
                self.local_pheromone_update(ant.visited[-2],
                                            ant.visited[-1])
        for ant in self.ants:
            self.local_pheromone_update(ant.visited[-1],
                                        ant.visited[0])
        for ant in self.ants:
            assert not any(ant.unvisited_mask)
            ant.value = self.problem.evaluate_solution(ant.visited)

    def _update_iteration_best(self):
        self.iteration_best = min(self.ants, key=lambda ant: ant.value)

    def _update_global_best(self):
        if not self.global_best or\
                (self.global_best.value > self.iteration_best.value):
            self.global_best = self.iteration_best
            self.stats['global_best_value'].\
                append((self.iteration, self.global_best.value))

    def run_init(self):
        """
        This is called by 'run' method at the beginning of the ACS execution.
        """
        self.stats = defaultdict(list)
        self._init_pheromone()
        self.global_best = None
        self.iteration_best = None
        self.ants = []
        self.iteration = 0

    def _move_ant(self, ant):
        """
        Moves ant to a next node selected using the pseudo-random proportional
        rule.
        """
        curr_node = ant.visited[-1]
        product = self.cached_product[curr_node]
        unvisited_mask = ant.unvisited_mask
        cand_list = self.cand_lists[curr_node]
        candidates = [node for node in cand_list if unvisited_mask[node]]
        chosen = -1
        if len(candidates) > 0:  # At least one el. of cand_list is unvisited
            if random() < self.q0:  # Greedy choice
                chosen = candidates[0]
                max_product = product[chosen]
                for node in candidates:
                    if product[node] > max_product:
                        max_product = product[node]
                        chosen = node
            else:  # A pseudo-random proportional choice
                total = 0.0
                for node in candidates:
                    total += product[node]
                threshold = random() * total
                partial_sum = 0.0
                chosen = candidates[-1]
                for node in candidates:
                    partial_sum += product[node]
                    if partial_sum >= threshold:
                        chosen = node
                        break
        else:  # All of the node's nearest neighbors have been already visited
            max_product = -1.0
            for node, unvisited in enumerate(unvisited_mask):
                if unvisited:
                    if product[node] > max_product:
                        chosen = node
                        max_product = product[node]
        assert chosen != -1 and unvisited_mask[chosen]
        ant.add_visited_node(chosen)

    def _move_ant_q0_optimized(self, ant):
        """
        This is equivalent to _move_ant but optimized for speed with pypy in
        mind.
        The hot-path, i.e. random() < q_0 is calculated always (up-front) even
        if not necessary.
        """
        curr_node = ant.visited[-1]
        product = self.cached_product[curr_node]
        unvisited_mask = ant.unvisited_mask
        # We expect q0 to be close to 1 so we can calculate up front the node
        # with maximum product of pheromone and heuristic info (hot-path)
        chosen = -1
        max_product = 0.0
        for node in self.cand_lists[curr_node]:
            if unvisited_mask[node]:
                if product[node] > max_product:
                    max_product = product[node]
                    chosen = node
        if chosen != -1:
            if random() >= self.q0:
                # Psuedo-random proportional selection
                candidates = [n for n in self.cand_lists[curr_node]
                              if unvisited_mask[n]]
                total = 0.0
                for node in candidates:
                    total += product[node]
                threshold = random() * total
                partial_sum = 0.0
                chosen = candidates[-1]
                for node in candidates:
                    partial_sum += product[node]
                    if partial_sum >= threshold:
                        chosen = node
                        break
        else:  # All of the node's nearest neighbors have been already visited
            max_product = -1.0
            for node, unvisited in enumerate(unvisited_mask):
                if unvisited:
                    if product[node] > max_product:
                        chosen = node
                        max_product = product[node]
        assert chosen != -1 and unvisited_mask[chosen]
        ant.add_visited_node(chosen)

    def local_pheromone_update(self, beg, end):
        """
        Local pheromone update diminishes pheromone level on a given edge with
        the aim of making it less frequently selected by subsequent ants.
        """
        self.pheromone.update(beg, end,
                              self.local_evaporation_rate,
                              self.initial_pheromone,
                              symmetric=self.problem.is_symmetric)
        self._update_cached_product(beg, end)

    def global_pheromone_update(self, ant):
        """
        Global pheromone update aims at reinforcing pheromone on the edges
        beloning to a given solution (ant). Typically it is the best so far
        solution (ant).
        """
        deposit = 1.0/ant.value
        nodes = ant.visited
        evap_rate = self.global_evaporation_rate
        prev = nodes[-1]
        for node in nodes:
            self.pheromone.update(prev, node, evap_rate, deposit,
                                  symmetric=self.problem.is_symmetric)
            self._update_cached_product(prev, node)
            prev = node

    def _update_cached_product(self, beg, end):
        """
        This updates the cache of the pheromone and heuristic data product.
        It is used when calculating the next node for an ant to move.
        """
        updated = self.pheromone.get_trail(beg, end) * self.heuristic[beg][end]
        self.cached_product[beg][end] = updated
        if self.problem.is_symmetric:
            self.cached_product[end][beg] = updated

    def set_q0(self, value):
        """
        Sets the value of q0 parameter. If 'value' > 1 then
        q0 = (n - value) / n, where n is the size of the
        problem.
        """
        if value < 1.0:
            self.q0 = value
        else:
            dim = self.problem.dimension
            self.q0 = max(0.0, (dim - value) / float(dim))
