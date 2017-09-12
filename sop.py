# -*- encoding: utf-8 -*-

"""
Sequential Ordering Problem related utilities.
"""

import re
from misc import array_int, array_bool


def load_sop_tsplib_instance(file_path):

    """
    Tries to load the SOP instance from a given file.

    Header has the following format:

    NAME: ESC78.sop
    TYPE: SOP
    COMMENT: Received by Norbert Ascheuer / Laureano Escudero
    DIMENSION: 80
    EDGE_WEIGHT_TYPE: EXPLICIT
    EDGE_WEIGHT_FORMAT: FULL_MATRIX
    EDGE_WEIGHT_SECTION
    80
    ...
    """
    with open(file_path, 'r') as file_:
        raw_lines = file_.readlines()

    dimension = -1
    instance = {}
    weights = []
    in_edge_weight_section = False
    for line in raw_lines:
        if 'EOF' in line:
            break
        elif in_edge_weight_section:
            tokens = re.split(r'\s+', line.strip())
            if len(tokens) == dimension:
                weights.append(array_int(map(int, tokens)))
        elif 'DIMENSION' in line:
            dimension = int(line.split(':')[1])
        elif 'EDGE_WEIGHT_TYPE' in line:
            weight_type = line.split(':')[1].strip()
            if weight_type != 'EXPLICIT':
                print('Unknown edge weight type: %s' % weight_type)
                return None
        elif 'EDGE_WEIGHT_FORMAT' in line:
            weight_format = line.split(':')[1].strip()
            if weight_format != 'FULL_MATRIX':
                print('Unknown edge weight format: %s' % weight_format)
                return None
        elif 'EDGE_WEIGHT_SECTION' in line:
            in_edge_weight_section = True
    assert len(weights) == dimension
    instance['dimension'] = dimension
    instance['edge_weight_matrix'] = weights
    return instance


class Ant(object):
    """
    Ant represents a solution to a SOP problem.
    """
    def __init__(self, problem):
        self.problem = problem
        self.node_count = problem.dimension
        self.visited = []
        all_nodes = range(self.node_count)
        self.preceding_count = array_int(problem.get_preceding_count(node)
                                         for node in all_nodes)
        self.visited_mask = array_bool(False for _ in all_nodes)
        self.unvisited_mask = array_bool(self.preceding_count[node] == 0
                                         for node in all_nodes)
        self.value = 0.0

    def is_visited_node(self, node):
        """ Returns True if node was already visited. """
        return self.visited_mask[node]

    def goto_initial_node(self):
        self.add_visited_node(0)

    def add_visited_node(self, node):
        """
        Appends a node to the current (partial) solution.
        """
        assert self.unvisited_mask[node]
        self.visited.append(node)
        self.visited_mask[node] = True
        self.unvisited_mask[node] = False
        for succ in self.problem.following[node]:
            self.preceding_count[succ] -= 1
            # If the preceding_count[succ] falls to 0 then all the nodes that
            # have to preceed succ are already a part of the solution and now
            # the succ may be visited
            if self.preceding_count[succ] == 0 and\
                    not self.visited_mask[succ]:
                self.unvisited_mask[succ] = True


class SOP(object):
    """
    Holds Sequential Ordering Problem data.
    """
    def __init__(self, instance):
        self.dimension = instance['dimension']
        self.dist_matrix = instance['edge_weight_matrix']
        self.is_symmetric = False
        self.preceding = None
        self.following = None
        self._process_constraints()

    def _process_constraints(self):
        dimension = self.dimension
        dist_matrix = self.dist_matrix
        preceding = self.preceding = [[] for _ in range(dimension)]
        following = self.following = [[] for _ in range(dimension)]
        for fst, weights in enumerate(dist_matrix):
            for snd, weight in enumerate(weights):
                if weight < 0:  # snd should be before fst
                    following[snd].append(fst)
                    preceding[fst].append(snd)

    def get_preceding_count(self, node):
        """
        Returns a number of node that preceed a specified node in a valid
        solution.
        """
        return len(self.preceding[node])

    def evaluate_solution(self, route):
        """
        Returns a value of a solution - it is equal to the length of the route.
        """
        dist_matrix = self.dist_matrix
        total = 0.0
        size = len(route)
        prev = route[0]
        i = 1
        while i < size:
            curr = route[i]
            total += dist_matrix[prev][curr]
            prev = curr
            i += 1
        return total

    def is_solution_valid(self, sol):
        """
        Checks if a solution is valid, i.e. all nodes were visited
        and all precedence constrains are satisfied.
        """
        dim = self.dimension
        dist_matrix = self.dist_matrix
        if (len(sol) != dim) or (set(sol) != set(range(dim))):
            return False
        # Now check if all the precedence constraints are satisfied
        for i, pred in enumerate(sol):
            for succ in sol[i+1:]:
                if dist_matrix[pred][succ] < 0:
                    # succ should be earlier than pred in the solution
                    return False
        return True

    def create_ant(self):
        """
        Returns a new ant representing an empty solution to the problem
        described by this instance.
        """
        return Ant(self)

    def build_greedy_solution(self):
        """
        Returns a greedily built solution.
        It starts from node 0 and then goes to a closest available node, and so
        forth.
        """
        ant = self.create_ant()
        ant.add_visited_node(0)  # 0 is the starting node or source
        remaining = set(range(1, self.dimension))
        while remaining:
            closest = None
            min_distance = -1.0
            distances = self.dist_matrix[ant.visited[-1]]
            for node in remaining:
                if ant.unvisited_mask[node]:
                    if closest is None or distances[node] < min_distance:
                        min_distance = distances[node]
                        closest = node
            assert closest is not None
            ant.add_visited_node(closest)
            remaining.remove(closest)
        return ant
