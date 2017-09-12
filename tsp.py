# -*- encoding: utf-8 -*-

"""
Traveling Salesman Problem related utilities.
"""

import re
from random import randint
from math import pi as M_PI
from math import cos, acos

from misc import array_double, array_bool


def geo_distance(x1, y1, x2, y2):
    """
    Compute geometric distance between two nodes rounded to next
    integer for TSPLIB instances. Based on the ACOTSP by Thomas Stuetzle
    """
    deg = int(x1)
    minute = x1 - deg
    lati = M_PI * (deg + 5.0 * minute / 3.0) / 180.0
    deg = int(x2)
    minute = x2 - deg
    latj = M_PI * (deg + 5.0 * minute / 3.0) / 180.0

    deg = int(y1)
    minute = y1 - deg
    longi = M_PI * (deg + 5.0 * minute / 3.0) / 180.0
    deg = int(y2)
    minute = y2 - deg
    longj = M_PI * (deg + 5.0 * minute / 3.0) / 180.0

    q1 = cos(longi - longj)
    q2 = cos(lati - latj)
    q3 = cos(lati + latj)
    dd = int(6378.388 * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
    return dd


def euclidean_distance(x1, y1, x2, y2):
    """
    Returns Euclidean distance between two points rounded to the nearest
    integer.
    """
    return int(((x2-x1)**2 + (y2-y1)**2)**0.5 + 0.5)


def calc_dist_matrix_euc_2d(coords, dim, distance_function):
    """
    Calculates a 2d matrix of Euclidean distances from list of coordinates.
    """
    initial_values = [-1.0 for i in range(dim)]
    matrix = [array_double(initial_values) for j in range(dim)]
    for i in range(dim):
        for j in range(dim):
            if i < j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dist = distance_function(x1, y1, x2, y2)
                matrix[i][j] = dist
            elif i > j:
                matrix[i][j] = matrix[j][i]
    print('First row sum:', sum(matrix[0]))
    return matrix


def load_tsplib_instance(file_path):
    """
    Loads a (A)TSP instance data from a file using TSPLIB format.
    The current version supports only EUC_2D, GEO and EXPLICIT format of edge
    weights.
    """
    coord_re = re.compile(r'\s*\d+\s[\d.e+-]+\s[\d.e+-]+\s*')
    header_re = re.compile(r'\s*([^:]+):\s*(.+)')
    coordinates = []
    desc = {}
    dist_matrix = None
    with open(file_path, 'r') as file_:
        lines = iter(file_.readlines())
        # Read header section
        for line_raw in lines:
            line = line_raw.strip().lower()
            if header_re.match(line):
                match = header_re.match(line)
                key, val = match.groups()
                desc[key.strip()] = val.strip()
            else:
                break
        # Now read edge wegiths section
        dimension = int(desc['dimension'])
        weights_section_header = line
        if weights_section_header == 'node_coord_section':
            for line in lines:
                if line == 'eof':
                    break
                if coord_re.match(line):
                    _, x, y = re.split(r'\s+', line.strip())
                    coordinates.append((float(x), float(y)))
        elif weights_section_header == 'edge_weight_section':
            if desc['edge_weight_format'] == 'full_matrix':
                glued = ' '.join(lines)
                tokens = re.split(r'\s+', glued.strip())
                if tokens[-1] == 'EOF':
                    del tokens[-1]
                weights = map(float, tokens)
                dist_matrix = [weights[i:i+dimension]
                               for i in range(0, dimension**2, dimension)]
            else:
                raise RuntimeError('Cannot read edge weight section')
        else:
            raise RuntimeError('Cannot read edge weight section')

    desc['coordinates'] = coordinates
    desc['dimension'] = dimension

    distance_function = None
    edge_weight_type = desc['edge_weight_type']
    if edge_weight_type == 'euc_2d':
        distance_function = euclidean_distance
    elif edge_weight_type == 'geo':
        distance_function = geo_distance
    elif edge_weight_type != 'explicit':
        print("Unknown edge weight type")
        return None
    if not dist_matrix:
        dist_matrix = calc_dist_matrix_euc_2d(coordinates, dimension,
                                              distance_function)
    desc['dist_matrix'] = dist_matrix
    desc['is_symmetric'] = (desc['type'] == 'tsp')
    return desc


class Ant(object):
    """
    Ant represents a solution to a TSP problem.
    """
    def __init__(self, node_count):
        self.node_count = node_count
        self.visited = []
        self.unvisited_mask = array_bool(True for _ in range(node_count))
        self.value = 0.0

    def is_visited_node(self, node):
        """ Returns True if node was already visited. """
        return not self.unvisited_mask[node]

    def add_visited_node(self, node):
        """
        Appends a node to the current (partial) solution.
        """
        assert not self.is_visited_node(node)
        self.visited.append(node)
        self.unvisited_mask[node] = False

    def goto_initial_node(self):
        self.add_visited_node(randint(0, self.node_count-1))


class TSP(object):
    """
    Holds TSP data.
    """
    def __init__(self, instance_data, is_symmetric=True):
        self.dimension = instance_data['dimension']
        self.dist_matrix = instance_data['dist_matrix']
        self.is_symmetric = instance_data['is_symmetric']
        print('is_symmetric', self.is_symmetric)

    def evaluate_solution(self, route):
        """
        Returns a value of a solution - it is equal to the length of the route.
        """
        dist_matrix = self.dist_matrix
        prev = route[-1]
        total = 0.0
        for node in route:
            total += dist_matrix[prev][node]
            prev = node
        return total

    def create_ant(self):
        """
        Returns a new ant representing an empty solution to the problem
        described by this instance.
        """
        return Ant(self.dimension)

    def build_greedy_solution(self):
        """
        Returns a greedily built solution.
        It starts from node 0 and then goes to a closest available node, and so
        forth.
        """
        ant = self.create_ant()
        start_node = 0
        remaining_nodes = range(1, self.dimension)
        ant.add_visited_node(start_node)
        prev = start_node
        while remaining_nodes:
            distances = self.dist_matrix[prev]
            closest = min(remaining_nodes, key=lambda node: distances[node])
            ant.add_visited_node(closest)
            remaining_nodes.remove(closest)
            prev = closest
        return ant
