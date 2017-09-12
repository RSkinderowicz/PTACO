#!/usr/bin/env python
# -*- encoding: utf-8 -*-

r"""PTACO

Author: Rafał Skinderowicz (rafal.skinderowicz [at] us.edu.pl)

Usage:
  ptaco.py --problem=<path> --alg=<algorithm>
           [--trials=<n>]
           [--iter_rel=<n>]
           [--iter_abs=<n>]
           [--out_dir=<path>]
           [--label=<str>]
           [--pheromone=<type>]
           [--ants=<n>]
           [--q0=<n>]
           [--beta=<n>]
           [--cand_list_size=<n>]
           [--global_evaporation_rate=<n>]
           [--local_evaporation_rate=<n>]
           [--iter_between_exchange=<n>]
           [--temperature_levels=<n>]
  ptaco.py (-h | --help)
  ptaco.py --version

Options:
  -h --help  Show this screen
  --version  Show version
  --problem=<path>  Path to a problem instance to solve
  --alg=<algorithm>  Name of the algorithm to run (ACS, EMMAS)
  --trials=<n>  Number of alg executions with the same params [default: 1].
  --iter_rel=<n>  (iter_rel x problem size) gives a total number of iterations
  --iter_abs=<n>  An absolute number of iterations to execute regardless of
                  a problem size
  --out_dir=<path>  Path to a directory where to store the experiment results.
  --label=<str>  An optional experiment "label" [default: NA]
  --pheromone=<type>  Type of pheromone memory to use [default: std]
  --ants=<n>  Number of ants [default: 10]
  --q0=<n>  q_0 param. of the ACS, if > 1 then q_0 = (n-q0)/n, (n - prob. size)
            [default: 10]
  --beta=<n>  \beta parameter of the ACO / ACS etc. algorithms [default: 2]
  --cand_list_size=<n>  Candidate lists size in the ACS [default: 25]
  --global_evaporation_rate=<n>  Global evap. rate in the ACS [default: 0.1]
  --local_evaporation_rate=<n>  Local evap. rate in the ACS [default: 0.01]
  --iter_between_exchange=<n>  Iterations between exchanges in the PTACO
                               [default: 500]
  --temperature_levels=<n>  How many parallel runs to use in PTACO (each with
                            its own temperature) [default: 4]
"""

from time import gmtime, strftime, time
import json
from os import path
import gc
from docopt import docopt

from tsp import TSP, load_tsplib_instance
from sop import SOP, load_sop_tsplib_instance
from acs import AntColonySystem
from acst import ACST
from ptaco import PTACO
from pheromone import PheromoneMatrix


def run_experiment(alg, iterations, trial_count):
    """
    Runs an experiment consisting of a 'trial_count' executions of the 'alg'
    algorithm.
    """
    stats = {}
    stats['trial_count'] = trial_count
    stats['datetime_begin'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    time_started = time()
    trials = stats['trials'] = []
    best_solution = None
    solution_values = []
    for i in range(trial_count):
        print('Starting trial {}...'.format(i+1))
        trial_start_time = time()
        solution = alg.run(iterations)
        trial_end_time = time()
        if best_solution is None or best_solution.value > solution.value:
            best_solution = solution
        solution_values.append(solution.value)
        trials.append(alg.stats)
        print('Trial {} best solution: {}'.format(i+1, solution.value))
        print('Trial elapsed: {}'.format(trial_end_time - trial_start_time))
        gc.collect()
    time_finished = time()
    stats['best_solution_value'] = best_solution.value
    stats['trials_solution_values'] = solution_values
    stats['best_solution'] = ' '.join(map(str, best_solution.visited))
    best_solutions_mean = sum(solution_values)/len(solution_values)
    stats['best_solution_mean'] = best_solutions_mean
    stats['datetime_end'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    stats['total_duration_sec'] = time_finished - time_started
    return best_solution, stats


def get_results_filename(arguments):
    """
    Returns a filename for the experiment results.
    The filename is generated based on program arguments as given in
    'arguments' and current date & time.
    """
    problem_path = arguments['--problem']
    problem_file = path.basename(problem_path)
    algorithm = arguments['--alg'].lower()
    curr_datetime = strftime("%Y-%m-%d_%H_%M_%S", gmtime())

    return '{}__{}__{}.js'.format(problem_file.replace('.', '_'),
                                  algorithm,
                                  curr_datetime)


def main():
    """
    Program's entry point
    """
    arguments = docopt(__doc__, version='PTACO 1.0')
    print(arguments)
    problem_path = arguments['--problem']
    file_ext = problem_path.lower().split('.')[-1]
    if file_ext in ('tsp', 'atsp'):
        instance = TSP(load_tsplib_instance(problem_path))
    elif file_ext == 'sop':
        instance = SOP(load_sop_tsplib_instance(problem_path))

    pheromone = arguments['--pheromone']
    pheromone_factory = None
    if pheromone == 'std':
        pheromone_factory = PheromoneMatrix

    aco_kwargs = {
        'ants': int(arguments['--ants']),
        'beta': float(arguments['--beta']),
        'pheromone_factory': pheromone_factory,
    }
    acs_kwargs = aco_kwargs.copy()
    acs_kwargs.update({
        'q0': float(arguments['--q0']),
        'cand_list_size': int(arguments['--cand_list_size']),
        'global_evaporation_rate': float(arguments['--global_evaporation_rate']),
        'local_evaporation_rate': float(arguments['--local_evaporation_rate']),
    })

    algorithm = arguments['--alg'].lower()
    if algorithm == 'acs':
        alg = AntColonySystem(problem=instance, **acs_kwargs)
    elif algorithm == 'ptaco':
        def aco_factory(problem, **args):
            "Creates an instance of the ACST"
            kwargs = acs_kwargs.copy()
            kwargs.update(args)
            return ACST(problem, **kwargs)
        iter_between_exchange = int(arguments['--iter_between_exchange'])
        temperature_levels = int(arguments['--temperature_levels'])
        alg = PTACO(problem=instance,
                    iter_between_exchange=iter_between_exchange,
                    temperature_levels=temperature_levels,
                    aco_factory=aco_factory)
    trials = int(arguments['--trials'])
    iterations = 0
    if '--iter_rel' in arguments:
        iterations = instance.dimension * int(arguments['--iter_rel'])
    elif '--iter_abs' in arguments:
        iterations = int(arguments['--iter_abs'])

    if iterations <= 0:
        raise RuntimeError('Wrong number of iterations, i.e. ' +
                           str(iterations))

    best_solution, exp_stats = run_experiment(alg, iterations=iterations,
                                              trial_count=trials)
    print('Best solution: %f' % best_solution.value)
    stats = {
        'arguments': arguments,
        'experiment': exp_stats,
        'label': arguments['--label'],
        'problem': {
            'dimension': instance.dimension,
            'is_symmetric': instance.is_symmetric,
            }
    }
    print(json.dumps(stats))
    out_dir = arguments.get('--out_dir', None)
    if out_dir:
        results_path = path.join(out_dir, get_results_filename(arguments))
        print('Saving results to {}'.format(results_path))
        with open(results_path, 'w') as file_:
            file_.write(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
