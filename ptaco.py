# -*- encoding: utf-8 -*-

# Author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)

r"""
This is an implementation of the Parallel tempering based ACS.  The basic idea
is to run a number of ACST (ACS with temperature) replicas (instances) in
parallel. Each of the ACST replicas has a temperature parameter which is used
to update its active_solution. The active_solution is used in the global
pheromone update rule.

The selection of the active_solution in the ACST is based on
Metropolis-Hastings rule which in turn uses a temperature parameter and the
difference (delta) between the cost (quality) of the proposed solution and the
current active_solution.

In the PTACO each of the replicas has a different temperature level ordered
from the smallest.
The replicas with the adjacent temperature levels are allowed to exchange state
or alternatively swap temperature values. This exchange affects the search
trajectory of the ACST allowing the low-temperature replica to escape a local
minimum, while the high-temperature replica can perform a local search in the
region of the solution search space containing the current active solution. 

The exchanges are performed periodically based on a parameter.
"""

from math import exp, log
from random import random, randint
from collections import defaultdict
from copy import copy

from misc import mean


class PTACO(object):
    """
    Parallel tempering inspired version of the ACO in which we run multiple
    instances at different "temperatures" and allow them to exchange states at
    certain times.
    """
    def __init__(self, problem, aco_factory,
                 temperature_levels=4,
                 iter_between_exchange=500):
        """
        aco_factory - a callable which returns a new instance of the
                      AntColonySystemSA (or compatible).
        temperature_levels - how many parallel runs to use, each with its own
                             temperature level
        iter_between_exchange - a number of iterations between solution
                                exchange attempts between the parallel runs;
                                if 0 then no exchange is made
        """
        self.problem = problem
        self.aco_factory = aco_factory
        self.temperature_levels = temperature_levels
        self.iter_between_exchange = iter_between_exchange

    def calc_temperatures(self, temperature_levels,
                          min_prob=0.001, max_prob=0.5):
        """
        Calculates a list of increasing temperature values so that
        at the lowest temperature the probability of accepting a solution
        worse by 1% is 'min_prob' and the max. temp. the probability is
        'max_prob'. The temperatures between are set so that the probability
        doubles with each level.
        """
        n = 2**(temperature_levels - 1) - 1
        span = max_prob - min_prob
        gap = span / n
        return [-0.01 / log(min_prob + gap * (2**i - 1))
                for i in range(temperature_levels)]

    def run(self, iterations):
        """
        Runs parallel tempering inspired version of the ACO. Returns best
        solution found. Statistics are available through self.stats.

        iterations - a number of iterations that the replicas should
                     perform in total
        """
        stats = self.stats = defaultdict(int)
        temperature_levels = self.temperature_levels
        iter_between_exchange = self.iter_between_exchange
        exchange_enabled = iter_between_exchange > 0

        initial_temperatures = self.calc_temperatures(temperature_levels)

        iter_per_replica = iterations / temperature_levels

        replicas = []
        for i, t in enumerate(initial_temperatures):
            replica = self.aco_factory(self.problem, initial_temperature=t)
            replica.run_init()
            replica.temperature_history = [(i, 0)]
            replicas.append(replica)

        best_solution_value = None
        # These are for logging purposes only
        energy_deltas = []
        exchange_successes_per_level = [0 for i in range(temperature_levels-1)]
        exchange_attempts_per_level = [0 for i in range(temperature_levels-1)]

        temperatures = copy(initial_temperatures)

        budget = iterations
        epoch = 1  # Epoch ends with an exchange between a pair of replicas
        while budget > 0:  # Main loop
            # Advance all replicas
            it = iter_between_exchange if exchange_enabled\
                                       else iter_per_replica
            for replica in replicas:
                replica.run(max(1, min(budget, it)), run_from_start=False)
                if best_solution_value is None or\
                        replica.global_best.value < best_solution_value:
                    best_solution_value = replica.global_best.value
                budget = max(0, budget - it)

            # Calculate exchange probabilities between the replicas (runs) We
            # do not have to calculate all of them but we do for the raporting
            # purposes
            exchange_probs = [0. for _ in range(0, temperature_levels-1)]
            for i in range(temperature_levels-1):
                delta_beta = 1./temperatures[i] - 1./temperatures[i+1]
                # We use the value of the current active_solution as a kind of
                # "energy" measure
                energy_1 = replicas[i].active_solution.value
                energy_2 = replicas[i+1].active_solution.value
                energy_deltas.append(energy_1-energy_2)  # For raporting
                # Delta E is normalized resp. to the current global best sol.
                delta_E = (energy_1 - energy_2) / best_solution_value
                exchange_probs[i] = min(1.0, exp(delta_beta * delta_E))

            # Now we can actually do an exchange between a pair of replicas at
            # different levels
            i = randint(0, temperature_levels-2)
            exchange_attempts_per_level[i] += 1
            if exchange_enabled and random() < exchange_probs[i]:
                fst, snd = replicas[i], replicas[i+1]
                fst.temperature, snd.temperature =\
                    snd.temperature, fst.temperature
                replicas[i].temperature_history.append((i+1, epoch))
                replicas[i+1].temperature_history.append((i, epoch))
                replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                exchange_successes_per_level[i] += 1
            epoch += 1

        solutions = [r.global_best for r in replicas]
        print('Final solutions %s' % [s.value for s in solutions])
        print('Mean energy diff: {}, mean. exch. prob.: {}'.format(
              mean(energy_deltas), mean(exchange_probs)))
        print('Exchanges: {}'.format(zip(exchange_successes_per_level,
                                         exchange_attempts_per_level)))
        print('Temperature histories:')
        runs_stats = stats['runs'] = []
        for i, replica in enumerate(replicas):
            print('Temp. history for replica {}: {}'.format(i,
                  replica.temperature_history))
            print('Iteration: ', replica.iteration)
            runs_stats.append(replica.stats)

        best_solution = min(solutions, key=lambda sol: sol.value)

        # Record some statistics / configuration
        stats.update({
            'initial_temperatures': initial_temperatures,
            'temperature_levels': temperature_levels,
            'iter_per_replica': iter_per_replica,
            'iter_between_exchange': iter_between_exchange,
            'total_iterations': iterations,
            'total_epochs': epoch,
            'final_temperatures': temperatures,
            'best_solution_value': best_solution.value,
            'exchange_enabled': exchange_enabled,
            'exchange_energy_delta_mean': mean(energy_deltas),
            'exchange_prob_mean': mean(exchange_probs),
            'exchange_attempts': exchange_attempts_per_level,
            'exchange_successes': exchange_successes_per_level,
            'temperature_histories': [r.temperature_history for r in replicas]
            })
        return best_solution
