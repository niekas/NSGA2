#! /home/albertas/how_to_use/env/bin/python2.7

#    This file is a modified part of DEAP package.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array
import random
import json
import sys
import os
import subprocess
from optparse import OptionParser

import numpy

from math import sqrt

from deap import algorithms
from deap import base
# from deap import benchmarks
import problems
from deap.benchmarks.tools import diversity, convergence
from tools import uniformity
from deap import creator
from deap import tools

try:
    # try importing the C version
    from _hypervolume import _hv
except ImportError:
    # fallback on python version
    from _hypervolume import pyhv as _hv
hypervolume = _hv.hypervolume


parser = OptionParser()
parser.add_option("--func_name", dest="func_name")
parser.add_option("--max_calls", dest="max_calls")
parser.add_option("--d", dest="d")
parser.add_option("--seed", dest="seed")
parser.add_option("--max_duration", dest="max_duration")
parser.add_option("--task_id", dest="task_id")
parser.add_option("--callback", dest="callback")
(options, args) = parser.parse_args()

max_calls = int(options.max_calls)
func_name = options.func_name
d = int(options.d)
seed = int(options.seed)
max_duration = options.max_duration
task_id = options.task_id
callback = options.callback

problem = problems.get_problem(options.func_name)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,)*problem.crits)
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# The problem with its parameters is set only here.
# Problem definition
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
## BOUND_LOW, BOUND_UP = 0.0, 1.0

# Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
# BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
# NDIM = 6 # 30

nadir = problem.nadir
NDIM = problem.dimension
BOUND_LOW, BOUND_UP = problem.bound_low, problem.bound_up


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]



toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", problem)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

def nsga2(max_calls, func_name, d, seed=None):
    random.seed(seed)

    MU = 20
    NGEN = max_calls/ MU # 750  # 250    # Number of evaluations = NGEN * MU
    CXPB = 0.9

    file_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    stats_file = open(file_path + '/log/stats_%s_%d__nsga2_%s.txt' % (func_name, d, str(seed)), 'w')
    front_file = open(file_path + '/log/front_%s_%d__nsga2_%s.txt' % (func_name, d, str(seed)), 'w')

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        # print(ind, fit)
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    # print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        # print(logbook.stream)
        hv = hypervolume(numpy.array(problem.pareto_front), nadir)
        uni = uniformity(problem.pareto_front)
        calls = problem.evals
        stats_file.write('%d %f %f\n' % (calls, hv, uni))

    for p in problem.pareto_front:
        front_file.write(str(p) +'\n')

    stats_file.close()
    front_file.close()

    subprocess.call([callback,
        '--calls=%d' % problem.evals,
        '--hyper_volume=%f' % hv,
        '--uniformity=%f' % uni,
        '--task_id=%d' % task_id,
        '--status=D',
        '-exe=%s' %  sys.argv[0],
    ])
    return pop, logbook


if __name__ == '__main__':
    pop, stats = nsga2(max_calls, func_name, d, seed=seed)
