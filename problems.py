#! /home/albertas/how_to_use/env/bin/python2.7
#    This file is part of DEAP.
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
"""
Regroup typical EC benchmarks functions to import easily and benchmark
examples.
"""

import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce

# Note: algorithm complexity reduction:
# Is tol value nearer the zero with new pareto_front points? If yes - update the tol.

def domination(q, p):
    # // 0 - q dominates p
    # // 1 - none of them dominates
    # // 2 - p dominates q
    q_better = 0
    p_better = 0
    for i in range(len(q)):
        if q[i] < p[i]:
            q_better += 1
        if q[i] > p[i]:
            p_better += 1
    if (q_better == len(q)):
        return 0
    if (p_better == len(p)):
        return 2
    return 1


def update_pareto_front(p, pareto_front):
    # if any(q > p for q in S):
    #     return
    # for q in [q for q in S if p > q]:
    #     S.remove(q)
    #     S.add(p)
    dominated = []
    for q in pareto_front:
        relation = domination(q, p)
        if (relation == 0):
            return False
        if (relation == 2):
            dominated.append(q)

    for q in dominated:
        pareto_front.remove(q)

    pareto_front.append(p)
    return True


def evals_dec(func):
    '''Decorator for objective functions, which calculates unique function evaluations.'''
    # This method should also track pareto front
    # Hypervolume and uniformity have to be found using the actual pareto front.
    def f(individual, *args, **kwargs):
        vals = func(individual, *args, **kwargs)
        # if individual not in f.evals_at:
            # f.evals_at.append(individual[:])
            # f.obj_vals.append(vals)
        f.evals += 1
        update_pareto_front(vals, f.pareto_front)
        return vals
    f.evals = 0
    # f.evals_at = []
    # f.obj_vals = []
    f.pareto_front = []
    return f


# Unimodal
def rand(individual):
    """Random test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization or maximization
       * - Range
         - none
       * - Global optima
         - none
       * - Function
         - :math:`f(\mathbf{x}) = \\text{\\texttt{random}}(0,1)`
    """
    return random.random(),

def plane(individual):
    """Plane test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - none
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\mathbf{x}) = x_0`
    """
    return individual[0],

def sphere(individual):
    """Sphere test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - none
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\mathbf{x}) = \sum_{i=1}^Nx_i^2`
    """
    return sum(gene * gene for gene in individual),

def cigar(individual):
    """Cigar test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - none
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\mathbf{x}) = x_0^2 + 10^6\\sum_{i=1}^N\,x_i^2`
    """
    return individual[0]**2 + 1e6 * sum(gene * gene for gene in individual),

def rosenbrock(individual):
    """Rosenbrock test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - none
       * - Global optima
         - :math:`x_i = 1, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = \\sum_{i=1}^{N-1} (1-x_i)^2 + 100 (x_{i+1} - x_i^2 )^2`

    .. plot:: code/benchmarks/rosenbrock.py
       :width: 67 %
    """
    return sum(100 * (x * x - y)**2 + (1. - x)**2 \
                   for x, y in zip(individual[:-1], individual[1:])),

def h1(individual):
    """ Simple two-dimensional function containing several local maxima.
    From: The Merits of a Parallel Genetic Algorithm in Solving Hard
    Optimization Problems, A. J. Knoek van Soest and L. J. R. Richard
    Casius, J. Biomech. Eng. 125, 141 (2003)

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - maximization
       * - Range
         - :math:`x_i \in [-100, 100]`
       * - Global optima
         - :math:`\mathbf{x} = (8.6998, 6.7665)`, :math:`f(\mathbf{x}) = 2`\n
       * - Function
         - :math:`f(\mathbf{x}) = \\frac{\sin(x_1 - \\frac{x_2}{8})^2 + \
            \\sin(x_2 + \\frac{x_1}{8})^2}{\\sqrt{(x_1 - 8.6998)^2 + \
            (x_2 - 6.7665)^2} + 1}`

    .. plot:: code/benchmarks/h1.py
       :width: 67 %
    """
    num = (sin(individual[0] - individual[1] / 8))**2 + (sin(individual[1] + individual[0] / 8))**2
    denum = ((individual[0] - 8.6998)**2 + (individual[1] - 6.7665)**2)**0.5 + 1
    return num / denum,

# Multimodal
def ackley(individual):
    """Ackley test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-15, 30]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = 20 - 20\exp\left(-0.2\sqrt{\\frac{1}{N} \
            \\sum_{i=1}^N x_i^2} \\right) + e - \\exp\\left(\\frac{1}{N}\sum_{i=1}^N \\cos(2\pi x_i) \\right)`

    .. plot:: code/benchmarks/ackley.py
       :width: 67 %
    """
    N = len(individual)
    return 20 - 20 * exp(-0.2*sqrt(1.0/N * sum(x**2 for x in individual))) \
            + e - exp(1.0/N * sum(cos(2*pi*x) for x in individual)),

def bohachevsky(individual):
    """Bohachevsky test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-100, 100]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         -  :math:`f(\mathbf{x}) = \sum_{i=1}^{N-1}(x_i^2 + 2x_{i+1}^2 - \
                   0.3\cos(3\pi x_i) - 0.4\cos(4\pi x_{i+1}) + 0.7)`

    .. plot:: code/benchmarks/bohachevsky.py
       :width: 67 %
    """
    return sum(x**2 + 2*x1**2 - 0.3*cos(3*pi*x) - 0.4*cos(4*pi*x1) + 0.7
                for x, x1 in zip(individual[:-1], individual[1:])),

def griewank(individual):
    """Griewank test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-600, 600]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = \\frac{1}{4000}\\sum_{i=1}^N\,x_i^2 - \
                  \prod_{i=1}^N\\cos\\left(\\frac{x_i}{\sqrt{i}}\\right) + 1`

    .. plot:: code/benchmarks/griewank.py
       :width: 67 %
    """
    return 1.0/4000.0 * sum(x**2 for x in individual) - \
        reduce(mul, (cos(x/sqrt(i+1.0)) for i, x in enumerate(individual)), 1) + 1,

def rastrigin(individual):
    """Rastrigin test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-5.12, 5.12]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = 10N + \sum_{i=1}^N x_i^2 - 10 \\cos(2\\pi x_i)`

    .. plot:: code/benchmarks/rastrigin.py
       :width: 67 %
    """
    return 10 * len(individual) + sum(gene * gene - 10 * \
                        cos(2 * pi * gene) for gene in individual),

def rastrigin_scaled(individual):
    """Scaled Rastrigin test objective function.

    :math:`f_{\\text{RastScaled}}(\mathbf{x}) = 10N + \sum_{i=1}^N \
        \left(10^{\left(\\frac{i-1}{N-1}\\right)} x_i \\right)^2 x_i)^2 - \
        10\cos\\left(2\\pi 10^{\left(\\frac{i-1}{N-1}\\right)} x_i \\right)`
    """
    N = len(individual)
    return 10*N + sum((10**(i/(N-1))*x)**2 -
                      10*cos(2*pi*10**(i/(N-1))*x) for i, x in enumerate(individual)),

def rastrigin_skew(individual):
    """Skewed Rastrigin test objective function.

     :math:`f_{\\text{RastSkew}}(\mathbf{x}) = 10N \sum_{i=1}^N \left(y_i^2 - 10 \\cos(2\\pi x_i)\\right)`

     :math:`\\text{with } y_i = \
                            \\begin{cases} \
                                10\\cdot x_i & \\text{ if } x_i > 0,\\\ \
                                x_i & \\text{ otherwise } \
                            \\end{cases}`
    """
    N = len(individual)
    return 10*N + sum((10*x if x > 0 else x)**2
                    - 10*cos(2*pi*(10*x if x > 0 else x)) for x in individual),

def schaffer(individual):
    """Schaffer test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-100, 100]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         -  :math:`f(\mathbf{x}) = \sum_{i=1}^{N-1} (x_i^2+x_{i+1}^2)^{0.25} \cdot \
                  \\left[ \sin^2(50\cdot(x_i^2+x_{i+1}^2)^{0.10}) + 1.0 \
                  \\right]`

    .. plot:: code/benchmarks/schaffer.py
        :width: 67 %
    """
    return sum((x**2+x1**2)**0.25 * ((sin(50*(x**2+x1**2)**0.1))**2+1.0)
                for x, x1 in zip(individual[:-1], individual[1:])),

def schwefel(individual):
    """Schwefel test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-500, 500]`
       * - Global optima
         - :math:`x_i = 420.96874636, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
       * - Function
         - :math:`f(\mathbf{x}) = 418.9828872724339\cdot N - \
            \sum_{i=1}^N\,x_i\sin\\left(\sqrt{|x_i|}\\right)`


    .. plot:: code/benchmarks/schwefel.py
        :width: 67 %
    """
    N = len(individual)
    return 418.9828872724339*N-sum(x*sin(sqrt(abs(x))) for x in individual),

def himmelblau(individual):
    """The Himmelblau's function is multimodal with 4 defined minimums in
    :math:`[-6, 6]^2`.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \in [-6, 6]`
       * - Global optima
         - :math:`\mathbf{x}_1 = (3.0, 2.0)`, :math:`f(\mathbf{x}_1) = 0`\n
           :math:`\mathbf{x}_2 = (-2.805118, 3.131312)`, :math:`f(\mathbf{x}_2) = 0`\n
           :math:`\mathbf{x}_3 = (-3.779310, -3.283186)`, :math:`f(\mathbf{x}_3) = 0`\n
           :math:`\mathbf{x}_4 = (3.584428, -1.848126)`, :math:`f(\mathbf{x}_4) = 0`\n
       * - Function
         - :math:`f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 -7)^2`

    .. plot:: code/benchmarks/himmelblau.py
        :width: 67 %
    """
    return (individual[0] * individual[0] + individual[1] - 11)**2 + \
        (individual[0] + individual[1] * individual[1] - 7)**2,

def shekel(individual, a, c):
    """The Shekel multimodal function can have any number of maxima. The number
    of maxima is given by the length of any of the arguments *a* or *c*, *a*
    is a matrix of size :math:`M\\times N`, where *M* is the number of maxima
    and *N* the number of dimensions and *c* is a :math:`M\\times 1` vector.

    :math:`f_\\text{Shekel}(\mathbf{x}) = \\sum_{i = 1}^{M} \\frac{1}{c_{i} +
    \\sum_{j = 1}^{N} (x_{j} - a_{ij})^2 }`

    The following figure uses

    :math:`\\mathcal{A} = \\begin{bmatrix} 0.5 & 0.5 \\\\ 0.25 & 0.25 \\\\
    0.25 & 0.75 \\\\ 0.75 & 0.25 \\\\ 0.75 & 0.75 \\end{bmatrix}` and
    :math:`\\mathbf{c} = \\begin{bmatrix} 0.002 \\\\ 0.005 \\\\ 0.005
    \\\\ 0.005 \\\\ 0.005 \\end{bmatrix}`, thus defining 5 maximums in
    :math:`\\mathbb{R}^2`.

    .. plot:: code/benchmarks/shekel.py
        :width: 67 %
    """
    return sum((1. / (c[i] + sum((individual[j] - aij)**2 for j, aij in enumerate(a[i])))) for i in range(len(c))),

# Multiobjectives
def kursawe(individual):
    """Kursawe multiobjective function.

    :math:`f_{\\text{Kursawe}1}(\\mathbf{x}) = \\sum_{i=1}^{N-1} -10 e^{-0.2 \\sqrt{x_i^2 + x_{i+1}^2} }`

    :math:`f_{\\text{Kursawe}2}(\\mathbf{x}) = \\sum_{i=1}^{N} |x_i|^{0.8} + 5 \\sin(x_i^3)`

    .. plot:: code/benchmarks/kursawe.py
       :width: 100 %
    """
    f1 = sum(-10 * exp(-0.2 * sqrt(x * x + y * y)) for x, y in zip(individual[:-1], individual[1:]))
    f2 = sum(abs(x)**0.8 + 5 * sin(x * x * x) for x in individual)
    return f1, f2


def schaffer_mo(individual):
    """Schaffer's multiobjective function on a one attribute *individual*.
    From: J. D. Schaffer, "Multiple objective optimization with vector
    evaluated genetic algorithms", in Proceedings of the First International
    Conference on Genetic Algorithms, 1987.
    :math:`f_{\\text{Schaffer}1}(\\mathbf{x}) = x_1^2`
    :math:`f_{\\text{Schaffer}2}(\\mathbf{x}) = (x_1-2)^2`
    """
    return individual[0] ** 2, (individual[0] - 2) ** 2

@evals_dec
def zdt1(individual):
    """ZDT1 multiobjective function.
    :math:`g(\\mathbf{x}) = 1 + \\frac{9}{n-1}\\sum_{i=2}^n x_i`
    :math:`f_{\\text{ZDT1}1}(\\mathbf{x}) = x_1`
    :math:`f_{\\text{ZDT1}2}(\\mathbf{x}) = g(\\mathbf{x})\\left[1 - \\sqrt{\\frac{x_1}{g(\\mathbf{x})}}\\right]`
    """
    g  = 1.0 + 9.0*sum(individual[1:])/(len(individual)-1)
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1/g))
    return f1, f2
zdt1.dimension = 6   # 30
zdt1.nadir = [11., 11.]
zdt1.bound_low = [0.] + [0.]*(zdt1.dimension-1)
zdt1.bound_up = [1.] + [1.]*(zdt1.dimension-1)
zdt1.crits = 2

@evals_dec
def zdt2(individual):
    """ZDT2 multiobjective function.
    :math:`g(\\mathbf{x}) = 1 + \\frac{9}{n-1}\\sum_{i=2}^n x_i`
    :math:`f_{\\text{ZDT2}1}(\\mathbf{x}) = x_1`
    :math:`f_{\\text{ZDT2}2}(\\mathbf{x}) = g(\\mathbf{x})\\left[1 - \\left(\\frac{x_1}{g(\\mathbf{x})}\\right)^2\\right]`
    """
    g  = 1.0 + 9.0*sum(individual[1:])/(len(individual)-1)
    f1 = individual[0]
    f2 = g * (1 - (f1/g)**2)
    return f1, f2
zdt2.dimension = 6  # 30
zdt2.nadir = [11., 11.]
zdt2.bound_low = [0.] + [0.]*(zdt2.dimension-1)
zdt2.bound_up = [1.] + [1.]*(zdt2.dimension-1)
zdt2.crits = 2

@evals_dec
def zdt3(individual):
    """ZDT3 multiobjective function.

    :math:`g(\\mathbf{x}) = 1 + \\frac{9}{n-1}\\sum_{i=2}^n x_i`

    :math:`f_{\\text{ZDT3}1}(\\mathbf{x}) = x_1`

    :math:`f_{\\text{ZDT3}2}(\\mathbf{x}) = g(\\mathbf{x})\\left[1 - \\sqrt{\\frac{x_1}{g(\\mathbf{x})}} - \\frac{x_1}{g(\\mathbf{x})}\\sin(10\\pi x_1)\\right]`

    """

    g  = 1.0 + 9.0*sum(individual[1:])/(len(individual)-1)
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1/g) - f1/g * sin(10*pi*f1))
    return f1, f2
zdt3.dimension = 6  # 30
zdt3.nadir = [11., 11.]
zdt3.bound_low = [0.] + [0.]*(zdt3.dimension-1)
zdt3.bound_up = [1.] + [1.]*(zdt3.dimension-1)
zdt3.crits = 2

@evals_dec
def zdt4(individual):
    """ZDT4 multiobjective function.

    :math:`g(\\mathbf{x}) = 1 + 10(n-1) + \\sum_{i=2}^n \\left[ x_i^2 - 10\\cos(4\\pi x_i) \\right]`

    :math:`f_{\\text{ZDT4}1}(\\mathbf{x}) = x_1`

    :math:`f_{\\text{ZDT4}2}(\\mathbf{x}) = g(\\mathbf{x})\\left[ 1 - \\sqrt{x_1/g(\\mathbf{x})} \\right]`

    """
    g  = 1 + 10*(len(individual)-1) + sum(xi**2 - 10*cos(4*pi*xi) for xi in individual[1:])
    f1 = individual[0]
    f2 = g * (1 - sqrt(f1/g))
    return f1, f2
zdt4.dimension = 6  # 10
zdt4.nadir = [11., 11.]
zdt4.bound_low = [0.] + [-5.]*(zdt4.dimension-1)
zdt4.bound_up = [1.] + [5.]*(zdt4.dimension-1)
zdt4.crits = 2

@evals_dec
def zdt6(individual):
    """ZDT6 multiobjective function.

    :math:`g(\\mathbf{x}) = 1 + 9 \\left[ \\left(\\sum_{i=2}^n x_i\\right)/(n-1) \\right]^{0.25}`

    :math:`f_{\\text{ZDT6}1}(\\mathbf{x}) = 1 - \\exp(-4x_1)\\sin^6(6\\pi x_1)`

    :math:`f_{\\text{ZDT6}2}(\\mathbf{x}) = g(\\mathbf{x}) \left[ 1 - (f_{\\text{ZDT6}1}(\\mathbf{x})/g(\\mathbf{x}))^2 \\right]`

    """
    g  = 1 + 9 * (sum(individual[1:]) / (len(individual)-1))**0.25
    f1 = 1 - exp(-4*individual[0]) * sin(6*pi*individual[0])**6
    f2 = g * (1 - (f1/g)**2)
    return f1, f2
zdt6.dimension = 6  # 10
zdt6.nadir = [11., 11.]
zdt6.bound_low = [0.] + [0.]*(zdt6.dimension-1)
zdt6.bound_up = [1.] + [1.]*(zdt6.dimension-1)
zdt6.crits = 2

@evals_dec
def dtlz1(individual, obj=3):
    """DTLZ1 multiobjective function. It returns a tuple of *obj* values.
    The individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.

    :math:`g(\\mathbf{x}_m) = 100\\left(|\\mathbf{x}_m| + \sum_{x_i \in \\mathbf{x}_m}\\left((x_i - 0.5)^2 - \\cos(20\pi(x_i - 0.5))\\right)\\right)`

    :math:`f_{\\text{DTLZ1}1}(\\mathbf{x}) = \\frac{1}{2} (1 + g(\\mathbf{x}_m)) \\prod_{i=1}^{m-1}x_i`

    :math:`f_{\\text{DTLZ1}2}(\\mathbf{x}) = \\frac{1}{2} (1 + g(\\mathbf{x}_m)) (1-x_{m-1}) \\prod_{i=1}^{m-2}x_i`

    :math:`\\ldots`

    :math:`f_{\\text{DTLZ1}m-1}(\\mathbf{x}) = \\frac{1}{2} (1 + g(\\mathbf{x}_m)) (1 - x_2) x_1`

    :math:`f_{\\text{DTLZ1}m}(\\mathbf{x}) = \\frac{1}{2} (1 - x_1)(1 + g(\\mathbf{x}_m))`

    Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m` is a
    vector of the remaining attributes :math:`[x_m~\\ldots~x_n]` of the
    individual in :math:`n > m` dimensions.

    """
    g = 100 * (len(individual[obj-1:]) + sum((xi-0.5)**2 - cos(20*pi*(xi-0.5)) for xi in individual[obj-1:]))
    f = [0.5 * reduce(mul, individual[:obj-1], 1) * (1 + g)]
    f.extend(0.5 * reduce(mul, individual[:m], 1) * (1 - individual[m]) * (1 + g) for m in reversed(xrange(obj-1)))
    return f
dtlz1.dimension = 6  # 10
dtlz1.nadir = [1., 1., 1.]    # PF: (sum fi) = 1, fi > 0
dtlz1.bound_low = [0.] + [0.]*(dtlz1.dimension-1)
dtlz1.bound_up = [1.] + [1.]*(dtlz1.dimension-1)
dtlz1.crits = 3

@evals_dec
def dtlz2(individual, obj=3):
    """DTLZ2 multiobjective function. It returns a tuple of *obj* values.
    The individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.

    :math:`g(\\mathbf{x}_m) = \\sum_{x_i \in \\mathbf{x}_m} (x_i - 0.5)^2`

    :math:`f_{\\text{DTLZ2}1}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\prod_{i=1}^{m-1} \\cos(0.5x_i\pi)`

    :math:`f_{\\text{DTLZ2}2}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{m-1}\pi ) \\prod_{i=1}^{m-2} \\cos(0.5x_i\pi)`

    :math:`\\ldots`

    :math:`f_{\\text{DTLZ2}m}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{1}\pi )`

    Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m` is a
    vector of the remaining attributes :math:`[x_m~\\ldots~x_n]` of the
    individual in :math:`n > m` dimensions.
    """
    xc = individual[:obj-1]
    xm = individual[obj-1:]
    g = sum((xi-0.5)**2 for xi in xm)
    f = [(1.0+g) *  reduce(mul, (cos(0.5*xi*pi) for xi in xc), 1.0)]
    f.extend((1.0+g) * reduce(mul, (cos(0.5*xi*pi) for xi in xc[:m]), 1) * sin(0.5*xc[m]*pi) for m in range(obj-2, -1, -1))

    return f
dtlz2.dimension = 6  # 10
dtlz2.nadir = [1.5, 1.5, 1.5]
dtlz2.bound_low = [0.] + [0.]*(dtlz2.dimension-1)
dtlz2.bound_up = [1.] + [1.]*(dtlz2.dimension-1)
dtlz2.crits = 3

@evals_dec
def dtlz3(individual, obj=3):
    """DTLZ3 multiobjective function. It returns a tuple of *obj* values.
    The individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.

    :math:`g(\\mathbf{x}_m) = 100\\left(|\\mathbf{x}_m| + \sum_{x_i \in \\mathbf{x}_m}\\left((x_i - 0.5)^2 - \\cos(20\pi(x_i - 0.5))\\right)\\right)`

    :math:`f_{\\text{DTLZ3}1}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\prod_{i=1}^{m-1} \\cos(0.5x_i\pi)`

    :math:`f_{\\text{DTLZ3}2}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{m-1}\pi ) \\prod_{i=1}^{m-2} \\cos(0.5x_i\pi)`

    :math:`\\ldots`

    :math:`f_{\\text{DTLZ3}m}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{1}\pi )`

    Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m` is a
    vector of the remaining attributes :math:`[x_m~\\ldots~x_n]` of the
    individual in :math:`n > m` dimensions.
    """
    xc = individual[:obj-1]
    xm = individual[obj-1:]
    g = 100 * (len(xm) + sum((xi-0.5)**2 - cos(20*pi*(xi-0.5)) for xi in xm))
    f = [(1.0+g) *  reduce(mul, (cos(0.5*xi*pi) for xi in xc), 1.0)]
    f.extend((1.0+g) * reduce(mul, (cos(0.5*xi*pi) for xi in xc[:m]), 1) * sin(0.5*xc[m]*pi) for m in range(obj-2, -1, -1))
    return f
dtlz3.dimension = 6  # 10
dtlz3.nadir = [1.5, 1.5, 1.5]
dtlz3.bound_low = [0.] + [0.]*(dtlz3.dimension-1)
dtlz3.bound_up = [1.] + [1.]*(dtlz3.dimension-1)
dtlz3.crits = 3

@evals_dec
def dtlz4(individual, obj=3, alpha=100):
    """DTLZ4 multiobjective function. It returns a tuple of *obj* values. The
    individual must have at least *obj* elements. The *alpha* parameter allows
    for a meta-variable mapping in :func:`dtlz2` :math:`x_i \\rightarrow
    x_i^\\alpha`, the authors suggest :math:`\\alpha = 100`.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.

    :math:`g(\\mathbf{x}_m) = \\sum_{x_i \in \\mathbf{x}_m} (x_i - 0.5)^2`

    :math:`f_{\\text{DTLZ4}1}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\prod_{i=1}^{m-1} \\cos(0.5x_i^\\alpha\pi)`

    :math:`f_{\\text{DTLZ4}2}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{m-1}^\\alpha\pi ) \\prod_{i=1}^{m-2} \\cos(0.5x_i^\\alpha\pi)`

    :math:`\\ldots`

    :math:`f_{\\text{DTLZ4}m}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{1}^\\alpha\pi )`

    Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m` is a
    vector of the remaining attributes :math:`[x_m~\\ldots~x_n]` of the
    individual in :math:`n > m` dimensions.
    """
    xc = individual[:obj-1]
    xm = individual[obj-1:]
    g = sum((xi-0.5)**2 for xi in xm)
    f = [(1.0+g) *  reduce(mul, (cos(0.5*xi**alpha*pi) for xi in xc), 1.0)]
    f.extend((1.0+g) * reduce(mul, (cos(0.5*xi**alpha*pi) for xi in xc[:m]), 1) * sin(0.5*xc[m]**alpha*pi) for m in range(obj-2, -1, -1))
    return f
dtlz4.dimension = 6  # 10
dtlz4.nadir = [1.5, 1.5, 1.5]
dtlz4.bound_low = [0.] + [0.]*(dtlz4.dimension-1)
dtlz4.bound_up = [1.] + [1.]*(dtlz4.dimension-1)
dtlz4.crits = 3

@evals_dec
def dtlz5(ind, n_objs=3):
    """DTLZ5 multiobjective function. It returns a tuple of *obj* values. The
    individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825-830, IEEE Press, 2002.
    """
    g = lambda x: sum([(a - 0.5)**2 for a in x])
    gval = g(ind[n_objs-1:])

    theta = lambda x: pi / (4.0 * (1 + gval)) * (1 + 2 * gval * x)
    fit = [(1 + gval) * cos(pi / 2.0 * ind[0]) * reduce(lambda x,y: x*y, [cos(theta(a)) for a in ind[1:]])]

    for m in reversed(range(1, n_objs)):
        if m == 1:
            fit.append((1 + gval) * sin(pi / 2.0 * ind[0]))
        else:
            fit.append((1 + gval) * cos(pi / 2.0 * ind[0]) *
                       reduce(lambda x,y: x*y, [cos(theta(a)) for a in ind[1:m-1]], 1) * sin(theta(ind[m-1])))
    return fit
dtlz5.dimension = 6  # 10
dtlz5.nadir = [1.5, 1.5, 1.5]
dtlz5.bound_low = [0.] + [0.]*(dtlz5.dimension-1)
dtlz5.bound_up = [1.] + [1.]*(dtlz5.dimension-1)
dtlz5.crits = 3
# Bi-objective case has only one point in Pareto-front

@evals_dec
def dtlz6(ind, n_objs=3):
    """DTLZ6 multiobjective function. It returns a tuple of *obj* values. The
    individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825-830, IEEE Press, 2002.
    """
    gval = sum([a**0.1 for a in ind[n_objs-1:]])
    theta = lambda x: pi / (4.0 * (1 + gval)) * (1 + 2 * gval * x)

    fit = [(1 + gval) * cos(pi / 2.0 * ind[0]) *
           reduce(lambda x,y: x*y, [cos(theta(a)) for a in ind[1:]])]

    for m in reversed(range(1, n_objs)):
        if m == 1:
            fit.append((1 + gval) * sin(pi / 2.0 * ind[0]))
        else:
            fit.append((1 + gval) * cos(pi / 2.0 * ind[0]) *
                       reduce(lambda x,y: x*y, [cos(theta(a)) for a in ind[1:m-1]], 1) * sin(theta(ind[m-1])))
    return fit
dtlz6.dimension = 6  # 10
dtlz6.nadir = [1.5, 1.5, 1.5]
dtlz6.bound_low = [0.] + [0.]*(dtlz6.dimension-1)
dtlz6.bound_up = [1.] + [1.]*(dtlz6.dimension-1)
dtlz6.crits = 3
# Bi-objective case has only one point in Pareto-front

@evals_dec
def dtlz7(ind, n_objs=3):
    """DTLZ7 multiobjective function. It returns a tuple of *obj* values. The
    individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825-830, IEEE Press, 2002.
    """
    gval = 1 + 9.0 / len(ind[n_objs-1:]) * sum([a for a in ind[n_objs-1:]])
    fit = [x for x in ind[:n_objs-1]]
    fit.append((1 + gval) * (n_objs - sum([a / (1.0 + gval) * (1 + sin(3 * pi * a)) for a in ind[:n_objs-1]])))
    return fit
dtlz7.dimension = 6  # 10
dtlz7.nadir = [15., 15., 15.]
dtlz7.bound_low = [0.] + [0.]*(dtlz7.dimension-1)
dtlz7.bound_up = [1.] + [1.]*(dtlz7.dimension-1)
dtlz7.crits = 3

def fonseca(individual):
    """Fonseca and Fleming's multiobjective function.
    From: C. M. Fonseca and P. J. Fleming, "Multiobjective optimization and
    multiple constraint handling with evolutionary algorithms -- Part II:
    Application example", IEEE Transactions on Systems, Man and Cybernetics,
    1998.

    :math:`f_{\\text{Fonseca}1}(\\mathbf{x}) = 1 - e^{-\\sum_{i=1}^{3}(x_i - \\frac{1}{\\sqrt{3}})^2}`

    :math:`f_{\\text{Fonseca}2}(\\mathbf{x}) = 1 - e^{-\\sum_{i=1}^{3}(x_i + \\frac{1}{\\sqrt{3}})^2}`
    """
    f_1 = 1 - exp(-sum((xi - 1/sqrt(3))**2 for xi in individual[:3]))
    f_2 = 1 - exp(-sum((xi + 1/sqrt(3))**2 for xi in individual[:3]))
    return f_1, f_2

def poloni(individual):
    """Poloni's multiobjective function on a two attribute *individual*. From:
    C. Poloni, "Hybrid GA for multi objective aerodynamic shape optimization",
    in Genetic Algorithms in Engineering and Computer Science, 1997.

    :math:`A_1 = 0.5 \\sin (1) - 2 \\cos (1) + \\sin (2) - 1.5 \\cos (2)`

    :math:`A_2 = 1.5 \\sin (1) - \\cos (1) + 2 \\sin (2) - 0.5 \\cos (2)`

    :math:`B_1 = 0.5 \\sin (x_1) - 2 \\cos (x_1) + \\sin (x_2) - 1.5 \\cos (x_2)`

    :math:`B_2 = 1.5 \\sin (x_1) - cos(x_1) + 2 \\sin (x_2) - 0.5 \\cos (x_2)`

    :math:`f_{\\text{Poloni}1}(\\mathbf{x}) = 1 + (A_1 - B_1)^2 + (A_2 - B_2)^2`

    :math:`f_{\\text{Poloni}2}(\\mathbf{x}) = (x_1 + 3)^2 + (x_2 + 1)^2`
    """
    x_1 = individual[0]
    x_2 = individual[1]
    A_1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2)
    A_2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2)
    B_1 = 0.5 * sin(x_1) - 2 * cos(x_1) + sin(x_2) - 1.5 * cos(x_2)
    B_2 = 1.5 * sin(x_1) - cos(x_1) + 2 * sin(x_2) - 0.5 * cos(x_2)
    return 1 + (A_1 - B_1)**2 + (A_2 - B_2)**2, (x_1 + 3)**2 + (x_2 + 1)**2

def dent(individual, lambda_ = 0.85):
    """Test problem Dent. Two-objective problem with a "dent". *individual* has
    two attributes that take values in [-1.5, 1.5].
    From: Schuetze, O., Laumanns, M., Tantar, E., Coello Coello, C.A., & Talbi, E.-G. (2010).
    Computing gap free Pareto front approximations with stochastic search algorithms.
    Evolutionary Computation, 18(1), 65--96. doi:10.1162/evco.2010.18.1.18103

    Note that in that paper Dent source is stated as:
    K. Witting and M. Hessel von Molo. Private communication, 2006.
    """
    d = lambda_ * exp(-(individual[0] - individual[1]) ** 2)
    f1 = 0.5 * (sqrt(1 + (individual[0] + individual[1]) ** 2) + \
                sqrt(1 + (individual[0] - individual[1]) ** 2) + \
                individual[0] - individual[1]) + d
    f2 = 0.5 * (sqrt(1 + (individual[0] + individual[1]) ** 2) + \
                sqrt(1 + (individual[0] - individual[1]) ** 2) - \
                individual[0] + individual[1]) + d
    return f1, f2


@evals_dec
def ep1(xs):
    """Returns bi-objective function value of first problem defined in the source.

    :param xs: point values in variable space.

    :source: A. Zilinskas and J. Zilinskas. "Adaptation of a one-step
    worst-case optimal univariate algorithm of bi-objective Lipschitz
    optimization to multidimensional problems." Communications in Nonlinear
    Science and Numerical Simulation 21.1 (2015): 89-98.
    """
    f1 = xs[0]
    f2 = min(abs(xs[0] - 1.), 1.5-xs[0]) + xs[1] + 1
    return f1, f2
ep1.nadir = [2., 3.]
ep1.bound_low = [0., 0.]
ep1.bound_up = [2., 2.]
ep1.dimension = 2.
ep1.crits = 2


@evals_dec
def ep2(xs):
    """Returns bi-objective funtion value of second problem defined in the source.

    :param xs: point values in variable space.

    :source: A. Zilinskas and J. Zilinskas. "Adaptation of a one-step
    worst-case optimal univariate algorithm of bi-objective Lipschitz
    optimization to multidimensional problems." Communications in Nonlinear
    Science and Numerical Simulation 21.1 (2015): 89-98.
    """
    f1 = (xs[0] - 1.)*xs[1]*xs[1] + 1.
    f2 = xs[1]
    return f1, f2
ep2.nadir = [1., 1.]
ep2.bound_low = [0., 0.]
ep2.bound_up = [1., 1.]
ep2.dimension = 2.
ep2.crits = 2


def get_problem(func_name):
    if (func_name == 'zdt1'): return zdt1
    if (func_name == 'zdt2'): return zdt2
    if (func_name == 'zdt3'): return zdt3
    if (func_name == 'zdt4'): return zdt4
    if (func_name == 'zdt6'): return zdt6
    if (func_name == 'dtlz1'): return dtlz1
    if (func_name == 'dtlz2'): return dtlz2
    if (func_name == 'dtlz3'): return dtlz3
    if (func_name == 'dtlz4'): return dtlz4
    if (func_name == 'dtlz5'): return dtlz5
    if (func_name == 'dtlz6'): return dtlz6
    if (func_name == 'dtlz7'): return dtlz7
