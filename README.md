Installing requirements
=======================

`sudo pip install deap`

Licence
=======================
LGPL, same as in DEAP project (see [Distributed Evolutionary Algorithms in Python](https://github.com/DEAP/deap)).

Changes
=======
- `tools.py` Added new benchmark tool: uniformity.
- `problems.py` Added two new bi-objective problems: EP1, EP2.
- `problems.py` Added function decorator, which calculates how many unique calles were made.
- `nsga2.py` slightly modified interface to work with function wrapper.
