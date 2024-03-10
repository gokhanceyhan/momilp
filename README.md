# momilp
Solver for a class of Multi-objective Mixed Integer Linear Programs (MOMILPs)

This repository provides an implementation of the algorithm, Cone-Based Search Algorithm (CBSA), developed in our following paper: 
```
Gökhan Ceyhan, Murat Köksalan, Banu Lokman (2023) Finding the Nondominated Set and Efficient Integer Vectors for a Class of Three-Objective Mixed-Integer Linear Programs. Management Science 69(10):6001-6020.
```

## Programming language
This project is implemented in Python 3.

## Setup the environment
After cloning the code, make sure you are in the root directory of the repo and follow these steps:
```
python3 -m venv momilp_venv
source momilp_venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:."
```

## Install the dependencies
```
pip install -r requirements.txt
```

## Run help to see the usage of the command line app
```
python ./src/apps/momilp_solver --help
```

## Instances
### TOMILP:
Three-objective mixed-integer programming problem instances in the `.lp` file format. Subfolder names indicate the number of constraints and the number of integer variables. Each instance has equal number of continuous and integer variables. 
For example, in subfolder `10`, instances contain 10 linear constraints with 5 continuous and 5 integer variables. 
- Instances under `I1` folder contain instances with `one` discrete-valued objective.
- Instances under `B1` folder contain instances with `one` discrete-valued objective and all integer variables are binary.
- Instances under `B2` folder contain instances with `two` discrete-valued objective and all integer variables are binary.

### BOMBLP:
Bi-objective mixed-binary linear programming instances used in `Boland et al. (2015)` that are converted to their corresponding `.lp` file formats.
```
@article{boland2015boilp,
  title={A criterion space search algorithm for biobjective integer programming: The balanced box method},
  author={Boland, Natashia and Charkhgard, Hadi and Savelsbergh, Martin},
  journal={INFORMS Journal on Computing},
  volume={27},
  number={4},
  pages={735--754},
  year={2015},
  publisher={INFORMS}
}
```
Subfolder names indicate the number of constraints and 
the number of binary variables. Each instance has equal number of continuous and binary variables. For example, subfolder `20` contain 20 linear constraints with 10 continuous and 10 binary variables.

### TOKP:
Three-objective 0/1 single knapsack problems provided in `Kirlik (2014)` that are converted to their corresponding `.lp` file formats.
```
@misc{kslib,
    author = {Kirlik, Gokhan},
    title = {{Test instances for multi-objective discrete optimization problems}},
    howpublished = {\url{http://home.ku.edu.tr/~moolibrary/}},
    note = {Online; accessed 13 June 2021},
    year = {2014}
}
```
Subfolder names indicate the number of items in the knapsack. The instances under `10_alternative_efficient_sols` contain
modified versions of 10-item instances that lead to efficient integer vectors.

### examples:
Contain `.lp` files for the examples provided in the manuscript.

## Example usage for momilp solver
In order to run the solver, a valid `Gurobi` installation is required.
```
python ./src/apps/momilp_solver -d '0' -m ./instances/TOMILP/O3-C10/B1/ -s gurobi -w ~/Downloads
python ./src/apps/momilp_solver -d '0, 1' -m ./instances/TOMILP/O3-C10/B2/ -s gurobi -w ~/Downloads
python ./src/apps/momilp_solver -d '0, 1, 2' -m ./instances/TOKP/10 -s gurobi -w ~/Downloads
```
