# pomdp
POMDP Approach to Personalize Mammography Screening Decisions
Alexander Goldberg, Lydia Goldberg, Gabriela Merz
Model based on http://pubsonline.informs.org.ezp-prod1.hul.harvard.edu/doi/suppl/10.1287/opre.1110.1019.

Directory contents:

model.py: Structure and methods for the POMDP model.

modelConstants.py: Constants used for actions and observations in the POMDP model.

monahan.py: Code for the POMDP solver using Monahan's Enumeration Algorithm.

perseus.py: Unfinished point based value iteration approximate solver.

pruneLP.py: Code to solve linear programs using pulp, cvxopt, and CPLEX.

stats.py: Statistics used in the POMDP model.

alpha: Directory containing pickle files of alpha vectors over 20, 40, and 80 time steps.

ampl: Directory containing code used for the CPLEX LP solver.

requirements.txt: Run "pip install -r requirements.txt" before attempting to run the project.

POMDPs for Personalized Mammography Screening.ipynb: IPython notebook to walk through the project.