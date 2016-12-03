import pulp
import numpy as np


def pruneLP(alpha, otherAlpha, printOut = False):
    # problem is a maximization problem
    prob = pulp.LpProblem("Reduce", pulp.LpMaximize)
    # set up arbitrary probability distribution over state pi
    pi = np.array([pulp.LpVariable("pi"+str(s), lowBound=0) for s in range(3)])
    # set up sigma
    sigma = pulp.LpVariable("sigma", lowBound=None, upBound=None)
    # Objective: max sigma
    prob += sigma, "Maximize sigma"
    # The pi must sum to 1
    prob += pulp.lpSum(pi) == 1, "ProbDistribution"
    # Check if this alpha vector does better than any other alpha vector
    for a in otherAlpha:
        prob += np.dot(alpha - a, pi) - sigma >= 0, ""
    prob.solve()
    obj = pulp.value(prob.objective)
    if printOut:
        print "Objective (sigma): ", obj
        print "Pi: ", [p.value() for p in pi]
    # return 0 if should not prune, 1 if should prune
    return obj <= 0