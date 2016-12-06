import time
import os

import numpy as np

# LP Solvers
import pulp
from cvxopt.modeling import variable, op, dot
from cvxopt import matrix
import cvxopt

M = 10**6


def pruneLPCvxopt(glob, i, alphas, marked, returnObj=False):
    # set up variables
    sigma = variable()
    pi = variable(3)

    start = time.time()
    # sum to 1 constraint
    c1 = (sum(pi) == 1)
    # the pi must be greater than 0
    c2 = (pi >= 0)
    # alpha best for some prob distribution
    c3 = [(M * dot(matrix(alphas[i] - a), pi) - sigma >= 0)
          for j, a in enumerate(alphas) if marked[j] and j != i]
    # if none of these constraints, then LP unbounded so return 1
    if len(c3) == 0:
        return 1
    lp = op(-1 * sigma, [c1, c2] + c3)
    cvxopt.solvers.options['show_progress'] = False
    end = time.time()
    glob.constructTime += (end - start)

    start = time.time()
    # can use either glpk or mosek
    lp.solve('dense', 'glpk')
    # if failed to find optimal soln
    if lp.status == 'unknown':
        # try using default python solver
        lp.solve()

    # if default python solver also does not work, then do not prune
    if lp.status == 'unknown':
        print "Cvxopt no optimal soln. Status:", lp.status
        # do not prune
        obj = 1
    else:
        obj = sigma.value[0]
    end = time.time()
    glob.solveTime += (end - start)

    if returnObj:
        return obj

    return obj > 0


def pruneLPCplex(glob, i, alphas, marked, returnObj=False):
    start = time.time()
    makeAMPLDataFile(i, alphas, marked)
    end = time.time()
    glob.constructTime += (end - start)

    start = time.time()
    os.system('ampl ampl/lp.run')
    with open("ampl/lpres.txt", "r") as f:
        obj = float(f.readline())
    end = time.time()
    glob.solveTime += (end - start)

    if returnObj:
        return obj

    return obj > 0


def makeAMPLDataFile(i, alphas, marked,
                     dataFile="ampl/lp.dat", alphaFile="ampl/diff.txt"):
    diff = np.array([M * (alphas[i] - a)
                     for j, a in enumerate(alphas) if marked[j] and j != i])

    dataFileTxt = "param numAlpha := {0};\n".format(len(diff))
    dataFileTxt += "read {i in alphaI, s in S} diff[i, s] < ampl/diff.txt;\n"

    with open(dataFile, 'w') as df:
        df.write(dataFileTxt)
        df.flush()

    np.savetxt(alphaFile, diff, fmt='%.10f')


def pruneLPPulp(glob, i, alphas, marked, printOut=False, returnObj=False):
    # alpha to check if prune
    alpha = alphas[i]
    start = time.time()
    # problem is a maximization problem
    prob = pulp.LpProblem("Reduce", pulp.LpMaximize)
    # set up arbitrary probability distribution over state pi
    pi = np.array([pulp.LpVariable("pi" + str(s), lowBound=0)
                   for s in range(3)])
    # set up sigma
    sigma = pulp.LpVariable("sigma", lowBound=None, upBound=None)
    # Objective: max sigma
    prob += sigma, "Maximize sigma"
    # The pi must sum to 1
    prob += pulp.lpSum(pi) == 1, "ProbDistribution"
    # Check if this alpha vector does better than any other alpha vector
    for j, a in enumerate(alphas):
        if i != j and marked[j]:
            prob += M * np.dot(alpha - a, pi) - sigma >= 0, ""
    end = time.time()
    glob.constructTime += (end - start)

    start = time.time()
    prob.solve()
    end = time.time()
    glob.solveTime += (end - start)

    obj = pulp.value(prob.objective)
    if printOut:
        print "Problem: ", prob
        print "Objective (sigma): ", obj
        print "Pi: ", [p.value() for p in pi]

    if returnObj:
        return obj

    # return True if should not prune, False if should prune
    return obj > 0
