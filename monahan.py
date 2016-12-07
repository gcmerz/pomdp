import itertools
import time
import os
import pickle

import numpy as np

from model import CancerPOMDP
from modelConstants import W, M, MNEG, MPOS, SDNEG, SDPOS
from pruneLP import pruneLPPulp, pruneLPCplex, pruneLPCvxopt


class MonahanSolve(CancerPOMDP):

    def __init__(self, *args, **kwargs):
        CancerPOMDP.__init__(self, *args, **kwargs)
        self.alpha = {}
        # receive 0 reward at last time step (alphas are stored as tuples of
        # actions and vectors)
        self.alpha[self.tmax] = np.array([
            (-1, np.array([self.terminalReward(s) for s in self.SPO]))], dtype=tuple)
        self.time = self.tmax - 1

        self.solveTime = 0
        self.constructTime = 0
        self.LPSolver = "pulp"

    def solve(self, write=False):
        print "Using LP Solver", self.LPSolver, "\n"

        while self.time >= self.t0:
            # generate all alpha
            start = time.time()
            self.generateAllAlpha()
            end = time.time()
            self.printReport("enumeration", start, end)

            # prune dominated alpha
            start = time.time()
            self.eagleReduction()
            end = time.time()
            self.printReport("Eagle", start, end)

            # use LP to prune
            start = time.time()
            self.solveTime = 0
            self.constructTime = 0
            self.monahanElimination()
            end = time.time()
            self.printReport("LP", start, end)
            print "\tLP construct time: ", self.constructTime
            print "\tLP solve time: ", self.solveTime

            print "Completed time step ", self.time, "\n"

            if write:
                # write alpha to file every 5 time steps
                if (self.tmax - self.time) % 5 == 0:
                    start = time.time()
                    self.writeAlpha()
                    end = time.time()
                    print "Wrote alpha in {0} secs.\n".format(end - start)

            self.time -= 1

    def printReport(self, s, start, end):
        out = "Total alpha after " + s + ": "
        out += str(len(self.alpha[self.time]))
        out += ", time: "
        out += "{:.2f}".format(end - start)
        print out

    ##############################################################
        # Alpha Generating Steps #
    ##############################################################

    def generateAllAlpha(self):
        # find alpha to use to get maximal value if false positive mammogram
        # (only do this once for given timestep)
        alphaMaxValue = self.bestFalsePosAlpha(self.time)

        # set up permutations of 2 future alpha to correspond with generated
        # alpha vectors
        perms = itertools.product(
            self.alpha[self.time + 1], self.alpha[self.time + 1])
        totalWAlpha = len(self.alpha[self.time + 1])**2
        totalMAlpha = len(self.alpha[self.time + 1])
        totalAlpha = totalWAlpha + totalMAlpha
        self.alpha[self.time] = np.array(
            [(-1, np.array([0 for _ in self.SPO]))
             for i in xrange(totalAlpha)], dtype=tuple)

        i = 0
        # enumerate wait vectors
        for alpha1, alpha2 in perms:
            # alphas for the  next step each associated with an observation
            futureAlphas = {SDNEG: alpha1[1], SDPOS: alpha2[1]}
            self.alpha[self.time][i] = (W, self.generateWAlpha(futureAlphas))
            i += 1

        # enumerate mammogram vectors
        for alpha in self.alpha[self.time + 1]:
            # alphas for the next step each associated with an observation
            futureAlphas = {MNEG: alpha[1], MPOS: alphaMaxValue}
            self.alpha[self.time][
                i] = (M, self.generateMAlpha(futureAlphas))
            i += 1

    def generateWAlpha(self, futureAlphas):
        # initialize alpha as vector of partially observable states
        alpha = [0 for _ in self.SPO]
        # compute alpha vector for each state
        for s in self.SPO:
            for o in self.O[W]:
                # future value given the two alpha vectors associated with
                # observations
                futureValue = sum([self.transProb(
                    self.time, s, newS) * futureAlphas[o][newS] for newS in self.SPO])
                value = self.obsProb(
                    self.time, s, o) * (self.reward(self.time, s, W, o) + futureValue)
                alpha[s] += value
        return alpha

    def generateMAlpha(self, futureAlphas):
         # initialize alpha as vector of partially observable states
        alpha = [0 for _ in self.SPO]
        # compute alpha vector for each state
        for s in self.SPO:
            for o in self.O[M]:
                if o == MPOS:
                    # if false positive, then know you do not have cancer, so know
                    # next belief state and can choose maximal alpha
                    if s == 0:
                        futureValue = futureAlphas[MPOS]
                    else:
                        # future reward is lump sum
                        futureValue = self.lumpSumReward(self.time + 1, s + 2)
                # otherwise, use usual update
                else:
                    futureValue = sum([self.transProb(
                        self.time, s, newS) * futureAlphas[o][newS] for newS in self.SPO])
                # update alpha with obsProb*(reward + futureValue)
                value = self.obsProb(
                    self.time, s, o) * (self.reward(self.time, s, M, o) + futureValue)
                alpha[s] += value
        return alpha

    def bestFalsePosAlpha(self, t):
        # find best alpha vector from time step t+1, if get biopsy and comes is
        # cancer free
        alphaMaxValue = None
        for _, alpha in self.alpha[t + 1]:
            value = sum([self.transProb(t, 0, newS)
                         * alpha[newS] for newS in self.SPO])
            if not alphaMaxValue or value > alphaMaxValue:
                alphaMaxValue = value
        return alphaMaxValue

    ##############################################################
        # Pruning Steps #
    ##############################################################

    def eagleReduction(self):
        M = 1000000
        alphas = np.array([M * np.array(val)
                           for _, val in self.alpha[self.time]])
        marked = np.ones(alphas.shape[0], dtype=bool)
        for i, c in enumerate(alphas):
            if marked[i]:
                # Preserve non-dominated parts
                marked[marked] = np.logical_or(
                    np.any(alphas[marked] > c, axis=1), np.all(alphas[marked] >= c, axis=1))

        self.alpha[self.time] = self.alpha[self.time][marked]

    def monahanElimination(self):
        if len(self.alpha[self.time]) == 1:
            return
        alphas = np.array([np.array(val) for _, val in self.alpha[self.time]])
        marked = np.ones(len(alphas)).astype(bool)

        for i in xrange(0, len(alphas)):
            marked[i] = self.pruneLP(i, alphas, marked, self.LPSolver)
        self.alpha[self.time] = self.alpha[self.time][marked]

    def pruneLP(self, i, alphas, marked, solver):
        if solver == "cvxopt":
            return pruneLPCvxopt(self, i, alphas, marked)
        elif solver == "cplex":
            return pruneLPCplex(self, i, alphas, marked)
        elif solver == "pulp":
            return pruneLPPulp(self, i, alphas, marked)
        elif solver == "checkAll":
            o1 = pruneLPCvxopt(self, i, alphas, marked, returnObj=True)
            o2 = pruneLPCplex(self, i, alphas, marked, returnObj=True)
            o3 = pruneLPPulp(self, i, alphas, marked, returnObj=True)
            s1 = o1 > 0
            s2 = o2 > 0
            s3 = o3 > 0
            if s1 != s2 or s2 != s3 or s3 != s1:
                print "Cvxopt {0}, Cplex {1}, Pulp {2}".format(o1, o2, o3)
            return s1
        else:
            raise "Invalid LP solver given"

    def setLPSolver(self, solver):
        self.LPSolver = solver

    ##############################################################
        # Saving alpha vectors #
    ##############################################################

    def writeAlpha(self, fname="alpha/alpha.txt"):
        with open(fname, "w") as f:
            pickle.dump(self.alpha, f)

    def readAlpha(self, fname="alpha/alpha.txt"):
        with open(fname, "r") as f:
            self.alpha = pickle.load(f)
        self.time = max(min(self.alpha.keys()) - 1, 0)

    ##############################################################
        # Making decisions based on alpha vectors #
    ##############################################################

    def chooseAction(self, b, t):
        # compute action that gives best value
        bestAction = None
        bestValue = None
        for action, alpha in self.alpha[t]:
            value = np.dot(b, alpha)
            if bestValue is None or value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
