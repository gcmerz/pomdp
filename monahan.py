import itertools
import time
from collections import deque

import numpy as np
import pulp

from cvxopt.modeling import variable, op, dot
from cvxopt import matrix
import cvxopt

from model import CancerPOMDP
from modelConstants import W, M, MNEG, MPOS, SDNEG, SDPOS


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

    def solve(self):
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
            print "LP construct time: ", self.constructTime
            print "LP solve time: ", self.solveTime
            self.printReport("LP", start, end)

            print "Completed time step ", self.time, "\n"
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
        def dominates(alpha, alphaOther):
            return np.greater_equal(alpha, alphaOther).all()

        alphas = self.alpha[self.time]
        marked = np.ones(alphas.shape[0]).astype(bool)
        for i in xrange(len(alphas)):
            if marked[i]:
                for j in xrange(i + 1, len(alphas)):
                    if marked[j]:
                        if dominates(alphas[j][1], alphas[i][1]):
                            marked[i] = False
                            break
                        elif dominates(alphas[i][1], alphas[j][1]):
                            marked[j] = False
        self.alpha[self.time] = alphas[marked]

    def monahanElimination(self):
        alphas = np.array([np.array(val) for _, val in self.alpha[self.time]])
        marked = np.ones(len(alphas)).astype(bool)
        if len(self.alpha[self.time]) == 1:
            return
        for i in xrange(0, len(alphas)):
            marked[i] = self.pruneLP(i, alphas, marked)
        self.alpha[self.time] = self.alpha[self.time][marked]

    def pruneLP(self, i, alphas, marked, printOut=False):
        sigma = variable()
        pi = variable(3)
        c1 = ( sum(pi) == 1 )
        c2 = ( pi >= 0 )
        diffs = matrix(np.array([(alphas[i] - a) for j, a in enumerate(alphas) if marked[j] and j != i]).transpose()) 
        c3 = ( dot(pi, diffs) - sigma >= 0 )
        lp = op(-1*sigma, [c1, c2, c3])
        cvxopt.solvers.options['show_progress'] = False
        lp.solve()
        val = lp.objective.value()
        # obj = val[0]
        print val

        # # alpha to check if prune
        # alpha = alphas[i]
        # start = time.time()
        # # problem is a maximization problem
        # prob = pulp.LpProblem("Reduce", pulp.LpMaximize)
        # # set up arbitrary probability distribution over state pi
        # pi = np.array([pulp.LpVariable("pi" + str(s), lowBound=0)
        #                for s in range(3)])
        # # set up sigma
        # sigma = pulp.LpVariable("sigma", lowBound=None, upBound=None)
        # # Objective: max sigma
        # prob += sigma, "Maximize sigma"
        # # The pi must sum to 1
        # prob += pulp.lpSum(pi) == 1, "ProbDistribution"
        # # Check if this alpha vector does better than any other alpha vector
        # for j, a in enumerate(alphas):
        #     if i != j and marked[j]:
        #         prob += np.dot(alpha - a, pi) - sigma >= 0, ""
        # end = time.time()
        # self.constructTime += (end - start)

        # start = time.time()
        # prob.solve()
        # end = time.time()
        # self.solveTime += (end - start)

        # obj = pulp.value(prob.objective)
        # if printOut:
        #     print "Problem: ", prob
        #     print "Objective (sigma): ", obj
        #     print "Pi: ", [p.value() for p in pi]
        # # return True if should not prune, False if should prune
        # print a
        # c = matrix([-1.0, 0.0, 0.0, 0.0])
        # b = matrix(np.zeros((1, len(aVals))))
        # print "A", A
        # print "b", b
        # print "c", c
        # sol = solvers.lp(c, A, b)
        return 1

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
