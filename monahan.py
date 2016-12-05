import itertools
import numpy as np
import pulp
from model import CancerPOMDP
import time


class MonahanSolve(CancerPOMDP):

    def __init__(self, *args, **kwargs):
        CancerPOMDP.__init__(self, *args, **kwargs)
        self.alpha = {}
        # receive 0 reward at last time step (alphas are stored as tuples of
        # actions and vectors)
        self.alpha[self.tmax] = np.array([
            (-1, np.array([self.terminalReward(s) for s in self.SPO]))], dtype=tuple)
        self.time = self.tmax - 1

    def solve(self):
        while self.time >= self.t0:

            # generate all alpha
            start = time.time()
            self.generateAllAlpha()
            end = time.time()
            print "Total alpha enumerated: ", len(self.alpha[self.time]), ", time: ", "{:.2f}".format(end - start)

            # prune dominated alpha
            start = time.time()
            self.eagleReduction()
            end = time.time()
            print "Total alpha after Eagle: ", len(self.alpha[self.time]), ", time: ", "{:.2f}".format(end - start)

            # use LP to prune
            start = time.time()
            self.monahanElimination()
            end = time.time()
            print "Total alpha after LP: ", len(self.alpha[self.time]), ", time: ", "{:.2f}".format(end - start)
            print "Completed time step ", self.time, "\n"
            self.time -= 1


    def eagleReduction(self):
        def dominates(alpha, alphaOther):
            return np.greater_equal(alpha, alphaOther).all()

        alphas = self.alpha[self.time]
        marked = np.ones(alphas.shape[0]).astype(bool)
        for i in xrange(len(alphas)):
            if marked[i]:
                for j in xrange(i+1, len(alphas)):
                    if dominates(alphas[j][1], alphas[i][1]):
                        marked[i] = False
                        break
                    elif dominates(alphas[i][1], alphas[j][1]):
                        marked[j] = False
        self.alpha[self.time] = alphas[marked]

    def monahanElimination(self):
        alphas = np.array([val for _, val in self.alpha[self.time]])
        marked = np.ones(alphas.shape[0]).astype(bool)
        if len(self.alpha[self.time]) == 1:
            return
        for i in xrange(0, len(alphas)):
            marked[i] = self.pruneLP(alphas[i], alphas)

    def generateAllAlpha(self):
        # find alpha to use to get maximal value if false positive mammogram
        # (only do this once for given timestep)
        alphaMaxValue = self.bestFalsePosAlpha()

        # set up permutations of 2 future alpha to correspond with generated
        # alpha vectors
        perms = itertools.product(
            self.alpha[self.time + 1], self.alpha[self.time + 1])
        self.alpha[self.time] = np.array(
            [(i % 2, np.array([0 for _ in self.SPO]))
             for i in xrange(2 * len(self.alpha[self.time + 1])**2)], dtype=tuple)

        i = 0
        for alpha1, alpha2 in perms:
            # alphas for the  next step each associated with an observation
            futureAlphas = {2: alpha1[1], 3: alpha2[1]}
            self.alpha[self.time][i] = (0, self.generateWAlpha(futureAlphas))
            # alphas for the  next step each associated with an observation
            futureAlphas = {0: alpha1[1], 1: alpha2[1],
                            "max": alphaMaxValue}
            self.alpha[self.time][
                i + 1] = (1, self.generateMAlpha(futureAlphas))
            i += 2

    def generateWAlpha(self, futureAlphas):
        # initialize alpha as vector of partially observable states
        alpha = [0 for _ in self.SPO]
        # compute alpha vector for each state
        for s in self.SPO:
            for o in self.O[0]:
                # future value given the two alpha vectors associated with
                # observations
                futureValue = sum([self.transProb(
                    self.time, s, newS) * futureAlphas[o][newS] for newS in self.SPO])
                value = self.obsProb(
                    self.time, s, o) * (self.reward(self.time, s, 0, o) + futureValue)
                alpha[s] += value
        return alpha

    def bestFalsePosAlpha(self):
        alphaMaxValue = None
        for _, alpha in self.alpha[self.time + 1]:
            value = sum([self.transProb(self.time, 0, newS)
                         * alpha[newS] for newS in self.SPO])
            if not alphaMaxValue or value > alphaMaxValue:
                alphaMaxValue = value
        return alphaMaxValue

    def generateMAlpha(self, futureAlphas):
         # initialize alpha as vector of partially observable states
        alpha = [0 for _ in self.SPO]
        # compute alpha vector for each state
        for s in self.SPO:
            for o in self.O[1]:
                # if false positive, then know you do not have cancer, so know
                # next belief state and can choose maximal alpha
                if o == 1 and s == 0:
                    alpha[s] += futureAlphas["max"]
                # otherwise, use usual update
                else:
                    futureValue = sum([self.transProb(
                        self.time, s, newS) * futureAlphas[o][newS] for newS in self.SPO])
                    value = self.obsProb(
                        self.time, s, o) * (self.reward(self.time, s, 1, o) + futureValue)
                    alpha[s] += value
        return alpha

    def pruneLP(self, alpha, otherAlpha, printOut=False):
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
        for a in otherAlpha:
            prob += np.dot(alpha - a, pi) - sigma >= 0, ""
        prob.solve()
        obj = pulp.value(prob.objective)
        if printOut:
            print "Objective (sigma): ", obj
            print "Pi: ", [p.value() for p in pi]
        # return True if should not prune, False if should prune
        return obj > 0
