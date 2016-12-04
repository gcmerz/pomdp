import pulp
import numpy as np

class MonahanSolve:

    def __init__(self, cancerModel):
        self.pomdp = cancerModel
        self.alpha = { t: np.array([]) for t in range(self.pomdp.t0, self.pomdp.tmax)}
        self.time = self.pomdp.t0

    def generateWAlpha(self):
        def alpha(state, alpha1, alpha2):
            obsAlpha = lambda (state, ): self.pomdp.obsProb(self.time, state, obs)*(self.pomdp.reward(self.time, state, 0, obs) + sum([self.pomdp.transProb(time, state, newState) for newState in self.pomdp.S])
            sum([obsAlpha() for obs in self.pomdp.O[0]])

        for alpha1 in self.alpha[self.time + 1]:
             for alpha2 in self.alpha[self.time + 1]:
                 newAlpha = [state for state in self.pomdp.SPO]

    def generateMAlpha(self, time, previousAlpha):
        return 1

    def pruneLP(self, alpha, otherAlpha, printOut = False):
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