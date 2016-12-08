from stats import SDStats, MStats, TMatrix
import numpy as np
from modelConstants import W, M, MNEG, MPOS, SDNEG, SDPOS


class CancerPOMDP(object):

    def __init__(self, t0=0, tmax=120):
        '''
        States (S):
            0-5 as described in paper

        Actions (A):
            0 -> Wait
            1 -> Mammography

        Observations (O):
            0 -> M-
            1 -> M+
            2 -> SD-
            3 -> SD+
        '''

        self.S = range(6)
        # partially observable states (0, 1, 2)
        self.SPO = range(3)
        self.A = [W, M]
        self.O = {M: [MNEG, MPOS], W: [SDNEG, SDPOS]}

        # initial age given
        self.t0 = t0
        # age to end at
        self.tmax = tmax

    def terminalReward(self, state):
        '''
            Reward received for being in state at
            last time step in model.
        '''
        return 0

    def reward(self, time, state, action, obs):
        '''
            Reward at time, given state, action and observation
            Incorporates life expectancy between time t and t+1
            and disutils associated with a mammogram that occurs
            in that time interval.
        '''
        deathRate = self.transProb(time, state, 5)
        reward = 0.5 * (1 - deathRate) + 0.25 * deathRate
        # if mamography, then subtract disutility of performing mammography
        if action == M:
            reward -= self.__disutil(obs, state)
        return reward

    def __disutil(self, obs, state):
        '''
            Disutilities of mamography based on resulting observation
            and underlying disease state
        '''
        # negative mammography, 0.5 days du
        if obs == MNEG:
            return 0.5 / 365
        if obs == MPOS:
            # false positive mammography, 4 weeks du
            if state == 0:
                # 4 weeks
                return 4.0 / 52
            # true positive mammography, 2 weeks du
            return 2.0 / 52

    def lumpSumReward(self, time, state):
        '''
            Lump sum reward at time, given state (either in situ or invasive)
            Decision process ends after receiving this lump sum reward and it
            should represent expected QUALYs given being in treatment for in situ
            or invasive cancer.
        '''

        def lumpSum(t, decayRate, laterDecay):
            numPeople = 100000

            initPeople = numPeople
            yearsTotal = 0
            for i in xrange(t, self.tmax):
                # update decay rate after first 5 years to reflect changes
                # in probability of death
                if i > 10:
                    decayRate = laterDecay
                numDied = decayRate * numPeople
                # find number of people that lived the full 6 mos
                yearsTotal += .5 * (numPeople - numDied) + .25 * (numDied)
                numPeople -= numDied

            return yearsTotal / float(initPeople)

        if state == 0:
            return lumpSum(time, .004, .004) + self.terminalReward(state)
        if state == 3:
        # death rate for in-situ cancer same as cancer-free
            return lumpSum(time, .004, .004) + self.terminalReward(state)
        # once you have survived five years with invasive cancer,
        # your probability of dying decreases
        if state == 4:
            return lumpSum(time, .008, .006) + self.terminalReward(state)

    def transProb(self, time, state, newState):
        '''
            Return probability of transitioning from state to newState at
            time t
        '''
        return TMatrix[state][newState]

    def obsProb(self, time, state, obs):
        '''
            Return observation probability given state as
            given by specificity and sensitivity rates from paper.
        '''

        # if cancer-free then use specificity, otherwise use sensitivity
        stat = "spec" if state == 0 else "sens"

        # determine ageGroup
        ageGroups = [20, 30, 40, 60, self.tmax + 1]
        for group, ageUpper in enumerate(ageGroups):
            if time < ageUpper:
                ageGroup = group
                break

        # return probability of observation | state
        if obs == MNEG:
            return MStats[ageGroup][stat]
        if obs == MPOS:
            return 1 - MStats[ageGroup][stat]
        if obs == SDNEG:
            return SDStats[stat]
        if obs == SDPOS:
            return 1 - SDStats[stat]

    def updateBeliefState(self, time, b, obs):
        '''
            Update belief state given observation and transition probability
        '''

        # if false positive mammography, then know that you are cancer free
        if obs == MPOS:
            return np.array([self.transProb(time, 0, newState) for newState in self.SPO])

        tauS = lambda newState: sum(
            [b[s] * self.obsProb(time, s, obs) * self.transProb(time, newState, s)
             for s in self.SPO])

        tau = np.array([tauS(newState) for newState in self.SPO])
        # return normalized tau
        return tau / np.sum(tau)
