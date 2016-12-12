from stats import SDStats, MStats, TDPMatrix, death_probs, healthy_inv_probs, TMatrix
import numpy as np
from modelConstants import W, M, MNEG, MPOS, SDNEG, SDPOS


class CancerPOMDP(object):

    def __init__(self, t0=0, tmax=120, timeDepProbs = False):
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
        # if we want time dependent transition probabilities
        self.tdp= timeDepProbs
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

        def lumpSum(t, state):
            numPeople = 100000

            initPeople = numPeople
            yearsTotal = 0
            for i in xrange(t, self.tmax):
                # probability of dying
                decayRate = self.transProb(t, 0, 5)
                if state == 4:
                    # update decay rate after first 5 years to reflect changes
                    # in probability of death
                    if i - t > 10:
                        decayRate *= 1.2
                    else:
                        decayRate *= 1.5
                numDied = decayRate * numPeople
                # find number of people that lived the full 6 mos
                yearsTotal += .5 * (numPeople - numDied) + .25 * (numDied)
                numPeople -= numDied

            return yearsTotal / float(initPeople)

        if state == 0:
            return lumpSum(time, 0) + self.terminalReward(state)
        if state == 3:
            # death rate for in-situ cancer same as cancer-free
            return lumpSum(time, 3) + self.terminalReward(state)
        # once you have survived five years with invasive cancer,
        # your probability of dying decreases
        if state == 4:
            return lumpSum(time, 4) + self.terminalReward(state)

    def setupTDPMatrix(self, time):
        # determine age
        age = self.t0 + time / 2
        # subtract 1 if on last timestep so probabilities work out
        if age == 100:
            age -= 1

        # mortality rates given for 5 year intervals
        ageIndex5Year = (age - 40) / 5

        # incidence probabilities given for 10 year intervals
        ageIndex10Year = (age - 40) / 10

        transMatrix = TDPMatrix
        # set healthy -> death prob
        transMatrix[0][5] = death_probs[ageIndex5Year]

        # set in-situ -> death prob
        transMatrix[1][5] = death_probs[ageIndex5Year]

        # set healthy -> invasive prob
        transMatrix[0][2] = healthy_inv_probs[ageIndex10Year]

        # set healthy -> healthy prob
        transMatrix[0][0] = 1 - (transMatrix[0][1] +
                                 transMatrix[0][2] + transMatrix[0][5])

        # set in-situ -> in-situ prob
        transMatrix[1][1] = 1 - (transMatrix[1][2] + transMatrix[1][5])

        return transMatrix

    def transProb(self, time, state, newState):
        '''
            Return probability of transitioning from state to newState at
            time t
        '''
        if self.tdp: 
            return self.setupTDPMatrix(time)[state][newState]
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
        tauS = lambda newState: sum(
            [b[s] * self.obsProb(time, s, obs) * self.transProb(time, s, newState)
             for s in self.SPO])

        # if false positive mammography, then know that you are cancer free
        if obs == MPOS:
            tau = np.array([self.transProb(time, 0, newState) for newState in self.SPO])
        else:
            tau = np.array([tauS(newState) for newState in self.SPO])

        # return normalized tau
        return 1.0*tau / np.sum(tau)
