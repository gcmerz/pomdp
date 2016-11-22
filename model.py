from stats import SDStats, MStats


class CancerPOMDP:

    def __init__(self, b0=[1, 0, 0], t0=0):
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
        self.A = range(2)
        self.O = range(4)

        # initial risks given
        self.b0 = b0
        # initial age given
        self.t0 = t0
        self.tmax = 120

    def reward(self, time, state, action, obs):
        '''
            Reward at time, given state, action and observation
            Incorporates life expectancy between time t and t+1
            and disutils associated with a mammogram that occurs
            in that time interval.
        '''

        # if positive mammography and patient has cancer, then lump sum
        if obs == 1 and (state == 1 or state == 2):
            # return 0 or lump sum here...
            return 0

        deathRate = self.transProb(time, state, 5)
        reward = 0.5 * (1 - deathRate) + 0.25 * deathRate
        # if mamography, then subtract disutility of performing mammography
        if action == 1:
            reward -= self.__disutil(obs, state)
        return reward

    def __disutil(self, obs, state):
            '''
                Disutilities of mamography based on resulting observation
                and underlying disease state
            '''
            # negative mammography, 0.5 days du
            if obs == 0:
                return 0.5 / 365
            if obs == 1:
                # false positive mammography, 4 weeks du
                if state == 0:
                    # 4 weeks
                    return 4.0 / 52
                # true positive mammography, 2 weeks du
                return 2.0 / 52

    def transProb(self, time, state, newState):
        '''
            Probability of transitioning from state to newState at
            time t.
        '''
        return 0.5

    def obsProb(self, time, state, obs):
        '''
            Return observation probabilities as
            given by specificity and sensitivity rates from paper.
        '''

        # if cancer-free then use specificty, otherwise use sensitivity
        stat = "spec" if state == 0 else "sens"

        # determine ageGroup
        if time < 20:
            ageGroup = 0
        elif time < 30:
            ageGroup = 1
        elif time < 40:
            ageGroup = 2
        elif time < 60:
            ageGroup = 3
        else:
            ageGroup = 4

        # return probability
        if obs == 0:
            return MStats[ageGroup][stat]
        if obs == 1:
            return 1 - MStats[ageGroup][stat]
        if obs == 2:
            return SDStats[stat]
        if obs == 3:
            return 1 - SDStats[stat]
