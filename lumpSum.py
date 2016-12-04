
def lumpSum(t):
    
    numPeople = 100000
    decayRate = .008 
    laterDecay = .006

    initPeople = numPeople
    yearsTotal = 0 
    for i in xrange(t, 121): 
        if i > 10: 
            decayRate = laterDecay
        numDied = decayRate * numPeople
        # find number of people that lived the full 6 mos
        yearsTotal += .5 * (numPeople - numDied) + .25 * (numDied)
        numPeople -= numDied
    
    return yearsTotal / float(initPeople)
