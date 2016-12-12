'''
	Attempt at approximate PBVI solver (unfinished)
	Todo: work on time dependence 
'''
import itertools
import time
import os
import pickle
from random import choice 

import numpy as np

from model import CancerPOMDP
from modelConstants import W, M, MNEG, MPOS, SDNEG, SDPOS
from pruneLP import pruneLPPulp, pruneLPCplex, pruneLPCvxopt

class PerseusSolve(CancerPOMDP): 

	def __init__(self, *args, **kwargs): 
		CancerPOMDP.__init__(self. *args, **kwargs)

		# initialize list of belief states
		self.B = []

		# initialize alphas
		self.alphas = None
		with open('alpha/Simple/alpha80.txt') as f: 
			self.alphas = pickle.loads(f)

		# initialize V: list of alpha vectors 
		self.V = []

	def value(self, belief): 
		'''
			return value for a given belief
		'''
		return max([np.dot(belief, a) for a in self.V])

	def collectBeliefs(self, totalBeliefs): 
	'''
		run "trials of exploration" in belief space 
		sample action / exploration and add it to our set of beliefs until we
		hit the number of totalBeliefs we want 
	'''
		return 

	def backup(self, val, belief):
	'''
		generate a new alpha vector
		for a given value and belief state 
	''' 
		# get all the alphas for an action and observation 

		ab_alphas = [CancerPOMDP.reward()]

		return 

	def isEqual(b1, b2): 
		'''
			compare 
		'''
		return all([x1 == x2 for (x1, x2) in zip(b1, b2)])

	def computeValueFunc(self, numIter):
		V = self.V 
		# q: set vs. list
		# q: likelihood of repeated alpha vectors / belief states
		for _ in numIter:  
			B_new = self.B 
				V_new = []
				# while B is not empty 
				while B_new: 
					
					b = choice(B_new)

					# get alpha vector for belief and value 
					alpha = self.backup(V, b)

					alpha_b = None

					if np.dot(alpha, b) >= self.value(V, b): 
						B_new = [bp for bp in B_new if np.dot(alpha, bp) < self.value(V, bp)]
						alpha_b = alpha 
					else: 
						B_new = [bp for bp in B_new if bp.isEqual(b)]
						alpha_b = max([(alpha, np.dot(alpha, b)) for alpha in V], key = lambda t: t[1])[0]
				V_new.append(alpha_b)
			V = V_new 
		return V_new 
