"""
Optimizers.

Todo:
as more predictors are finished, add optimizers for them (e.g. kNN, RBM)
"""
import numpy as N

def sequentialSVDOptimizer(s, paramsearchdist = 0.003, boundarysearchdist = 0.01, boundaryparams = ['umin', 'umax', 'mmin', 'mmax']):
	"""
	USAGE:
	sequentitalSVDOptimizer(s) # yes, that's it.

	Optimizes each parameter thoroughly with respect to all others.
	In other words, we don't move on to the next parameter until the first one 
	has a local minimum.
	
	The only difference between the serial and the sequential optimizer is the relative
	placement of the for loop over all parameters and the while loop requiring 
	parameters to be optimized.
	"""
	import os
	curdir = os.getcwd()
	finished = [False]*len(s.parameters)
	direction = [0]*len(s.parameters)
	# Get a baseline RMSE
	print('=========================================================')
	print('Getting a baseline RMSE')
	print('=========================================================')
	s.svd()
	baselineRMSE = N.min(s.RMSEhistory)
	print('\n\nBaseline RMSE: %f' % baselineRMSE)
	os.chdir(curdir)
	

	# for each parameter
	for i, param in enumerate(s.parameters):
		while finished[i] is not True:
			
			currentvalue = getattr(s, param)
			
			if param not in boundaryparams:
				# build a search range
				if direction[i] == 0: # no direction picked yet
					paramrange = N.r_[currentvalue-paramsearchdist, currentvalue+paramsearchdist]
				elif direction[i] == -1: # searching down
					paramrange = N.r_[currentvalue - paramsearchdist, currentvalue - 2*paramsearchdist]
				elif direction[i] == 1: # searching up
					paramrange = N.r_[currentvalue + paramsearchdist, currentvalue + 2*paramsearchdist]
					
				# make sure it's all positive
				paramrange = paramrange[paramrange>0]
				if len(paramrange) == 0:
					print('\n\nNo valid values left for parameter %s, so we must move on!' % param)
					finished[i] = True
					continue
				
			else: 
				# build a search range
				if direction[i] == 0: # no direction picked yet
					paramrange = N.r_[currentvalue - boundarysearchdist, currentvalue + boundarysearchdist]
				elif direction[i] == -1: # searching down
					paramrange = N.r_[currentvalue - boundarysearchdist, currentvalue - 2*boundarysearchdist]
				elif direction[i] == 1: # searching up
					paramrange = N.r_[currentvalue + boundarysearchdist, currentvalue + 2*boundarysearchdist]
				
			print('\n\nOptimizing parameter %s, currently set at %f.\nBaseline RMSE: %f' % (param, currentvalue, baselineRMSE))
			if direction[i] == 0: 
				print('Searching below and above current parameter value')
			elif direction[i] == -1:
				print('Searching below current parameter value')
			elif direction[i] == 1:
				print('Searching above current parameter value')
			
			# Okay, we've got the parameters we want to try. Time to try them!
			tmpError = []
			for trythisparamvalue in paramrange:
				setattr(s, param, trythisparamvalue)
				s.svd()
				os.chdir(curdir) # always go back to the directory we started in, or there's a risk of weird file placement
				tmpError.append(N.min(s.RMSEhistory))
				
			# Let's see how well the different parameters did.
			# If none are less than the baseline
			if N.min(tmpError) > baselineRMSE:
				print('\n\nCurrent value %f for parameter %s is the optimum! Moving on...' % (currentvalue, param))
				setattr(s, param, currentvalue) # restore the old value
				finished[i] = True # we're done here...
			else:
				bestvalue = paramrange[N.argmin(tmpError)]
				setattr(s, param, bestvalue) # stick in the new best value
				baselineRMSE = N.min(tmpError) # update the baselineRMSE with the new best RMSE
				print('Found a new optimum for parameter %s! Value: %f\n. New baseline RMSE: %f' % (param, bestvalue, baselineRMSE))
				# If we haven't yet picked a direction, PICK ONE
				if direction[i] == 0:
					if bestvalue > currentvalue: 
						direction[i] = 1
					elif bestvalue < currentvalue:
						direction[i] = -1