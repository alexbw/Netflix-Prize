# Now is ze time on Shprockets ver vee BLEND
import numpy as N

def simpleblend(modellist, td, qd, param=5, resultdir='/home/alex/workspace/flix/results/'):
	"""
	USAGE:
	modellist = ['12featureBRISMF', '40featureSPUMF']
	td = flixdata.FlixData('../data/probe10')
	blendedratings = blend.simpleblend(modellist, td.userRatings, blender=blend.ridgeregression)
	
	Make a blend folder, call it blend_date,
	stick in the qualifying file,
	along with a little logfile detailing the model names
	and their blending weights, along with the technique used to blend.
	"""
	from os import makedirs
	from os.path import join
	from time import gmtime
	# Pull out the data we need from td and qd, the FlixData instances with 
	# the probe10 and qualifying data, respectively
	probevals = td.userRatings
	movieIDs  = qd.movieIDs+1 # add 1, because they're zero-indexed
	
	# Make the blend
	probe10ratings, quizratings = getModelPredictions(modellist)
	blendproportions = ridgeregression(probe10ratings, probevals, param)
	probeblend = N.sum(probe10ratings*blendproportions, 1)
	finalblend = N.sum(quizratings*blendproportions, 1)
	
	# Figure out how good it is on the probe
	rmse = N.sqrt(N.sum((probeblend-probevals)**2)/N.float(probevals.size))
	
	# Give a dog a home
	date = gmtime()
	blendname = 'blend_%d_%d-%d_%dh%02d' % date[:5]
	blenddir = join(resultdir, blendname)
	makedirs(blenddir)
	
	# Put said dog in aforementioned home
	from predict import writeprediction
	predictionfile = join(blenddir, blendname+'.txt')
	writeprediction(movieIDs, finalblend, predictionfile)
	
	# Remorselessly gunzip the dog
	import gzip
	f = open(predictionfile, 'rb')
	fout = gzip.open(predictionfile+'.gz', 'wb')
	fout.writelines(f)
	fout.close()
	f.close()
	
	# Produce a little log file with some details
	logfile = open(join(blenddir, 'log.txt'), 'wt')
	logfile.write('Num models blended: %d\n' % len(modellist))
	logfile.write('Blending technique: %s\n' % 'ridgeregression')
	logfile.write('Probe 10 RMSE: %f\n' % rmse)
	
	logfile.write('\nModels:\n')
	for i in range(len(modellist)):
		logfile.write('%f\t%s\n' % (blendproportions[i], modellist[i]))
		
	logfile.close()
	
	

def getModelPredictions(modellist, resultdir = '/home/alex/workspace/flix/results/', loadquiz=True):
	"""
	# For a list of models,
	modellist = ['12featureBRISMF', '40featureSPUMF', '1000featureBRISMF']
	# load in in the probe10 predictions, 
	# and if loadquiz=True, load the quiz predictions too
	probe10ratings, quizratings = getModelPredictions(modellist, loadquiz=True)
	"""
	from os.path import join
	
	probe10ratings = []
	quizratings = []
	
	for model in modellist:
		modeldir = join(resultdir, model)
		probefile = join(modeldir, 'probe10ratings')
		quizfile = join(modeldir, 'quizratings')
		probe10ratings.append(N.load(probefile))
		if loadquiz:
			quizratings.append(N.load(quizfile))
	
	if loadquiz:
		return N.asarray(probe10ratings).T, N.asarray(quizratings).T
	else:
		return N.asarray(probe10ratings).T
	
def ridgeregression(ratings, probevals, gamma=1):
	"""
	USAGE:
	x = ridgeregression(ratings, probevals gamma=1)
	
	The blending weights of each prediction set are in x.
	
	ratings   - a NumPy array of predictions of a probe set
	probevals - the true ratings of the probe set
	gamma	  - the regularization parameter.
	"""
	from numpy.linalg import inv
	# Format the ratings (from tuple or NumPy array) into a matrix
	# and make sure they're the right orientation
	A = N.matrix(ratings)
	numratings, nummodels = A.shape
	if numratings < nummodels:
		numratings, nummodels = nummodels, numratings
		A = A.T
		
	# make sure probevals is numratings x 1
	if len(probevals.shape)>1:
		numproberatings, shouldequalone = N.shape(probevals)

		if shouldequalone != 1:
			if numproberatings == 1:
				probevals = probevals.T
			else:
				raise ValueError, 'The probe array must have the same number of ratings as the prediction array'
	
	# The muscle of the ridge regression.
	# This is straight from Wikipedia
	gamma = N.matrix(N.eye(nummodels))*gamma
	x = N.dot(inv(A.T*A + gamma*gamma.T) * A.T, probevals)
	
	return N.ravel(x)
	
def linreg(ratings, probevals):
	"""
	USAGE:
	x = linreg(ratings, probevals)
	
	ratings   - a NumPy array of predictions of a probe set
	probevals - the true ratings of the probe set
	
	This is equivalent to ridge regression, where gamma = 0
	"""
	from numpy.linalg import lstsq
	# Format the ratings (from tuple or NumPy array) into a matrix
	# and make sure they're the right orientation
	A = N.matrix(ratings)
	numratings, nummodels = A.shape
	if numratings < nummodels:
		numratings, nummodels = nummodels, numratings
		A = A.T
	# make sure probevals is numratings x 1
	numproberatings, shouldequalone = probevals.shape
	if shouldequalone != 1:
		if numproberatings == 1:
			probevals = probevals.T
		else:
			raise ValueError, 'The probe array must have the same number of ratings as the prediction array'

	# The muscle of the function is pretty simple.
	x, resids, rank, s = lstsq(A, probevals)
	
	return x
	
def readquizfile(filename):
	f = open(filename, 'rt')
	data = f.readlines()
	ratings = []
	for i in range(len(data)):
		if ':' not in data[i]:
			ratings.append(N.float(data[i][:-1]))
	return N.asarray(ratings)