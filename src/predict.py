import numpy as N

def readtestfile(probefile):
	"""
	mids, uids = readtestfile('probe.txt', fd)
	
	Read a txt probe file and convert the movie and user IDs
	to use a zero-index.
	
	We require a FlixData class instance (fd) because it's the 
	easiest way (in my current setup) to zero out user IDs.
	"""
	from flixdata import FlixData, readProbeFile
	fd = FlixData(loaddata=False)
	data = readProbeFile(probefile)
	print('Re-indexing the data file...')
	data[:,0] -= 1 # zero-index the movies
	data[:,1] = fd.zeroIndexUserIDs(data[:,1])
	
	return data[:,0], data[:,1]
	

def predict(probefile, U, M):
	"""
	Very simple function--
	Take the movie IDs and user IDs that match up with
	user and movie feature matrices, U and M, 
	and make simple dot-product predictions with them.
	
	It's helpful if the IDs are in the order of the probe.
	Makes things simpler later on.
	"""
	
	mids, uids = readtestfile(probefile)
	
	numratings = mids.shape[0]
	ratings = N.zeros(numratings, dtype='float32')
	
	for i in range(numratings):
		if i % 100000 == 0: print('Rating %d' % i)
		ratings[i] = N.sum(U[uids[i]]*M[mids[i]])
	
	return ratings
	
def writeprediction(mids, ratings, probefile):
	"""
	Write the predictions, held in the ratings array
	into a probe file.
	Clip predictions to fit between 1.0 and 5.0
	"""
	f = open(probefile, 'wt')
	
	numratings = mids.shape[0]
	ratings[ratings<1.0] = 1.0
	ratings[ratings>5.0] = 5.0
	
	if min(mids) == 0: mids += 1
	
	movieid = 0
	
	for i in range(numratings):
		if i % 100000 == 0: print('Rating %d' % i)
		
		if mids[i] == movieid:
			f.write('%f\n' % ratings[i])
		else:
			movieid = mids[i]
			f.write('%d:\n' % movieid)
			f.write('%f\n' % ratings[i])
	
			