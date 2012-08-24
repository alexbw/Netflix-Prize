cimport cython
cimport numpy as N
import numpy as N

ctypedef N.float64_t f64_t
ctypedef N.float32_t f32_t
ctypedef N.int8_t i8_t
ctypedef N.int16_t i16_t
ctypedef N.int32_t i32_t
ctypedef N.int64_t i64_t

cdef extern from "stdlib.h":
	ctypedef unsigned long size_t
	void free(void *ptr)
	void *malloc(size_t size)
	void *realloc(void *ptr, size_t size)
	size_t strlen(char *s)
	char *strcpy(char *dest, char *src)


class SVD(object):
	"""
	SVD is a class wrapping functionality implementing various matrix factorization algorithms,
	currently all of which are flavors of singular-value decompositions. They differ mostly in
	hard constraints of feature values (e.g. non-negativity) and the inclusion of bias features.
	
	Usage:
	# first, build FlixData class data containers. 
	# These are described and implemented in flixdata.py
	> modelname = '40fBRISMF'
	s = svd.SVD(fd, td, modelname)
	# Then, set the various parameters of the model
	> s.numfeatures = 40; s.lrateU = 0.007; s.lrateM = 0.005 #... etc
	
	The matrices and logging information will be saved into a folder at the current directory,
	named after modelname.
	"""
	
	def __init__(self, fd, td, qd=None, modelname='untitled_model'):
		self.fd = fd # Full data FlixData class
		self.td = td # Test data FlixData class (Gravity's probe10)
		if qd is not None:
			self.qd = qd # Quiz data FlixData class
		
		self.numratings = fd.numratings
		self.nummovies = 17770 # hard-coded for now...
		self.numusers = fd.numusers
		self.modelname = modelname
		
		self.parameters = ['lrateU', 'lrateM', 'lrateUb', 'lrateMb', 'dampfactU', \
							'dampfactM',  'dampfactUb', 'dampfactMb', 'umin', 'umax', \
							'mmin', 'mmax']
		self.lrateU = 0.019
		self.lrateM = 0.004
		self.dampfactU = 0.019
		self.dampfactM = 0.019
		self.lrateUb = 0.004
		self.lrateMb = 0.013
		self.dampfactUb = 0.019
		self.dampfactMb = 0.007
		self.umin = -0.01
		self.umax = 0.01
		self.mmin = -0.01
		self.mmax = 0.01
		
		
		
		self.properties = ['numfeatures', 'miniters', 'maxepochs', 'seed', 'algorithm', \
							'keepUpositive', 'keepMpositive', 'minimprovement']
		self.numfeatures = 12
		self.miniters = 2
		self.maxepochs = 100
		self.seed = 123456789 # seed for the random number generator.
		self.algorithm = 'brismf'
		self.keepUpositive = 0
		self.keepMpositive = 0
		self.itersForConvergence = 1 # number of iterations triggering the stopping criteria before stopping.
		self.minimprovement = 0.00004 # pretty arbitrary, just thrown in to save time.
		
	def startSVDLog(self, overwrite=True):
		"""
		Create a folder of the model, and begin a logfile in that folder.
		The logfile will be consistently structures, so that later functions
		can read lists of parameters in logfiles in order to quickly and
		consistently rerun models. 
		"""
		
		import os
		from time import ctime
		cwd = os.curdir
		locfiles = os.listdir(cwd)
		
		# If the directory exists, either choke up if we want to preserve the folder's contents,
		# or allow ourselves to write over its contents if overwrite=True
		if os.path.exists(self.resultdir):
			if overwrite is False:
				raise IOError, 'Directory %s already exists. Either delete it or rename the model.' % self.resultdir
		else:
			# makedirs doesn't choke on a path input, mkdir does. 
			os.makedirs(self.resultdir)
			
		os.chdir(self.resultdir)
		
		# Open up a logfile
		self.logfile = open('log.txt', 'wt')
		
		# Write the name and date
		self.logfile.write('Modelname: %s\n' % self.modelname)
		self.logfile.write('%s\n\n' % ctime())
		
		# Dump each item listed in self.parameters.
		# If a model contains a special parameter value, 
		# all that has to be done is to do a couple self.parameters.append()
		self.logfile.write('Parameters:\n')
		for parameter in self.parameters:
			paramvalue = getattr(self, parameter)
			self.logfile.write('%s:\t\t%s\n' % (parameter, str(paramvalue)))
		
		self.logfile.write('\nLog:\n')
		
	def updateLog(self, message, displaytext=True):
		"""
		USAGE:
		message = 'RMSE=0.9728, continuing to next training round...'
		updateLog(message, displaytext=True)
		
		This will print the message, and save it to the logfile.
		displaytext is True by default. If False, then the message
		will be written to the logfile without displaying it in the console.
		"""
		print(message)
		self.logfile.write(message+'\n')
		
	
	def svd(self, havemats=False):
		"""
		Run an SVD algorithm.
		"""
		from time import time
		
		# Locate the folder where the model will be stored
		self.outnameU = 'uf'+self.modelname
		self.outnameM = 'mf'+self.modelname
		self.resultdir = self.fd.arrayfolder.split('/')[:-3]
		self.resultdir.append('results')
		self.resultdir.append(self.modelname)
		from string import join
		self.resultdir = join(self.resultdir, '/')
		
		# Begin a log of the model training
		self.startSVDLog()
		
		# Get the proper update function
		updateFeatures = getattr(self, self.algorithm)
		
		# Give names and types to the user and movie arrays
		cdef N.ndarray[f32_t, ndim=2] U 
		cdef N.ndarray[f32_t, ndim=2] M
		
		if not havemats:
			U = N.zeros((self.numusers, self.numfeatures), dtype='float32')
			M = N.zeros((self.nummovies, self.numfeatures), dtype='float32')
			print('Initializing user and movie feature arrays...')
			rs = N.random.RandomState(self.seed)
			# We fill up the arrays with with random numbers sampled from a uniform distribution described
			# by either umin and umax or mmin and mmax. We fill them up column-by-column to avoid
			# a whole-array memory copy. Kind of hacky, but it works. No help from the NumPy message boards on an alternative.
			for i in range(self.numfeatures):
				U[:,i] = rs.uniform(low=self.umin, high=self.umax, size=self.numusers).astype('float32')
				M[:,i] = rs.uniform(low=self.mmin, high=self.mmax, size=self.nummovies).astype('float32')
			if updateFeatures == self.nsvd1 or updateFeatures == self.hybrid:
				self.W = N.zeros((self.nummovies, self.numfeatures), dtype='float32')
				for i in range(self.numfeatures):
					self.W[:,i] = rs.uniform(low=self.mmin, high=self.mmax, size=self.nummovies).astype('float32')
			elif updateFeatures==self.nsvd2 or updateFeatures==self.hybrid2:
				self.W = N.zeros((self.numusers, self.numfeatures), dtype='float32')
				for i in range(self.numfeatures):
					self.W[:,i] = rs.uniform(low=self.umin, high=self.umax, size=self.numusers).astype('float32')
			elif updateFeatures==self.snmf:
				self.W = N.zeros((self.numnodes, self.numfeatures), dtype='float32')
				for i in range(self.numfeatures):
					self.W[:,i] = rs.uniform(low=self.umin, high=self.umax, size=self.numnodes).astype('float32')
		else:
			print('Using pre-defined matrices')
			if not hasattr(self, 'U') and not hasattr(self, 'M'):
				raise ValueError, 'Must have U and M matrices declared to use the havemats=True option'
			U = self.U
			M = self.M
		
		if updateFeatures in [self.brismf, self.nsvd0, self.nsvd1, self.nsvd2, self.nsvd3, self.snmf]:
			U[:,0] = 1.0
			M[:,1] = 1.0
		
		if updateFeatures in [self.hybrid, self.hybrid2]:
			U[:,self.numfeaturesmf] = 1.0
			M[:,self.numfeaturesmf+1] = 1.0
			assert hasattr(self, 'beta'), 'Need parameter beta'
			assert hasattr(self, 'numfeaturesmf'), 'Need parameter numfeaturesmf'
			assert hasattr(self, 'lrateUn'), 'Need parameter lrateUn'
			assert hasattr(self, 'lrateMn'), 'Need parameter lrateMn'
			assert hasattr(self, 'dampfactUn'), 'Need parameter dampfactUn'
			assert hasattr(self, 'dampfactMn'), 'Need parameter dampfactMn'
			if 'beta' not in self.parameters: self.parameters.append('beta')
			if 'lrateUn' not in self.parameters: self.parameters.append('lrateUn')
			if 'lrateMn' not in self.parameters: self.parameters.append('lrateMn')
			if 'dampfactUn' not in self.parameters: self.parameters.append('dampfactUn')
			if 'dampfactMn' not in self.parameters: self.parameters.append('dampfactMn')
			if 'numfeaturesmf' not in self.properties: self.properties.append('numfeaturesmf')
			if updateFeatures == self.hybrid:
				U[:,0:self.numfeaturesmf] = U[:,0:self.numfeaturesmf]*self.beta
			elif updateFeatures == self.hybrid2:
				M[:,0:self.numfeaturesmf] = M[:,0:self.numfeaturesmf]*self.beta
				
		if updateFeatures in [self.snmf]:
			assert hasattr(self, 'numnodes'), 'Need parameter numnodes'
			assert hasattr(self, 'numconnectednodes'), 'Need paramter numconnectednodes'
			if 'numnodes' not in self.properties: self.properties.append('numnodes')
			if 'numconnectednodes' not in self.properties: self.properties.append('numconnectednoes')
				
		cdef int counter	  = 0
		cdef int epoch		  = 0
		cdef float oldRMSE	  = 0.0
		cdef float RMSE		  = 0.0
		self.RMSEhistory	  = []
		done = False
		
		RMSE = self.checkError(U, M, self.td.userRatings, self.td.userIDsForUsers, self.td.movieIDs)
			
		self.updateLog('\nRMSE Error of random initialized data: %f' % RMSE)
		
		while not done:
			epoch += 1
			self.updateLog('\nTraining epoch: %d' % epoch)
			
			
			t1 = time()
			# Update all features
			if updateFeatures in [self.nsvd2, self.hybrid2, self.nsvd3]:
				updateFeatures(U, M, self.fd.movieRatings, self.fd.userIDs)
			elif updateFeatures in [self.rmf, self.brismf, self.pmf, self.bpmf]:
				updateFeatures(U, M, self.fd.userRatings, self.fd.userIDsForUsers, self.fd.movieIDs)
			elif updateFeatures in [self.nsvd0, self.nsvd1, self.hybrid, self.snmf]:
				updateFeatures(U, M, self.fd.userRatings, self.fd.movieIDs)
			t2 = time()
			
			oldRMSE = RMSE

			RMSE = self.checkError(U, M, self.td.userRatings, self.td.userIDsForUsers, self.td.movieIDs) # Calculate the RMSE							
				
			if not RMSE < 0 and not RMSE > 0: # check for nan
				self.updateLog('\n\nWARNING: NaN values present!')
				RMSE = 99999.0
				done = True
				continue
			
			self.RMSEhistory.append(RMSE)
			self.updateLog('\tRMSE: %f' % RMSE)
			self.updateLog('\tTime: %f' % (t2-t1))
			
			deltaRMSE = oldRMSE - RMSE
			if epoch >= self.miniters:
				if deltaRMSE <= self.minimprovement:
					counter += 1
					if counter == self.itersForConvergence:
						done = True
					continue
				elif RMSE > self.minimprovement:
					N.save(self.outnameU, U)
					N.save(self.outnameM, M)
					if hasattr(self, 'W'):
						N.save('w' + self.outnameU[1:], self.W)
					
				if epoch == self.maxepochs:
					done = True
					continue
				else:
					counter = 0
		
		# If the run was successful, load the models in,
		self.U = N.load(self.outnameU + '.npy')
		self.M = N.load(self.outnameM + '.npy')
		if hasattr(self, 'W'): 
			self.W = N.load('w'+self.outnameU[1:] + '.npy')
		# close the logfile,
		self.logfile.close()
		# and spit out a probe10 prediction
		probeRatings = self.makeProbePrediction(self.td.userIDsForUsers, self.td.movieIDs, 'probe')
		probeRatings.dump('probe10ratings')

		if hasattr(self, 'qd'):
			print('Making quiz ratings')
			quizRatings = self.makeProbePrediction(self.qd.userIDsForUsers, self.qd.movieIDs, 'quiz')
			quizRatings.dump('quizratings')


	
	def makeProbePrediction(self, N.ndarray[i32_t, ndim=1] userIDsForUsers, N.ndarray[i16_t, ndim=1] movieIDs, predicttype='probe'):
		"""
		Make predictions for the probe data based on the loaded model.
		It's not very efficient code, because it doesn't particularly need to be.
		"""
		
		cdef N.ndarray[f32_t, ndim=2] U = self.U
		cdef N.ndarray[f32_t, ndim=2] M = self.M
		
		if predicttype is 'probe':
			tmp = self.td.numratings
		elif predicttype is 'quiz':
			tmp = userIDsForUsers.size
		else:
			raise ValueError, 'predicctype argument must be either "probe" or "quiz"'
		cdef int numratings = tmp
		
		cdef int numfeatures = U.shape[1]
		cdef int uid, mid, i, k
		cdef float prediction
		
		cdef N.ndarray[f32_t, ndim=1] probeRatings = N.zeros(numratings, dtype='float32')
		
		for i from 0 <= i < numratings:
			uid = userIDsForUsers[i]
			mid = movieIDs[i]
			
			prediction = 0.0
			for k from 0 <= k < numfeatures:
				prediction += U[uid,k]*M[mid,k]
				
			probeRatings[i] = prediction
		
		# Clip predictions
		probeRatings[probeRatings<1.0] = 1.0
		probeRatings[probeRatings>5.0] = 5.0
		return probeRatings
		
	def checkError(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] userRatings, \
							N.ndarray[i32_t, ndim=1] userIDsForUsers, N.ndarray[i16_t, ndim=1] movieIDs):
							
		cdef float error, prediction
		cdef int uid, mid, i, k
		cdef int numratings = self.td.numratings
		numfeatures = self.numfeatures
		
		error = 0.0
		
		for i from 0 <= i < numratings:
			uid = userIDsForUsers[i]
			mid = movieIDs[i]

			prediction = 0.0
			for k from 0 <= k < numfeatures: 
				prediction += U[uid,k]*M[mid,k]
			error += (userRatings[i] - prediction)*(userRatings[i] - prediction)
		
		return N.sqrt(error/N.float(numratings))
		
		
	#~ def nmf(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] userRatings, \
							#~ N.ndarray[i32_t, ndim=1] userIDsForUsers, N.ndarray[i16_t, ndim=1] movieIDs):
		#~ """Non-negative matrix factorization. Experimental, like WHOA DON'T STOP ME"""
							
		#~ cdef unsigned int i, k, uid, mid
		#~ cdef float error, prediction, uf, mf, tmp

		#~ cdef unsigned int numratings = <unsigned int>self.fd.numratings
		#~ cdef unsigned int numfeatures = <unsigned int>self.numfeatures
		
		#~ # == USER MATRIX UPDATE
		#~ cdef N.ndarray[f32_t] numerator = N.zeros_like(U)
		#~ cdef N.ndarray[f32_t, ndim=2] denominator = N.dot(U, N.dot(M.T, M)) # UMtM
		
		#~ for i from 0 <= i < numratings:
			#~ if i % 10000000 == 0: self.updateLog('\tTraining rating %d for the user step' % i)
			#~ uid = <unsigned int>userIDsForUsers[i]
			#~ mid = <unsigned int>movieIDs[i]
			#~ for k from 0 <= k < numfeatures:
				#~ numerator[uid,k] += userRatings[i]*M[mid,k]
		#~ U*=numerator/denominator
		#~ del numerator, denominator
		
		#~ cdef N.ndarray[f32_t, ndim=2] numerator = N.zeros_like(M)
		#~ cdef N.ndarray[f32_t, ndim=2] denominator = N.dot(N.dot(U.T, U), M)
		
		#~ for i from 0 <= i < numratings:
			#~ if i % 10000000 == 0: self.updateLog('\tTraining rating %d for the movue step' % i)
			#~ uid = <unsigned int>userIDsForUsers[i]
			#~ mid = <unsigned int>movieIDs[i]
			#~ for k from 0 <= k < numfeatures:
				#~ numerator[mid,k] += userRatings[i]*U[uid,k]
		#~ M*=numerator/denominator
		
	
	
	# ============================================================
	# BEHOLD, THE MATRIX FACTORIZATION ALGORITHMS
	# ============================================================
	
	def snmf(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] userRatings, \
							N.ndarray[i16_t, ndim=1] movieIDs):
		"""
		The beginning of an experiment in generalizing Arek paterek's NSVD1.
		The rows of the W matrix in NSVD1 each represent movies. 
		Sparse Node Matrix Factorization relaxes that definition, and each row is simply a node
		that is randomly connected to users, representing some aspect in movie-space.
		We'll see how it works.
		
		The main drawback that I can think of right now is that users cannot modify
		their linkage at all to various sparse nodes, even if they do not do a good job
		describing that user.
		So, allow some parameter to vary the link strength, in [0,1]
		"""
			
		# This is the special, extra item matrix that NSVD1 requires.
		cdef N.ndarray[f32_t, ndim=2] W = self.W 
		
		cdef unsigned int i, k, uid, mid, nid, inumRatings, istart
		cdef unsigned int u0 = <unsigned int>0 # to speed up access of bias features. 
		cdef unsigned int u1 = <unsigned int>1 # Cython wants to check bounds otherwise...
		cdef float error, prediction, uf, mf, biasreg, usernorm, tmpu, tmpm

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures
		cdef unsigned int numusers = <unsigned int> self.fd.numusers
		cdef unsigned int numnodes = <unsigned int>self.numnodes
		
		# WARNING ALERT WA WA WA: HARD-WIRED 0.5 CONNECTION PROBABILITY
		cdef unsigned int numconnectednodes = <unsigned int>self.numconnectednodes 
		
		cdef N.ndarray[i32_t, ndim=2] userIndex = self.fd.userIndex
		cdef N.ndarray[i32_t, ndim=1] numRatingsForUser = userIndex[:,1] - userIndex[:,0]
		cdef N.ndarray[f32_t, ndim=1] userFeatureCache = N.zeros(numfeatures, dtype='float32')
		cdef N.ndarray[i64_t, ndim=1] nodelist
		
		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM
		
		# special bias features
		cdef float lrateUb = self.lrateUb
		cdef float lrateMb = self.lrateMb
		cdef float dampfactUb = self.dampfactUb
		cdef float dampfactMb = self.dampfactMb
		
		
		
		for uid from 0 <= uid < numusers:
			if uid % 20000 == 0: self.updateLog('\tUpdating user %d' % uid)
			
			inumRatings = <unsigned int>numRatingsForUser[uid]
			istart = <unsigned int>userIndex[uid,u0]
			
			# Make a nodelist, and shuffle it using the userID as a seed.
			# This way, we get a replicable shuffle for each user, and
			# we don't have to store the giant connection matrix in memory.
			nodelist = N.arange(numnodes)
			rs = N.random.RandomState(N.int(uid))
			rs.shuffle(nodelist)

			# construct the user vector from movie vectors
			for i from 0 <= i < numconnectednodes:
				nid = <unsigned int>nodelist[i]
				for k from 2 <= k < numfeatures:
					userFeatureCache[k] += W[nid,k]
					
			# normalize based on movie support
			usernorm = 1/N.sqrt(numnodes)
			userFeatureCache *= usernorm
			
			# put the user vector into the user matrix
			for k from 2 <= k < numfeatures: 
				U[uid,k] = userFeatureCache[k]
			
			
			# The Engine of NSVD1 begins here----
			for i from 0 <= i < inumRatings:
				if userRatings[istart+i] == 0: continue # the (uid, mid) pair was from the quiz, no real rating
				mid = <unsigned int>movieIDs[istart+i]
				
				# 1. Compute error
				prediction = 0.0
				for k from 0 <= k < numfeatures:
					prediction += U[uid,k]*M[mid,k]
				error = userRatings[istart+i] - prediction
				
				# 2. Update the bias features
				biasreg = U[uid,u1] + M[mid,u0]
				U[uid,u1] += lrateUb*(error - dampfactUb*biasreg)
				M[mid,u0] += lrateMb*(error - dampfactMb*biasreg)
				
				# 3. Update the meat of the features
				for k from 2 <= k < numfeatures:
					uf = U[uid,k]
					mf = M[mid,k]
					U[uid,k] += lrateU*(error*mf - dampfactU*uf)
					M[mid,k] += lrateM*(error*uf - dampfactM*mf)
					
			# 4. Update the node vectors
			for i from 0 <= i < numconnectednodes:
				nid = <unsigned int>nodelist[i]
				for k from 2 <= k < numfeatures:
					W[nid,k] += (U[uid,k] - userFeatureCache[k])*usernorm
					
		return
	
	@cython.boundscheck(False)
	def nsvd0(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] userRatings, \
							N.ndarray[i16_t, ndim=1] movieIDs):
		"""
		Arek Paterek's NSVD1 algorithm, implemented a la Gravity's suggestion, from
		Takacs et al., "A Unified Approach of Factor Models and Neighbor Based Methods for Large Recommender Systems"
		Algorithm 1.
		HOWEVER, for nsvd0, w = q, that is, there is only ONE set of movie features.
		"""
		cdef unsigned int i, k, uid, mid, inumRatings, istart
		cdef unsigned int u0 = <unsigned int>0 # to speed up access of bias features. 
		cdef unsigned int u1 = <unsigned int>1 # Cython wants to check bounds otherwise...
		cdef float error, prediction, uf, mf, biasreg, usernorm

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures
		cdef unsigned int numusers = <unsigned int> self.fd.numusers
		
		cdef N.ndarray[i32_t, ndim=2] userIndex = self.fd.userIndex
		cdef N.ndarray[i32_t, ndim=1] numRatingsForUser = userIndex[:,1] - userIndex[:,0]
		cdef N.ndarray[f32_t, ndim=1] userFeatureCache = N.zeros(numfeatures, dtype='float32')

		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM
		
		# special bias features
		cdef float lrateUb = self.lrateUb
		cdef float lrateMb = self.lrateMb
		cdef float dampfactUb = self.dampfactUb
		cdef float dampfactMb = self.dampfactMb
		
		
		for uid from 0 <= uid < numusers:
			if uid % 20000 == 0: self.updateLog('\tUpdating user %d' % uid)
			
			inumRatings = <unsigned int>numRatingsForUser[uid]
			istart = <unsigned int>userIndex[uid,u0]
			usernorm = 1/N.sqrt(inumRatings)
			
			# zero the user vector
			for k from 0 <= k < numfeatures:
				userFeatureCache[k] = 0.0
				
			# load the biases
			userFeatureCache[u0] = 1.0
			userFeatureCache[u1] = U[uid,u1]
			
			# construct the user vector from movie vectors
			for i from 0 <= i < inumRatings:
				mid = <unsigned int>movieIDs[istart+i]
				for k from 2 <= k < numfeatures:
					userFeatureCache[k] += M[mid,k]
					
			# normalize based on movie support
			userFeatureCache *= usernorm
			
			# put the user vector into the user matrix
			for k from 2 <= k < numfeatures: 
				U[uid,k] = userFeatureCache[k]
			
			
			# The Engine of NSVD0 begins here----
			for i from 0 <= i < inumRatings:
				mid = <unsigned int>movieIDs[istart+i]
				
				# 1. Compute error
				prediction = 0.0
				for k from 0 <= k < numfeatures:
					prediction += U[uid,k]*M[mid,k]
				error = userRatings[istart+i] - prediction
				
				# 2. Update the bias features
				# biasreg = U[uid,u1] + M[mid,u0]
				# U[uid,u1] += lrateUb*(error - dampfactUb*biasreg)
				 # M[mid,u0] += lrateMb*(error - dampfactMb*biasreg)
				U[uid,u1] += lrateUb*error
				M[mid,u0] += lrateMb*error
					
				# 3. Update the meat of the features
				for k from 2 <= k < numfeatures:
					uf = U[uid,k]
					mf = M[mid,k]
					U[uid,k] += lrateU*(error*mf - dampfactU*uf)
					M[mid,k] += lrateM*(error*uf - dampfactM*mf)
					
			# 4. Update the movie feature vectors
			for i from 0 <= i < inumRatings:
				mid = <unsigned int>movieIDs[istart+i]
				for k from 2 <= k < numfeatures:
					M[mid,k] += (U[uid,k] - userFeatureCache[k])*usernorm
					
		return


	@cython.boundscheck(False)
	def nsvd1(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] userRatings, \
							N.ndarray[i16_t, ndim=1] movieIDs):
		"""
		Arek Paterek's NSVD1 algorithm, implemented a la Gravity's suggestion, from
		Takacs et al., "A Unified Approach of Factor Models and Neighbor Based Methods for Large Recommender Systems"
		Algorithm 1.
		"""
		
		# This is the special, extra item matrix that NSVD1 requires.
		cdef N.ndarray[f32_t, ndim=2] W = self.W 
		
		
		cdef unsigned int i, k, uid, mid, inumRatings, istart
		cdef unsigned int u0 = <unsigned int>0 # to speed up access of bias features. 
		cdef unsigned int u1 = <unsigned int>1 # Cython wants to check bounds otherwise...
		cdef float error, prediction, uf, mf, biasreg, usernorm

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures
		cdef unsigned int numusers = <unsigned int> self.fd.numusers
		
		cdef N.ndarray[i32_t, ndim=2] userIndex = self.fd.userIndex
		cdef N.ndarray[i32_t, ndim=1] numRatingsForUser = userIndex[:,1] - userIndex[:,0]
		cdef N.ndarray[f32_t, ndim=1] userFeatureCache = N.zeros(numfeatures, dtype='float32')

		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM
		
		# special bias features
		cdef float lrateUb = self.lrateUb
		cdef float lrateMb = self.lrateMb
		cdef float dampfactUb = self.dampfactUb
		cdef float dampfactMb = self.dampfactMb
		
		
		for uid from 0 <= uid < numusers:
			if uid % 20000 == 0: self.updateLog('\tUpdating user %d' % uid)
			
			inumRatings = <unsigned int>numRatingsForUser[uid]
			istart = <unsigned int>userIndex[uid,u0]
			usernorm = 1/N.sqrt(inumRatings)
			
			# zero the user vector
			for k from 0 <= k < numfeatures:
				userFeatureCache[k] = 0.0
			
			# construct the user vector from movie vectors
			for i from 0 <= i < inumRatings:
				mid = <unsigned int>movieIDs[istart+i]
				for k from 2 <= k < numfeatures:
					userFeatureCache[k] += W[mid,k]
					
			# normalize based on movie support
			userFeatureCache *= usernorm
			
			# put the user vector into the user matrix
			for k from 2 <= k < numfeatures: 
				U[uid,k] = userFeatureCache[k]
			
			
			# The Engine of NSVD1 begins here----
			for i from 0 <= i < inumRatings:
				if userRatings[istart+i] == 0: continue # the (uid, mid) pair was from the quiz, no real rating
				mid = <unsigned int>movieIDs[istart+i]
				
				# 1. Compute error
				prediction = 0.0
				for k from 0 <= k < numfeatures:
					prediction += U[uid,k]*M[mid,k]
				error = userRatings[istart+i] - prediction
				
				# 2. Update the bias features
				biasreg = U[uid,u1] + M[mid,u0]
				U[uid,u1] += lrateUb*(error - dampfactUb*biasreg)
				M[mid,u0] += lrateMb*(error - dampfactMb*biasreg)
				
				# 3. Update the meat of the features
				for k from 2 <= k < numfeatures:
					uf = U[uid,k]
					mf = M[mid,k]
					U[uid,k] += lrateU*(error*mf - dampfactU*uf)
					M[mid,k] += lrateM*(error*uf - dampfactM*mf)
					
			# 4. Update the movie feature vectors
			for i from 0 <= i < inumRatings:
				mid = <unsigned int>movieIDs[istart+i]
				for k from 2 <= k < numfeatures:
					W[mid,k] += (U[uid,k] - userFeatureCache[k])*usernorm
					
		return
	
	
	@cython.boundscheck(False)
	def nsvd2(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] movieRatings, \
							N.ndarray[i32_t, ndim=1] userIDs):
		"""
		Arek Paterek's NSVD1 algorithm, implemented a la Gravity's suggestion, from
		Takacs et al., "A Unified Approach of Factor Models and Neighbor Based Methods for Large Recommender Systems"
		Algorithm 1.
		HOWEVER, NSVD2 is user-oriented. The W matrix is an extra USER matrix, not item.
		"""
		
		# This is the special, extra user matrix that NSVD2 requires.
		cdef N.ndarray[f32_t, ndim=2] W = self.W 
		
		
		cdef unsigned int i, k, uid, mid, inumRatings, istart
		cdef unsigned int u0 = <unsigned int>0 # to speed up access of bias features. 
		cdef unsigned int u1 = <unsigned int>1 # Cython wants to check bounds otherwise...
		cdef float error, prediction, uf, mf, biasreg, movienorm

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures
		cdef unsigned int nummovies = <unsigned int> self.fd.nummovies
		
		cdef N.ndarray[i32_t, ndim=2] movieIndex = self.fd.movieIndex
		cdef N.ndarray[i32_t, ndim=1] numRatingsForMovie = movieIndex[:,1] - movieIndex[:,0]
		cdef N.ndarray[f32_t, ndim=1] movieFeatureCache = N.zeros(numfeatures, dtype='float32')

		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM
		
		# special bias features
		cdef float lrateUb = self.lrateUb
		cdef float lrateMb = self.lrateMb
		cdef float dampfactUb = self.dampfactUb
		cdef float dampfactMb = self.dampfactMb
		
		
		for mid from 0 <= mid < nummovies:
			if mid % 2000 == 0: self.updateLog('\tUpdating movie %d' % mid)
			
			inumRatings = <unsigned int>numRatingsForMovie[mid]
			istart = <unsigned int>movieIndex[mid,u0]
			movienorm = 1/N.sqrt(inumRatings)
			
			# zero the movie vector
			for k from 0 <= k < numfeatures:
				movieFeatureCache[k] = 0.0
			
			# construct the movie vector from movie vectors
			for i from 0 <= i < inumRatings:
				uid= <unsigned int>userIDs[istart+i]
				for k from 2 <= k < numfeatures:
					movieFeatureCache[k] += W[uid,k]
					
			# normalize based on user support
			movieFeatureCache *= movienorm
			
			# put the user vector into the user matrix
			for k from 2 <= k < numfeatures: 
				M[mid,k] = movieFeatureCache[k]
			
			
			# The Engine of NSVD2 begins here----
			for i from 0 <= i < inumRatings:
				if movieRatings[istart+i] == 0: continue # the (uid, mid) pair was from the quiz, no real rating
				uid = <unsigned int>userIDs[istart+i]
				
				# 1. Compute error
				prediction = 0.0
				for k from 0 <= k < numfeatures:
					prediction += U[uid,k]*M[mid,k]
				error = movieRatings[istart+i] - prediction
				
				# 2. Update the bias features
				U[uid,u1] += lrateUb*error
				M[mid,u0] += lrateMb*error
				
				# 3. Update the meat of the features
				for k from 2 <= k < numfeatures:
					uf = U[uid,k]
					mf = M[mid,k]
					U[uid,k] += lrateU*(error*mf - dampfactU*uf)
					M[mid,k] += lrateM*(error*uf - dampfactM*mf)
					
			# 4. Update the movie feature vectors
			for i from 0 <= i < inumRatings:
				uid = <unsigned int>userIDs[istart+i]
				for k from 2 <= k < numfeatures:
					W[uid,k] += (M[mid,k] - movieFeatureCache[k])*movienorm
					
		return

	@cython.boundscheck(False)
	def nsvd3(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] movieRatings, \
							N.ndarray[i32_t, ndim=1] userIDs):
		"""
		Movie-oriented version of NSVD0
		"""
				
		cdef unsigned int i, k, uid, mid, inumRatings, istart
		cdef unsigned int u0 = <unsigned int>0 # to speed up access of bias features. 
		cdef unsigned int u1 = <unsigned int>1 # Cython wants to check bounds otherwise...
		cdef float error, prediction, uf, mf, biasreg, movienorm

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures
		cdef unsigned int nummovies = <unsigned int> self.fd.nummovies
		
		cdef N.ndarray[i32_t, ndim=2] movieIndex = self.fd.movieIndex
		cdef N.ndarray[i32_t, ndim=1] numRatingsForMovie = movieIndex[:,1] - movieIndex[:,0]
		cdef N.ndarray[f32_t, ndim=1] movieFeatureCache = N.zeros(numfeatures, dtype='float32')

		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM
		
		# special bias features
		cdef float lrateUb = self.lrateUb
		cdef float lrateMb = self.lrateMb
		cdef float dampfactUb = self.dampfactUb
		cdef float dampfactMb = self.dampfactMb
		
		
		for mid from 0 <= mid < nummovies:
			if mid % 2000 == 0: self.updateLog('\tUpdating movie %d' % mid)
			
			inumRatings = <unsigned int>numRatingsForMovie[mid]
			istart = <unsigned int>movieIndex[mid,u0]
			movienorm = 1/N.sqrt(inumRatings)
			
			# zero the movie vector
			for k from 0 <= k < numfeatures:
				movieFeatureCache[k] = 0.0
			
			# construct the movie vector from movie vectors
			for i from 0 <= i < inumRatings:
				uid= <unsigned int>userIDs[istart+i]
				for k from 2 <= k < numfeatures:
					movieFeatureCache[k] += U[uid,k]
					
			# normalize based on user support
			movieFeatureCache *= movienorm
			
			# put the user vector into the user matrix
			for k from 2 <= k < numfeatures: 
				M[mid,k] = movieFeatureCache[k]
			
			
			# The Engine of NSVD2 begins here----
			for i from 0 <= i < inumRatings:
				if movieRatings[istart+i] == 0: continue # the (uid, mid) pair was from the quiz, no real rating
				uid = <unsigned int>userIDs[istart+i]
				
				# 1. Compute error
				prediction = 0.0
				for k from 0 <= k < numfeatures:
					prediction += U[uid,k]*M[mid,k]
				error = movieRatings[istart+i] - prediction
				
				# 2. Update the bias features
				U[uid,u1] += lrateUb*error
				M[mid,u0] += lrateMb*error
				
				# 3. Update the meat of the features
				for k from 2 <= k < numfeatures:
					uf = U[uid,k]
					mf = M[mid,k]
					U[uid,k] += lrateU*(error*mf - dampfactU*uf)
					M[mid,k] += lrateM*(error*uf - dampfactM*mf)
					
			# 4. Update the movie feature vectors
			for i from 0 <= i < inumRatings:
				uid = <unsigned int>userIDs[istart+i]
				for k from 2 <= k < numfeatures:
					U[uid,k] += (M[mid,k] - movieFeatureCache[k])*movienorm
					
		return
	

	
	@cython.boundscheck(False)
	def hybrid(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] userRatings, \
							N.ndarray[i16_t, ndim=1] movieIDs):
		"""
		Hybrid NMF + BRISMF, a la Gravity in 
		Takacs et al., "A Unified Approach of Factor Models and Neighbor Based Methods for Large Recommender Systems"
		self.numfeatures - number of total features
		self.numfeaturesmf - number of features for just the MF, the rest belong to NSVD1
		"""
				# This is the special, extra item matrix that NSVD1 requires.
		cdef N.ndarray[f32_t, ndim=2] W = self.W
		
		cdef unsigned int i, k, uid, mid, inumRatings, istart
		cdef unsigned int u0 = <unsigned int>0 # to speed up access of bias features. 
		cdef unsigned int u1 = <unsigned int>1 # Cython wants to check bounds otherwise...
		cdef float error, prediction, uf, mf, biasreg, usernorm, ubupdate, mbupdate

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures
		cdef unsigned int numfeaturesmf = <unsigned int>self.numfeaturesmf
		cdef unsigned int numusers = <unsigned int> self.fd.numusers
		
		cdef N.ndarray[i32_t, ndim=2] userIndex = self.fd.userIndex
		cdef N.ndarray[i32_t, ndim=1] numRatingsForUser = userIndex[:,1] - userIndex[:,0]
		cdef N.ndarray[f32_t, ndim=1] userFeatureCache = N.zeros(numfeatures, dtype='float32')

		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM
		cdef float beta = self.beta
		
		# special bias features
		cdef float lrateUb = self.lrateUb
		cdef float lrateMb = self.lrateMb
		cdef float dampfactUb = self.dampfactUb
		cdef float dampfactMb = self.dampfactMb
		
		# NSVD1 specific parameters
		cdef float lrateUn = self.lrateUn
		cdef float lrateMn = self.lrateMn
		cdef float dampfactUn = self.dampfactUn
		cdef float dampfactMn = self.dampfactMn
		
		
		for uid from 0 <= uid < numusers:
			if uid % 20000 == 0: self.updateLog('\tUpdating user %d' % uid)
			
			inumRatings = <unsigned int>numRatingsForUser[uid]
			istart = <unsigned int>userIndex[uid,u0]
			usernorm = inumRatings**-0.5
			
			# zero the user vector
			for k from numfeaturesmf <= k < numfeatures:
				userFeatureCache[k] = 0.0
			
			# construct the user vector from movie vectors
			for i from 0 <= i < inumRatings:
				mid = <unsigned int>movieIDs[istart+i]
				for k from numfeaturesmf+2 <= k < numfeatures:
					userFeatureCache[k] += W[mid,k]
			
			# normalize based on movie support, multiply in (1-beta)
			userFeatureCache *= usernorm
			
			# put the user vector into the user matrix
			for k from numfeaturesmf+2 <= k < numfeatures: 
				U[uid,k] = userFeatureCache[k]
			
			# The Engine of NSVD1 begins here----
			for i from 0 <= i < inumRatings:
				if userRatings[istart+i] == 0: continue # the (uid, mid) pair was from the quiz, no real rating
				mid = <unsigned int>movieIDs[istart+i]
				
				# 1. Compute error
				prediction = 0.0
				for k from 0 <= k < numfeatures:
					prediction += U[uid,k]*M[mid,k]
				error = userRatings[istart+i] - prediction
				
				# 2. Update the bias features
				biasreg = U[uid,u1+numfeaturesmf] + M[mid,u0+numfeaturesmf]
				U[uid,u1+numfeaturesmf] += lrateUb*(error*beta - dampfactUb*biasreg)
				M[mid,u0+numfeaturesmf] +=  lrateMb*(error*beta - dampfactMb*biasreg)
				
				# 3. Update the bulk of the features
				# 	for the MF
				for k from 0 <= k < numfeaturesmf:
					uf = U[uid,k]
					mf = M[mid,k]
					U[uid,k] += lrateU*(error*beta*mf - dampfactU*uf)
					M[mid,k] += lrateM*(error*beta*uf - dampfactM*mf)
					
				# 	for NSVD1
				for k from numfeaturesmf+2 <= k < numfeatures:
					uf = U[uid,k]
					mf = M[mid,k]
					U[uid,k] += lrateUn*(error*(1-beta)*mf - dampfactUn*uf)
					M[mid,k] += lrateMn*(error*(1-beta)*uf - dampfactMn*mf)
					
			# 4. Update the movie feature vectors
			for i from 0 <= i < inumRatings:
				mid = <unsigned int>movieIDs[istart+i]
				for k from 2+numfeaturesmf <= k < numfeatures:
					W[mid,k] += (U[uid,k] - userFeatureCache[k])*usernorm
			
			
		return
	
	@cython.boundscheck(False)
	def hybrid2(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] movieRatings, \
							N.ndarray[i32_t, ndim=1] userIDs):
		"""
		Hybrid NMF + BRISMF, a la Gravity in 
		Takacs et al., "A Unified Approach of Factor Models and Neighbor Based Methods for Large Recommender Systems"
		self.numfeatures - number of total features
		self.numfeaturesmf - number of features for just the MF, the rest belong to NSVD1
		"""
		# This is the special, extra user matrix that NSVD1 requires.
		cdef N.ndarray[f32_t, ndim=2] W = self.W
		
		cdef unsigned int i, k, uid, mid, inumRatings, istart
		cdef unsigned int u0 = <unsigned int>0 # to speed up access of bias features. 
		cdef unsigned int u1 = <unsigned int>1 # Cython wants to check bounds otherwise...
		cdef float error, prediction, uf, mf, biasreg, usernorm, ubupdate, mbupdate

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures
		cdef unsigned int numfeaturesmf = <unsigned int>self.numfeaturesmf
		cdef unsigned int nummovies = <unsigned int> self.fd.nummovies
		
		cdef N.ndarray[i32_t, ndim=2] movieIndex = self.fd.movieIndex
		cdef N.ndarray[i32_t, ndim=1] numRatingsForMovie = movieIndex[:,1] - movieIndex[:,0]
		cdef N.ndarray[f32_t, ndim=1] movieFeatureCache = N.zeros(numfeatures, dtype='float32')

		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM
		cdef float beta = self.beta
		
		# special bias features
		cdef float lrateUb = self.lrateUb
		cdef float lrateMb = self.lrateMb
		cdef float dampfactUb = self.dampfactUb
		cdef float dampfactMb = self.dampfactMb
		
		# NSVD1 specific parameters
		cdef float lrateUn = self.lrateUn
		cdef float lrateMn = self.lrateMn
		cdef float dampfactUn = self.dampfactUn
		cdef float dampfactMn = self.dampfactMn
		
		for mid from 0 <= mid < nummovies:
			if mid % 2000 == 0: self.updateLog('\tUpdating movie %d' % mid)
			
			inumRatings = <unsigned int>numRatingsForMovie[mid]
			istart = <unsigned int>movieIndex[mid,u0]
			movienorm= inumRatings**-0.5
			
			# zero the movie vector
			for k from numfeaturesmf <= k < numfeatures:
				movieFeatureCache[k] = 0.0
			
			# construct the movie vector from user vectors
			for i from 0 <= i < inumRatings:
				uid = <unsigned int>userIDs[istart+i]
				for k from numfeaturesmf+2 <= k < numfeatures:
					movieFeatureCache[k] += W[uid,k]
			
			# normalize based on movie support, multiply in (1-beta)
			movieFeatureCache *= movienorm#*(1-beta)
			
			# put the user vector into the user matrix
			for k from numfeaturesmf+2 <= k < numfeatures: 
				M[mid,k] = movieFeatureCache[k]
			
			# The Engine of NSVD1 begins here----
			for i from 0 <= i < inumRatings:
				if movieRatings[istart+i] == 0: continue # the (uid, mid) pair was from the quiz, no real rating
				uid = <unsigned int>userIDs[istart+i]
				
				# 1. Compute error
				prediction = 0.0
				for k from 0 <= k < numfeatures:
					prediction += U[uid,k]*M[mid,k]
				error = movieRatings[istart+i] - prediction
				
				# 2. Update the bias features
				biasreg = U[uid,u1+numfeaturesmf] + M[mid,u0+numfeaturesmf]
				U[uid,u1+numfeaturesmf] += lrateUb*(error - dampfactUb*biasreg)
				M[mid,u0+numfeaturesmf] +=  lrateMb*(error - dampfactMb*biasreg)

				
				# 3. Update the bulk of the features
				# 	for the MF
				for k from 0 <= k < numfeaturesmf:
					uf = U[uid,k]
					mf = M[mid,k]
					U[uid,k] += lrateU*(beta*error*mf - dampfactU*uf)
					M[mid,k] += lrateM*(beta*error*uf - dampfactM*mf)
				# 	for NSVD1
				for k from numfeaturesmf+2 <= k < numfeatures:
					uf = U[uid,k]
					mf = M[mid,k]
					U[uid,k] += lrateUn*(error*(1-beta)*mf - dampfactUn*uf)
					M[mid,k] += lrateMn*(error*(1-beta)*uf - dampfactMn*mf)
					
			# 4. Update the movie feature vectors
			for i from 0 <= i < inumRatings:
				uid = <unsigned int>userIDs[istart+i]
				for k from 2+numfeaturesmf <= k < numfeatures:
					W[uid,k] += (U[mid,k] - movieFeatureCache[k])*movienorm
			
			
		return
	
	
	@cython.boundscheck(False)		
	def rmf(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] userRatings, \
							N.ndarray[i32_t, ndim=1] userIDsForUsers, N.ndarray[i16_t, ndim=1] movieIDs):
		"""Regularized Matrix Factorization.
		Adds a damping constant to keep feature values low."""				
		cdef unsigned int i, k, uid, mid
		cdef float error, prediction, uf, mf

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures

		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM
		
		for i from 0 <= i < numratings:
			if i % 10000000 == 0: self.updateLog('\tTraining rating %d' % i)
			uid = <unsigned int>userIDsForUsers[i]
			mid = <unsigned int>movieIDs[i]
			# Find errors and predictions, before we do any updates
			prediction = 0.0
			for k from 0 <= k < numfeatures: 
				prediction += U[uid,k]*M[mid,k]
			error = userRatings[i] - prediction

			# Update non-bias features
			for k from 0 <= k < numfeatures:
				uf = U[uid,k]
				mf = M[mid,k]
				U[uid,k] += lrateU*(error*mf - dampfactU*uf)
				M[mid,k] += lrateM*(error*uf - dampfactM*mf)
					
		return

	
	@cython.boundscheck(False)		
	def brismf(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] userRatings, \
							N.ndarray[i32_t, ndim=1] userIDsForUsers, N.ndarray[i16_t, ndim=1] movieIDs):
		"""Biased Regularized Incremental Simultaneous Matrix Factorization, a la
		"Investigation of Various Matrix Factorization Methods for 
		Large Recommender Systems", as seen in KDD 2008"""
		cdef unsigned int i, k, uid, mid
		cdef float error, prediction, uf, mf, biasreg

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures

		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM
		
		# special bias features
		cdef float lrateUb = self.lrateUb
		cdef float lrateMb = self.lrateMb
		cdef float dampfactUb = self.dampfactUb
		cdef float dampfactMb = self.dampfactMb
		
		for i from 0 <= i < (numratings):
			if i % 10000000 == 0: self.updateLog('\tTraining rating %d' % i)
			uid = <unsigned int>userIDsForUsers[i]
			mid = <unsigned int>movieIDs[i]
			# Find errors and predictions, before we do any updates
			prediction = 0.0
			for k from 0 <= k < numfeatures: 
				prediction += U[uid,k]*M[mid,k]
			error = userRatings[i] - prediction
			
			# Update bias features
			biasreg = U[uid,1] + M[mid,0]
			U[uid,1] += lrateUb*(error - dampfactUb*biasreg)
			M[mid,0] += lrateMb*(error - dampfactMb*biasreg)
			
			# Update non-bias features
			for k from 2 <= k < numfeatures:
				uf = U[uid,k]
				mf = M[mid,k]
				U[uid,k] += lrateU*(error*mf - dampfactU*uf)
				M[mid,k] += lrateM*(error*uf - dampfactM*mf)
					
		return
	
	
	@cython.boundscheck(False)
	def bpmf(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] userRatings, \
							N.ndarray[i32_t, ndim=1] userIDsForUsers, N.ndarray[i16_t, ndim=1] movieIDs):
		"""
		Biased Positive matrix factorization.
		Keep either user and movie features positive, depending on if
		keepUpositive and keepMpositive are 1 or 0.
		"""
		cdef unsigned int i, k, uid, mid
		cdef float error, prediction, uf, mf, biasreg, tmpm, tmpu

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures
		
		cdef int keepUpositive = self.keepUpositive
		cdef int keepMpositive = self.keepMpositive
		
		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM
		
		# special bias features
		cdef float lrateUb = self.lrateUb
		cdef float lrateMb = self.lrateMb
		cdef float dampfactUb = self.dampfactUb
		cdef float dampfactMb = self.dampfactMb
		
		for i from 0 <= i < (numratings):
			if i % 10000000 == 0: self.updateLog('\tTraining rating %d' % i)
			uid = <unsigned int>userIDsForUsers[i]
			mid = <unsigned int>movieIDs[i]
			# Find errors and predictions, before we do any updates
			prediction = 0.0
			for k from 0 <= k < numfeatures: 
				prediction += U[uid,k]*M[mid,k]
			error = userRatings[i] - prediction
			
			# Update bias features
			biasreg = U[uid,1] + M[mid,0]
			tmpu = U[uid,1] + lrateUb*(error - dampfactUb*biasreg)
			tmpm = M[mid,0] + lrateMb*(error - dampfactMb*biasreg)
			if keepUpositive == 1 and tmpu < 0.0: tmpu = 0.0
			if keepMpositive == 1 and tmpm < 0.0: tmpm = 0.0
			U[uid,1] = tmpu
			M[mid,0] = tmpm
			
			# Update non-bias features
			for k from 2 <= k < numfeatures:
				uf = U[uid,k]
				mf = M[mid,k]
				tmpu = U[uid,k] + lrateU*(error*mf - dampfactU*uf)
				tmpm = M[mid,k] + lrateM*(error*uf - dampfactM*mf)
				if keepUpositive == 1 and tmpu < 0.0: tmpu = 0.0
				if keepMpositive == 1 and tmpm < 0.0: tmpm = 0.0
				U[uid,k] = tmpu
				M[mid,k] = tmpm
		
		return
				
	@cython.boundscheck(False)
	def pmf(self, N.ndarray[f32_t, ndim=2] U, N.ndarray[f32_t, ndim=2] M, N.ndarray[i8_t, ndim=1] userRatings, \
							N.ndarray[i32_t, ndim=1] userIDsForUsers, N.ndarray[i16_t, ndim=1] movieIDs):
		"""Positive matrix factorization.
		Keep either user and movie features positive, depending on if
		keepUpositive and keepMpositive are 1 or 0."""
		cdef unsigned int i, k, uid, mid
		cdef float error, prediction, uf, mf, biasreg, tmpm, tmpu

		cdef unsigned int numratings = <unsigned int>self.fd.numratings
		cdef unsigned int numfeatures = <unsigned int>self.numfeatures

		cdef int keepUpositive = self.keepUpositive
		cdef int keepMpositive = self.keepMpositive

		cdef float lrateU = self.lrateU
		cdef float lrateM = self.lrateM
		cdef float dampfactU = self.dampfactU
		cdef float dampfactM = self.dampfactM

		for i from 0 <= i < (numratings):
			if i % 10000000 == 0: self.updateLog('\tTraining rating %d' % i)
			uid = <unsigned int>userIDsForUsers[i]
			mid = <unsigned int>movieIDs[i]
			# Find errors and predictions, before we do any updates
			prediction = 0.0
			for k from 0 <= k < numfeatures: 
				prediction += U[uid,k]*M[mid,k]
			error = userRatings[i] - prediction

			# Update non-bias features
			for k from 0  <= k < numfeatures:
				uf = U[uid,k]
				mf = M[mid,k]
				tmpu = U[uid,k] + lrateU*(error*mf - dampfactU*uf)
				tmpm = M[mid,k] + lrateM*(error*uf - dampfactM*mf)
				if keepUpositive == 1 and tmpu < 0.0: tmpu = 0.0
				if keepMpositive == 1 and tmpm < 0.0: tmpm = 0.0
				U[uid,k] = tmpu
				M[mid,k] = tmpm

		return
	
	def subsampleUsers(self, fraction=0.3):
		"""
		Update the svd model to use only a subsample of users
		
		CAUSES A SEGMENTATION FAULT AFTER THE ALLOTED NUMBER
		"""
		newnumusers = int(480189*fraction)
		
		# Make sure that the last user is in the probe set.
		done = False
		while done is False:
			numproberatings = N.argwhere(self.td.userIDsForUsers==newnumusers)
			if len(numproberatings) == 0:
				newnumusers += 1
			else:
				done = True
		
		
		self.numusers = newnumusers
		self.fd.numusers = newnumusers
		numproberatings = numproberatings[0][0] # make it just an integer
		self.td.numratings = numproberatings
		self.fd.numratings = self.fd.userIndex[self.numusers-1,1]
		self.numratings = self.fd.numratings
	
	def restoreFromSubsample(self):
		self.numusers = 480189
		self.fd.numusers = 480189
		self.td.numratings = self.td.userRatings.size
		self.fd.numratings = self.fd.userRatings.size
		self.numratings = self.fd.numratings
	
	# cdef inline the update equations--
	# they deal with non-array floats, so we could simplify factoring multiple update-types.

# END OF SVD CLASS

def loadModel(modelname, loadarrays=False, resultdir='/home/alex/workspace/flix/results/'):
	"""
	Make an instance of the SVD class using information from the logfile
	
	NOT DONE
	"""
	from os import join, chdir
	modeldir = join(resultdir, modelname)
	pass
	
	
	