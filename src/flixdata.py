import numpy as N
"""
A convenient class for loading in
the Netflix data.
"""

class FlixData(object):
	"""Wrapper class for various array memory and speed tests"""
	
	
	def __init__(self, arrayfolder='/home/alex/workspace/flix/data/arrays/', loaddata=True):
		super(FlixData, self).__init__()
		self.arrayfolder = arrayfolder
		self.moviearrayfolder = self.arrayfolder + 'movie_indexed/'
		self.userarrayfolder = self.arrayfolder + 'user_indexed/'
		self.nummovies = 17770
		self.numusers = 480189
		
		if loaddata is True: 
			self.getUserIndexedData()
			self.getMovieIDsForSortedUsers()
			self.getUserIDsForSortedUsers()
			self.numratings = self.userRatings.size
			self.numusers = self.userIndex.shape[0]	
			self.nummovies = 17770 # hard-wired is fine...
			
		if loaddata is 'quiz':
			self.getUserInfoWithoutRatings()

# Data loading functions
	# ===========================	
	def getEverythingButTheRatings(self):
		"""For the quiz and test datasets..."""
		self.getMovieIDsForSortedMovies()
		self.getMovieIDsForSortedUsers()
		self.getUserIDsForSortedMovies()
		self.getUserIDsForSortedUsers()
		self.getJustBothIndexes()
		
		self.numusers = self.userIndex.shape[0]
		self.nummovies = self.movieIndex.shape[0]
		self.numratings = self.userIDs.size # could well be movieIDs or any of those arrays
		
	def getMovieIndexedData(self):
		ratingsFile = open(self.moviearrayfolder + 'ratings', 'rb')
		self.movieRatings = N.fromfile(ratingsFile, dtype='int8')
		ratingsFile.close()
		
		movieIndexFile = open(self.moviearrayfolder + 'movieindex_BOUNDS')
		self.movieIndex = N.load(movieIndexFile)
		movieIndexFile.close()
		
		self.getMovieIDsForSortedMovies()
		self.getUserIDsForSortedMovies()
		
		self.numratings = self.movieRatings.size
		
	def getUserIndexedData(self):
		ratingsFile = open(self.userarrayfolder + 'ratings', 'rb')
		self.userRatings = N.fromfile(ratingsFile, dtype='int8')
		ratingsFile.close()
		
		userIndexFile = open(self.userarrayfolder + 'userindex_BOUNDS')
		self.userIndex = N.load(userIndexFile)
		userIndexFile.close()
		
		self.getMovieIDsForSortedUsers()
		self.getUserIDsForSortedUsers()
		
		self.numratings = self.userRatings.size
		
	def getJustUserIndex(self):
		userIndexFile = open(self.userarrayfolder + 'userindex_BOUNDS')
		self.userIndex = N.load(userIndexFile)
		userIndexFile.close()
		
	def getJustMovieIndex(self):
		movieIndexFile = open(self.moviearrayfolder + 'movieindex_BOUNDS')
		self.movieIndex = N.load(movieIndexFile)
		movieIndexFile.close()
		
	def getJustBothIndexes(self):
		self.getJustUserIndex()
		self.getJustMovieIndex()
		
	def getBothIndexedData(self):
		self.getMovieIndexedData()
		self.getUserIndexedData()
		
	def getUserIDsForSortedMovies(self):
		userIDsFile = open(self.moviearrayfolder + 'userids', 'rb')
		self.userIDs = N.fromfile(userIDsFile, dtype='int32')
		userIDsFile.close()
		
	def getUserIDsForSortedUsers(self):
		"""This is often an unnecessary function, 
		because the user IDs for arrays sorted by users
		is a bit redundant with faster bounds-based indexing
		(that are loaded in with getUserIndexedData)"""
		userIDsFile = open(self.userarrayfolder + 'userids', 'rb')
		self.userIDsForUsers = N.fromfile(userIDsFile, dtype='int32')
		userIDsFile.close()
	
	def getOriginalUserIDs(self):
		"""
		We usually work with translated user IDs in a continuous
		range from 0 to 480188, but sometimes we need the original
		IDs.
		"""
		originalUserIDFile = open(self.arrayfolder + 'originalUserIDs', 'rb')
		self.originalUserIDs = N.fromfile(originalUserIDFile, dtype='int32')
		originalUserIDFile.close()
	
	def getMovieIDsForSortedUsers(self):
		movieIDsFile = open(self.userarrayfolder + 'movieids', 'rb')
		self.movieIDs = N.fromfile(movieIDsFile, dtype='int16')
		movieIDsFile.close()
		
	def getMovieIDsForSortedMovies(self):
		"""This is often an unnecessary function, 
		because the movie IDs for arrays sorted by movies
		is a bit redundant with faster bounds-based indexing
		(that are loaded in with getMovieIndexedData)"""
		movieIDsFile = open(self.moviearrayfolder + 'movieids', 'rb')
		self.movieIDsForMovies = N.fromfile(movieIDsFile, dtype='int16')
		movieIDsFile.close()
	
	def getOppositeIDsForSortedArrays(self):
		self.getUserIDsForSortedMovies()
		self.getMovieIDsForSortedUsers()
		
	def getBothRatingsAndIndices(self):
		self.getBothIndexedData()
		self.getOppositeIDsForSortedArrays()
		
	def getDatesSortedByUser(self):
		datesByUserFile = open(self.userarrayfolder + 'dates', 'rb')
		self.datesByUser = N.fromfile(datesByUserFile, dtype='int32')
		datesByUserFile.close()
		
	def getDatesSortedByMovie(self):
		datesByMovieFile = open(self.moviearrayfolder + 'dates', 'rb')
		self.datesByMovie = N.fromfile(datesByMovieFile, dtype='int32')
		datesByMovieFile.close()
	
	
	def getUserInfoWithoutRatings(self):
		"""Useful for pulling in movie and userids in quiz data"""
		self.getUserIDsForSortedUsers()
		self.getMovieIDsForSortedUsers()
		
	def getMovieInfoWithoutRatings(self):
		self.getUserIDsForSortedMovies()
		self.getMovieIDsForSortedMovies()
	
	def getUserAverages(self):
		averageUserRatingFile = open(self.arrayfolder + 'useraverages', 'rb')
		self.userAverages = N.load(averageUserRatingFile)
		averageUserRatingFile.close()
	
	def getMovieAverages(self):
		averageMovieRatingFile = open(self.arrayfolder + 'movieaverages', 'rb')
		self.movieAverages = N.load(averageMovieRatingFile)
		averageMovieRatingFile.close()
		
	def getUserVariances(self):
		if not hasattr(self, 'userAverages'): self.getUserAverages()
		self.userVariances = N.zeros((self.numusers,), dtype=N.float32)
		for i in range(self.numusers):
			self.userVariances[i] = N.var(self.userRatings[fd.ui(i)])
	
	def getMovieVariances(self):
		if not hasattr(self, 'movieAverages'): self.getMovieAverages()
		self.movieVariances = N.zeros((self.nummovies,), dtype=N.float32)
		for i in range(self.nummovies):
			self.movieVariances[i] = N.var(self.movieRatings[fd.mi(i)])

	def sortUserRatingsByDate(self):
		"""
		Within each user's set of ratings, reorder them by date
		This apparently improves prediction performance with SVD
		(according to team gravity's 2008 KDD paper)
		"""
		if not hasattr(self, 'datesByUser'): self.getDatesSortedByUser()
		if not hasattr(self, 'userIDsForUsers'): self.getUserIDsForSortedUsers()
		
		for i in range(self.numusers):
			if i % 10000 == 0: print('Resorting ratings by date for user %d...' % i)
			uidx = self.ui(i)
			didx = N.argsort(self.datesByUser[uidx])
			self.userRatings[uidx] = self.userRatings[uidx][didx]
			self.userIDsForUsers[uidx] = self.userIDsForUsers[uidx][didx]
			self.movieIDs[uidx] = self.movieIDs[uidx][didx]
			self.datesByUser[uidx] = self.datesByUser[uidx][didx]
	
	def sortUserRatingsByMovie(self):
		"""
		Within each user's set of ratings, reorder them by movie
		"""
		if not hasattr(self, 'datesByUser'): self.getDatesSortedByUser()
		if not hasattr(self, 'userIDsForUsers'): self.getUserIDsForSortedUsers()
		
		for i in range(self.numusers):
			if i % 10000 == 0: print('Resorting ratings by movie for user %d...' % i)
			uidx = self.ui(i)
			didx = N.argsort(self.movieIDs[uidx])
			self.userRatings[uidx] = self.userRatings[uidx][didx]
			self.userIDsForUsers[uidx] = self.userIDsForUsers[uidx][didx]
			self.movieIDs[uidx] = self.movieIDs[uidx][didx]
			self.datesByUser[uidx] = self.datesByUser[uidx][didx]
	
	def sortMovieRatingsByDate(self):
		"""For the movie-sorted index, subsort by date."""
		if not hasattr(self, 'datesByMovie'): self.getDatesSortedByMovie()
		if not hasattr(self, 'movieIDsForMovies'): self.getMovieIDsForSortedMovies()
		
		for i in range(self.nummovies):
			if i % 1000 == 0: print('Resorting ratings by date for movie %d...' % i)
			midx = self.mi(i)
			didx = N.argsort(self.datesByMovie[midx])
			self.movieRatings[midx] = self.movieRatings[midx][didx]
			self.movieIDsForMovies[midx] = self.movieIDsForMovies[midx][didx]
			self.userIDs[midx] = self.userIDs[midx][didx]
			self.datesByMovie[midx] = self.datesByMovie[midx][didx]
		
	def sortMovieRatingsByUser(self):
		"""For the movie-sorted index, subsort by user."""
		if not hasattr(self, 'datesByMovie'): self.getDatesSortedByMovie()
		if not hasattr(self, 'movieIDsForMovies'): self.getMovieIDsForSortedMovies()
		
		for i in range(self.nummovies):
			if i % 1000 == 0: print('Resorting ratings by date for movie %d...' % i)
			midx = self.mi(i)
			didx = N.argsort(self.userIDs[midx])
			self.movieRatings[midx] = self.movieRatings[midx][didx]
			self.movieIDsForMovies[midx] = self.movieIDsForMovies[midx][didx]
			self.userIDs[midx] = self.userIDs[midx][didx]
			self.datesByMovie[midx] = self.datesByMovie[midx][didx]	

	"""
	Probe functions
	"""
	def loadProbe(self, probename='probe10'):
		probeFile = open(self.arrayfolder + probename, 'rb')
		self.probe = N.load(probeFile)
		probeFile.close()
	
	def identifyProbeData(self, probename='probe10'):
		"""
		The rating information contains the probe data,
		so when we're training to test performance on the probe,
		remove all of the entries that live in the probe
		"""
		if not hasattr(self, 'probe'): self.loadProbe(probename)
		probelength = len(self.probe)
		userIndexedProbes = N.zeros(probelength, dtype='int32')
		movieIndexedProbes = N.zeros(probelength, dtype='int32')
		
		for i in range(probelength):
			if i % 1000 == 0: print('Analyzing probe rating %d...' % i)
			movie = self.probe[i,0]
			user = self.probe[i,1]
			uidx = self.ui(user)
			midx = self.mi(movie)
			userIndexedProbes[i] = N.argwhere(self.movieIDs[uidx]==movie)+uidx[0]
			movieIndexedProbes[i] = N.argwhere(self.userIDs[midx]==user)+midx[0]
		
		return userIndexedProbes, movieIndexedProbes
	
	def spawnProbeData(self, uip=None, mip=None, probename='probe10'):
		"""
		Using identifyProbeData(), remove all probe entries
		from the active class, and return a new instance
		of FlixData containing just the probe data.
		
		td = fd.spawnProbeData()
		"""
		if uip is None or mip is None:
			self.uip, self.mip = self.identifyProbeData(probename)
		
		# Make sure we've loaded all the data we can
		if not hasattr(self, 'movieIDsForMovies'): 
			print('Loading movie IDs...')
			self.getMovieIDsForSortedMovies()
		if not hasattr(self, 'userIDsForUsers'): 
			print('Loading user IDs...')
			self.getUserIDsForSortedUsers()
		if not hasattr(self, 'datesByMovie'): 
			print('Loading dates sorted by movie...')
			self.getDatesSortedByMovie()
		if not hasattr(self, 'datesByuser'): 
			print('Loading dates sorted by user...')
			self.getDatesSortedByUser()
		
		# Make sure the data is properly sorted
		print('Sorting the probe indices...')
		uip = N.sort(uip)
		mip = N.sort(mip)
		
		print('Creating the new probe FlixData instance...')
		td = FlixData(arrayfolder='/home/alex/workspace/flix/data/probe10/', loaddata=False)
		td.movieIDs			 = self.movieIDs[uip]
		td.movieIDsForMovies = self.movieIDsForMovies[mip]
		td.userIDs			 = self.userIDs[mip]
		td.userIDsForUsers	 = self.userIDsForUsers[uip]
		td.movieRatings		 = self.movieRatings[mip]
		td.userRatings		 = self.userRatings[uip]
		td.datesByMovie		 = self.datesByMovie[mip]
		td.datesByUser		 = self.datesByUser[uip]
		
		# We'll have to rebuild the movie index
		print('Rebuilding the movie and user indices...')
		td.userIndex  = makeBoundedIndex(td.userIDsForUsers)
		td.movieIndex = makeBoundedIndex(td.movieIDsForMovies)
		
		# Update some constants
		td.numratings = td.userRatings.size
		td.nummovies  = td.movieIndex.shape[0]
		td.numusers   = td.userIndex.shape[0]
		
		self.removeProbeData()
		
		return td
	
	def removeProbeData(self, probename=None):
		"""
		Remove probe data specified in a NumPy array
		created with the function readProbeFile(), 
		then zero-indexing the users, then saved in
		the arrayfolder of the current FlixData instance.
		
		"""
		if probename is None:
			if not hasattr(self, 'uip') or not hasattr(self, 'mip'):
				raise ValueError, 'No probe information found, and no probe file specified.'
		else:
			self.uip, self.mip = self.identifyProbeData(probename)
		
		if not hasattr(self, 'userIDsForUsers'): self.getUserIDsForSortedUsers()
		if not hasattr(self, 'movieIDsForMovies'): self.getMovieIDsForSortedMovies()
		
		# delete the entries from fd
		print('Indexing the probe entries to be removed...')
		self.uip = N.sort(self.uip)
		self.mip = N.sort(self.mip)
		uiRemove = invertSet(0, self.userRatings.size, self.uip)
		miRemove = invertSet(0, self.userRatings.size, self.mip)

		print('Deleting probe entries from current FlixData instance...')
		self.movieIDs			 = self.movieIDs[uiRemove]
		self.userIDs			 = self.userIDs[miRemove]
		self.movieRatings		 = self.movieRatings[miRemove]
		self.userRatings		 = self.userRatings[uiRemove]
		self.userIDsForUsers	 = self.userIDsForUsers[uiRemove]
		self.movieIDsForMovies	 = self.movieIDsForMovies[miRemove]
		if hasattr(self, 'datesByMovie'): 
			self.datesByMovie	 = self.datesByMovie[miRemove]
		if hasattr(self, 'datesByUser'):
			self.datesByUser	 = self.datesByUser[uiRemove]
		
		del uiRemove, miRemove
		
		# Rebuild the user and movie index for the original array
		print('Rebuilding the current user and movie indices...')
		self.userIndex = makeBoundedIndex(self.userIDsForUsers)
		self.movieIndex = makeBoundedIndex(self.movieIDsForMovies)
		
		self.numratings = self.userRatings.size
		
	def resaveData(self):
		"""
		If any modifications have been made to the data that need saving,
		use this function.
		====== DANGER DANGER DANGER ====== 
		This will overwrite pre-existing data
		"""
		
		if not hasattr(self, 'movieIDsForMovies'): 
			print('Loading movie IDs...')
			self.getMovieIDsForSortedMovies()
		if not hasattr(self, 'userIDsForUsers'): 
			print('Loading user IDs...')
			self.getUserIDsForSortedUsers()
		if not hasattr(self, 'datesByMovie'): 
			print('Loading dates sorted by movie...')
			self.getDatesSortedByMovie()
		if not hasattr(self, 'datesByUser'): 
			print('Loading dates sorted by user...')
			self.getDatesSortedByUser()
		
		f = open(self.moviearrayfolder + 'dates', 'wb')
		f.write(self.datesByMovie.tostring())
		f.close()
		
		f = open(self.moviearrayfolder + 'movieids', 'wb')
		f.write(self.movieIDsForMovies.tostring())
		f.close()
		
		f = open(self.moviearrayfolder + 'movieindex_BOUNDS', 'wb')
		self.movieIndex.dump(f)
		f.close()
		
		f = open(self.moviearrayfolder + 'ratings', 'wb')
		f.write(self.movieRatings.tostring())
		f.close()
		
		f = open(self.moviearrayfolder + 'userids', 'wb')
		f.write(self.userIDs.tostring())
		f.close()
		
		f = open(self.userarrayfolder + 'dates', 'wb')
		f.write(self.datesByUser.tostring())
		f.close()
		
		f = open(self.userarrayfolder + 'movieids', 'wb')
		f.write(self.movieIDs.tostring())
		f.close()
		
		f = open(self.userarrayfolder + 'userindex_BOUNDS', 'wb')
		self.userIndex.dump(f)
		f.close()
		
		f = open(self.userarrayfolder + 'ratings', 'wb')
		f.write(self.userRatings.tostring())
		f.close()
		
		f = open(self.userarrayfolder + 'userids', 'wb')
		f.write(self.userIDsForUsers.tostring())
		f.close()
		
		
	
	
	
	"""	
	Data access functions
	========================
	NOTE: calling these functions results in a modest slowdown
	as opposed to if you rewrote this code elsewhere.
	
	It's best to do something like
	>> fd.userRatings[fd.ui(6)]
	as opposed to 
	>> fd.getRatingsForUser(6)
	"""

	def getRatingsForUser(self, userid):
		return self.userRatings[self.userIndex[userid,0]:self.userIndex[userid,1]]
		
	def getRatingsForMovie(self, movieid):
		return self.movieRatings[self.movieIndex[movieid,0]:self.movieIndex[movieid,1]]
		
	def getUserIndex(self, userid):
		# The getUserIndex functions should be called only if you're really lazy.
		# It results in a lot of overhead. There's usually no sense using it
		# in production. I use it in testing, to make my life easier.
		return N.r_[self.userIndex[userid,0]:self.userIndex[userid,1]]
		
	def getMovieIndex(self, movieid):
		return N.r_[self.movieIndex[movieid,0]:self.movieIndex[movieid,1]]
		
	def ui(self, userid):
		"""convenience function for making a user index"""
		return N.r_[self.userIndex[userid,0]:self.userIndex[userid,1]]
		
	def mi(self, movieid):
		"""convenience function for making a movie index"""
		return N.r_[self.movieIndex[movieid,0]:self.movieIndex[movieid,1]]
		

	
	def zeroIndexUserIDs(self, uids):
		self.getOriginalUserIDs()
		convert = N.r_[0:N.max(self.originalUserIDs)+1]
		convert[self.originalUserIDs] = N.r_[0:self.numusers]
		return convert[uids]

def saveMiniset(fd, index, outputdir):
	"""
	fd - Instance of FlixData class
	Make some mini probes, picking ratings specified
	by the 'index' numpy array. 
	The indices specify ratings in the movie-sorted 
	ratings array.
	
	In the user_indexed/ folder, the 'index' file
	still refers to a sorted movie index
	"""
	from os import mkdir

	index = N.asarray(index)
	
	# Make movie-indexed subset
	moviedir = outputdir+'movie_indexed/'
	try:
		mkdir(moviedir)
	except:
		print('movie indexed directory already exists...')
	movieindex_BOUNDS = makeBoundedIndex(fd.movieIDsForMovies[index])
	print('Dumping movie files...')
	fm = open(moviedir+'movieid', 'wb')
	fu = open(moviedir+'userid', 'wb')
	fr = open(moviedir+'ratings', 'wb')
	ft = open(moviedir+'dates', 'wb')
	fb = open(moviedir+'movieindex_BOUNDS', 'wb')
	fi = open(moviedir+'index', 'wb')
	
	fd.movieIDsForMovies[index].dump(fm)
	fd.userIDs[index].dump(fu)
	fd.movieRatings[index].dump(fr)
	fd.datesByMovie[index].dump(ft)
	movieindex_BOUNDS.dump(fb)
	index.dump(fi)
	fm.close()
	fu.close()
	fr.close()
	ft.close()
	fb.close()
	fi.close()
	
	# Make user-indexed subset
	print('Sorting the index by user...')
	uindex = N.argsort(fd.userIDsForUsers[index])
	userdir = outputdir+'user_indexed/'
	try:
		mkdir(userdir)
	except:
		print('user indexed directory already exists...')
	userindex_BOUNDS = makeBoundedIndex(fd.userIDsForUsers[index])
	print('Dumping user files...')
	fm = open(userdir+'movieid', 'wb')
	fu = open(userdir+'userid', 'wb')
	fr = open(userdir+'ratings', 'wb')
	ft = open(userdir+'dates', 'wb')
	fb = open(userdir+'userindex_BOUNDS', 'wb')
	fi = open(userdir+'index', 'wb')
	
	fd.movieIDs[uindex].dump(fm)
	fd.userIDsForUsers[uindex].dump(fu)
	fd.userRatings[uindex].dump(fr)
	fd.datesByUser[uindex].dump(ft)
	userindex_BOUNDS.dump(fb)
	uindex.dump(fi)
	fm.close()
	fu.close()
	fr.close()
	ft.close()
	fi.close()
	
	del index, uindex
	
def makeMinisets(fd, numsets, index=None, setsize=5000000, probedir="/home/alex/workspace/flix/data/bin_data/arrays/minisets/"):
	"""
	Wrapper function for creating multiple minisets.
	"""
	if index==None:
		index = N.arange(fd.numratings)
	else:
		assert len(index) == len(fd.movieRatings)
	
	print('Shuffling the ratings...')
	N.random.shuffle(index)
	
	print('Gathering all extra information...')
	if not hasattr(fd, 'datesByMovie'):
		fd.getDatesSortedByMovie()
	if not hasattr(fd, 'movieIDsForMovies'):
		fd.getMovieIDsForSortedMovies()
	if not hasattr(fd, 'datesByUser'):
		fd.getDatesSortedByUser()
	if not hasattr(fd, 'userIDsForUsers'):
		fd.getUserIDsForSortedUsers()
		
			
	for i in range(numsets):
		print('Saving miniset %d' % (i+1))
		outputdir = probedir + str(i+1) + '/'
		try: 
			mkdir(outputdir)
		except:
			print('Directory already exists...')
		saveMiniset(fd, index[setsize*i:setsize*(i+1)], outputdir)
	
	f = open(probedir+'fullshuffledinex', 'wb')
	index.dump(f)
	f.close()
	
	del index
		
def makeBoundedIndex(arrayToIndex):
	"""
	Given a sorted array, figure out the 
	bounds of each item (we're assuming a lot of
	redundancy)
	"""
	
	uniqueItems = N.unique(arrayToIndex)
	numItems = uniqueItems.size
	tmp = N.searchsorted(arrayToIndex, uniqueItems)
	tmp = N.r_[tmp, len(arrayToIndex)]
	index = N.zeros((numItems, 2), dtype='int32')
	
	for i in range(numItems):
		index[i] = (tmp[i], tmp[i+1])
	
	return index
	
def dateStringToEpoch(datestring):
	"""Convert a string like '2005-12-19' into a unix epoch"""
	from time import mktime
	year, month, day = datestring.split('-')
	epoch = mktime((int(year), int(month), int(day), 0, 0, 0, 0, 0, 0))
	return int(epoch)
	
def readProbeFile(probefileloc):
	f = open(probefileloc, 'rt')
	
	print('Reading the data file..')
	testdata = f.readlines()
	# We have a list of strings of one of two forms:
	# "112:" header for a list of ratings for movie 112
	# "1046323,2005-12-19" user 1046323, rating at 2005-12-19

	# First detect where the movie headers are
	movieheaders = []
	filetype = 'probe'
	print('Parsing the data file...')
	for i, line in enumerate(testdata):
		testdata[i] = testdata[i][:-1] # get rid of the \n's.
		if ':' in line: 
			movieheaders.append(i)

		# Do a little formatting for ourselves
		else:
			if ',' in line:
				filetype = 'qualifying'
				tmp = line.split(',')
				tmp[0] = int(tmp[0])
				tmp[1] = dateStringToEpoch(tmp[1])
			else:
				tmp = int(line)

			testdata[i] = tmp

	print('Formatting the rating information...')
	info = []
	movieid = []
	
	for i in range(len(testdata)):
		if isinstance(testdata[i], str):
			tmpmovieid = int(testdata[i][:-1])
		if isinstance(testdata[i], list) or isinstance(testdata[i], int):
			info.append(testdata[i])
			movieid.append(tmpmovieid)


	info = N.vstack(info)
	movieid = N.vstack(movieid)
	info = N.hstack((movieid, info))
	return info.astype(N.uint32)
	
def invertSet(minval, maxval, origset):
	"""
	Invert a set, origset, such that
	the new set contains ever integer from
	min to max, except the elements in origset
	
	origset must be sorted, and fit between minval and maxval
	
	"""
	newset = N.zeros(maxval-minval-origset.size, dtype=N.int64)
	
	insert = N.arange(minval, origset[0])
	newset[minval:origset[0]] = insert
	count = insert.size
	
	for i in N.arange(len(origset)-1)+1:
		insert = N.arange(origset[i-1]+1, origset[i])
		newset[count:insert.size+count] = insert
		count += insert.size
		
	insert = N.arange(origset[i]+1, maxval)
	newset[count:insert.size+count] = insert
	
	return newset
