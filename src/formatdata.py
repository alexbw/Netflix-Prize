"""
Make binary data from the raw rating data
"""
import numpy as N

def txtdata2bin(dataloc):
	from os import chdir
	chdir(dataloc)
	movieratings, movieids, userids, dates, allusers = makearrays('')
	chdir('../')
	print('Saving movie-indexed data...')
	savearrays('', movieratings, movieids, userids, dates, allusers, index='movie')
	print('Reindexing data for users...')
	movieratings, movieids, userids, dates = reindexmoviearrays(movieratings, movieids, userids, dates)
	print('Saving user-indexed data...')
	savearrays('', movieratings, movieids, userids, dates, allusers, index='user')
	
	
	
def makearrays(dataloc):
	from time import mktime
	nummovies = 17770
	numusers = 480189
	numratings = 100480507
	
	movieratings = N.zeros(numratings, dtype=N.int8)
	movieids = N.zeros(numratings, dtype=N.int16)
	userids = N.zeros(numratings, dtype=N.int32)
	dates = N.zeros(numratings, dtype=N.int32)
	counter = 0
	
	for i in range(nummovies):
		if i % 100 == 0: print('Extracting movie %d' % i)
		f = open(dataloc + 'mv_%07d.txt' % (i+1), 'rt')
		data = f.readlines()
		f.close()
		
		movieid = int(data.pop(0)[:-2])
		inumratings = len(data)
		movieids[counter:counter+inumratings] = movieid
		
		for j in range(inumratings):
			userid, stars, date = data[j][:-1].split(',')
			year, month, day = date.split('-')
			epoch = mktime((int(year), int(month), int(day), 0, 0, 0, 0, 0, 0))
			userids[counter] = int(userid)
			movieratings[counter] = int(stars)
			dates[counter] = epoch
			counter += 1
	
	print('Finding unique users')
	allusers = N.unique(userids)
	
	print('Zero-indexing users and movies')
	movieids -= 1
	convertusers = N.zeros(N.max(allusers)+1, dtype=N.int32)
	convertusers[allusers] = N.r_[0:numusers]
	userids = convertusers[userids]
	
	return movieratings, movieids, userids, dates, allusers
	
def savearrays(savedir, movieratings, movieids, userids, dates, allusers, index='movie'):
	from os import mkdir
	from flixdata import makeBoundedIndex
	try: 
		mkdir(savedir + 'arrays/')
	except:
		pass
		
	if index == 'movie':
		try:
			mkdir(savedir + 'arrays/movie_indexed/')
		except:
			pass
		outdir = savedir+'arrays/movie_indexed/'
	elif index == 'user':
		try:
			mkdir(savedir + 'arrays/user_indexed/')
		except:
			pass
		outdir = savedir+'/arrays/user_indexed/'
	else:
		raise ValueError, 'only user or movie indexing, bud'
	
	f = open(outdir+'ratings', 'wb')
	f.write(movieratings.tostring())
	f.close()
	
	f = open(outdir+'movieids', 'wb')
	f.write(movieids.tostring())
	f.close()
	
	f = open(outdir+'userids', 'wb')
	f.write(userids.tostring())
	f.close()
	
	f = open(outdir+'dates', 'wb')
	f.write(dates.tostring())
	f.close()
	
	if index == 'movie':
		idx = makeBoundedIndex(movieids)
		f = open(outdir+'movieindex_BOUNDS', 'wb')
		f = open(savedir+'arrays/originalUserIDs', 'wb')
		f.write(allusers.tostring())
		f.close()
		
	elif index == 'user':
		idx = makeBoundedIndex(userids)
		f = open(outdir+'userindex_BOUNDS', 'wb')
	idx.dump(f) # we'll only be slicing NumPy arrays, so save it as NumPy
	f.close()
	
	
	
def reindexmoviearrays(movieratings, movieids, userids, dates):
	newindex = N.argsort(userids)
	return movieratings[newindex], movieids[newindex], userids[newindex], dates[newindex]
	
def makeQuizArrays(qualfile):
	"""
	Given the text quiz file, make rating arrays
	"""
	from flixdata import readProbeFile
	qualdata = readProbeFile(qualfile)
	qualdata[:,0] -= 1 
	# get the original user IDs
	origUserIDs = N.fromfile(origuserIDfile, dtype='int32')
	convert = N.r_[0:N.max(origUserIDs)+1]
	convert[origUserIds] = N.r_[0:480189]
	qualdata[:,1] = convert[qualdata[:,1]]
	
	# etc etc