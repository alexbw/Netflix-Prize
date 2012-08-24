import xml.dom.minidom as xml
import string
# 906g must contain movingim
# 245a == movie title
# 246a == alternate movie titles
# 257a country (United States / U.S.), parse before semicolon
# 508a, parse between ';', get job title as first phrase before ',', each member between ','
# 511a actors
# 521a 'MPAA rating: ' then rating
# 650a (multiple) genre
# 655a is genre, 655v == Feature for full movie. 
#		Check that there is an entry for 'Feature' in 655
#		If not, check that no 655 tag contains 'trailer'
# Get all 655 and 650 tags
# 700a name, 700e role
# 546 can indicate language via subtitles
# 655 cannot include 'Trailer'

# TOFIND:
# 	country of origin
#	if it's a trailer or not...

def getMovieMetaData(movieTitle):
	xmldata = getMovieXMLFromLoC(movieTitle)
	xmldata = parseLoCXML(xmldata)
	record = findOurMovieInLoCXML(xmldata)
	
	tagsToPull = ['245a', '246a', '257a', '508a', '511a', '521a', '650a', '655av', '546a']
	metaData = getMetaDataFromRecord(record)
	
def getMovieXMLFromLoC(movieTitle):
	
	import urllib2
	# ADD STUFF ABOUT DATES, TOO
	baseURL = "http://z3950.loc.gov:7090/voyager?operation=searchRetrieve&version=1.1&recordPacking=xml&startRecord=1&maximumRecords=100&query=(dc.description=%22Movie%22)%20and%20(dc.title="
	queryURL = '"%s"' % (movieTitle) + ')'
	fullURL = baseURL + queryURL

	try :
	  xmldata = urllib2.urlopen(fullURL).read()
	except urllib2.HTTPError,e:
	  raise urllib2.HTTPError, "Couldn't get the URL"

	return xmldata
	
def parseLoCXML(xmldata):
	return xml.parseString(xmldata)
	
def findOurMovieInLoCXML(xmldata, movieTitle):
	records = xmldata.getElementsByTagName('record')
	for i, record in enumerate(records):
		if not checkIfRecordIsMovie(record): records[i].remove()
		
		
def checkIfRecordIsMovie(record):
	"""
	The movie will be useless if it does not have
	any tags in 508, 511, 650, 655, 700.
	"""
	rdata = reapRecord(record)
	if 'movingim' not in rdata[906]['g']: return False
	
	checktags = [508, 511, 650, 655, 700]
	itags = 0
	for checktag in checktags:
		if checktag in rdata.keys(): itags += 1
	if itags == 0: return False




def reapRecord(record):
	datafields = record.getElementsByTagName('datafield')
	moviedata = {}
	for datafield in datafields:
		tagdata, tagname = reapDataField(datafield)
		if moviedata.has_key(tagname):
			if type(moviedata[tagname]) is list:
				moviedata[tagname].append(tagdata)
			else:
				moviedata[tagname] = [tagdata]
		else:
			moviedata[tagname] = tagdata
	return moviedata

# NOTE: problem of multiple fields...
def reapDataField(datafield):
	tagdata = {}
	tagname = int(datafield.getAttribute('tag'))
	for subfield in datafield.getElementsByTagName('subfield'):
		tagdata[str(subfield.getAttribute('code'))] = subfield.childNodes[0].toxml()
		
	return tagdata, tagname

	
	# Then we'll iterate through every field,
	# looking for specific MARC tags, which we can access in the following manner:
	
	
	# 906g must contain movingim
	# 245a == movie title
	# 246a == alternate movie titles
	# 257a country (United States / U.S.), parse before semicolon
	# 508a, parse between ';', get job title as first phrase before ',', each member between ','
	# 511a actors
	# 521a 'MPAA rating: ' then rating
	# 650a (multiple) genre
	# 655a is genre
	#		Check that there is an entry for 'Feature' in 655
	#		If not, check that no 655 tag contains 'trailer'
	# Get all 655 and 650 tags
	# 700a name, 700e role
	# 546 can indicate language via subtitles
	
	# 655 cannot include 'Trailer'
	# for datafield in datafields:
	# 	# The MARC ID.
	# 	
	# 	
	# 	
	# 	# I'll have to look up the specific one I want..
	# 	# And then in each datafield tag, there is a <subfield code="a/b/c/..."> tag
	# 	for subfield in datafield.getElementsByTagName('subfield'):
	# 		print subfield.attributes['code'] # the 'X' in <subfield code='X'>
	# 		print subfield.childNodes[0].toxml() # the actual content of the subfield
	# 	
	# Again, all this means nothing unless I get to know which MARC fields 
	# and subfields are relevant to our movie search.
	# And after all this stuff, we should be inserting it into a mysql database
	# that's the next step, for sure.
	# Hopefully this runs in a reasonable amount of time.
	# And hopefully we can find all the movies we need to find.
	

	
	# 906: entry[-8:] == 'movingim' signifies movie
	# 521: MPAA rating
	# 650 genres
	# 655 genres
	# 508: crew
	# 511: cast
	# 520: synopsis (summary, from IMDb)
	# 246 alternate titles

	