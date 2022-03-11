from nltk.corpus import stopwords

def removepunctandstopwords(infilename, outfilename):
	# remove stop words and punctuation
	# initializing punctuations string
	punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

	# word_tokenize accepts
	# a string as an input, not a file.
	stop_words = set(stopwords.words('english'))
	file1 = open(infilename)

	appendfile = open(outfilename, 'a')
	for line in file1:

		# remove punctuation
		for character in line:
			if character in punc:
				line = line.replace(character, "")
		# end remove punctuation

		# remove stopwords
		words = line.split()
		for r in words:
			if not r in stop_words:
				appendfile.write(" " + r)
		appendfile.write("\n")
	appendfile.close()

def lemmatize(infilename, outfilename):
	# import these modules
	from nltk.stem import WordNetLemmatizer

	lemmatizer = WordNetLemmatizer()

	file1 = open(infilename)
	appendfile = open(outfilename, 'a')

	for line in file1:
		words = line.split()
		for r in words:
			lemmatizedword = lemmatizer.lemmatize(r)
			appendfile.write(" " + lemmatizedword)
		appendfile.write("\n")
	appendfile.close()

def toSentences(infilename, outfilename):
	# import these modules
	import nltk

	file1 = open(infilename)
	appendfile = open(outfilename, 'a')

	for line in file1:
		sent_text = nltk.sent_tokenize(line)  # this gives us a list of sentences
		for s in sent_text:
			appendfile.write(s)
			appendfile.write("\n")
	appendfile.close()

def doSentiment(infilename, outfilename):
	# import these modules
	import nltk
	nltk.download('vader_lexicon')
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	sid = SentimentIntensityAnalyzer()
	# end imports

	file1 = open(infilename)
	appendfile = open(outfilename, 'a')

	for line in file1:
		sentimentScore = sid.polarity_scores(line)
		appendfile.write(line + " " + str(sentimentScore))
		appendfile.write("\n");
	appendfile.close()

positivereviews = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/20220110-echo4-downloadedReviews-working-positive.txt'
negativereviews = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/20220110-echo4-downloadedReviews-working-negative.txt'

nopunctnostopwordposreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/noPunctNoStopWordPosReviewFilename.txt'
nopunctnostopwordnegreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/noPunctNoStopWordNegReviewFilename.txt'

removepunctandstopwords(positivereviews, nopunctnostopwordposreviewfilename)
removepunctandstopwords(negativereviews, nopunctnostopwordnegreviewfilename)

lemmatizeposreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/lemmatizePosReviewFilename.txt'
lemmatizenegreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/lemmatizeNegReviewFilename.txt'

lemmatize(nopunctnostopwordposreviewfilename, lemmatizeposreviewfilename)
lemmatize(nopunctnostopwordnegreviewfilename, lemmatizenegreviewfilename)

sentencesposreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/sentencesPosReviewFilename.txt'
sentencesnegreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/sentencesNegReviewFilename.txt'

toSentences(positivereviews, sentencesposreviewfilename)
toSentences(negativereviews, sentencesnegreviewfilename)

sentencessentimentposreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/sentencesSentimentPosReviewFilename.txt'
sentencessentimentnegreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/sentencesSentimentNegReviewFilename.txt'

doSentiment(sentencesposreviewfilename, sentencessentimentposreviewfilename)
doSentiment(sentencesnegreviewfilename, sentencessentimentnegreviewfilename)