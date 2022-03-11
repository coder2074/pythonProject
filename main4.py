from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.collocations import *
from nltk import FreqDist, classify, NaiveBayesClassifier

import re, string, random
# remove hyperlinks, remove tags
# stop words
# lemmatize

# START DEFINE CLASSES

class Review:
  def __init__(self, sentences):
    self.sentences = sentences

class Sentence:
  def __init__(self, orig, words):
    self.orig = orig
    self.tokens = words

# END DEFINE CLASSES

# USE THE CLASS


# END USE THE CLASS

def removeStopwords_removePunctuation_lowercase_lemmatize(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)






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
	outfile = open(outfilename, 'w')

	for line in file1:
		sent_text = nltk.sent_tokenize(line)  # this gives us a list of sentences
		for s in sent_text:
			outfile.write(s)
			outfile.write("\n")
	outfile.close()

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

def fixContractionsAndTokenize(infilename):
	import contractions

	linetokens = []

	file1 = open(infilename)

	for line in file1:
		fixedContractionsInLine = contractions.fix(line)
		linetokens.append(word_tokenize(fixedContractionsInLine))
	#	for word in line.split():
			# using contractions.fix to expand the shotened words
	#		linetokens.append(contractions.fix(word))

	return linetokens

def fixContractions(infilename):
	import contractions

	myArray = []

	file1 = open(infilename)

	for line in file1:
		fixedContractionsInLine = contractions.fix(line)
		myArray.append(fixedContractionsInLine)
		#	for word in line.split():
			# using contractions.fix to expand the shotened words
	#		linetokens.append(contractions.fix(word))

	return myArray

def getCollocations(myArray):
	import nltk

	listToStr = ' '.join([str(elem) for elem in myArray])
	tokens = word_tokenize(listToStr)

	bigram_measures = nltk.collocations.BigramAssocMeasures()
	trigram_measures = nltk.collocations.TrigramAssocMeasures()

	# change this to read in your data
	bifinder = BigramCollocationFinder.from_words(tokens)
	trifinder = TrigramCollocationFinder.from_words(tokens)

	# only bigrams that appear 3+ times
	bifinder.apply_freq_filter(2)
	trifinder.apply_freq_filter(2)

	# return the 10 n-grams with the highest PMI
	ngramsBi = bifinder.nbest(bigram_measures.pmi, 10)
	ngramsTri = trifinder.nbest(trigram_measures.pmi, 10)

	print('ngrams BI')
	print(ngramsBi)
	print('ngrams Tri')
	print(ngramsTri)

	return ngramsBi


positivereviews = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/20220110-echo4-downloadedReviews-working-positive.txt'
negativereviews = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/20220110-echo4-downloadedReviews-working-negative.txt'

nopunctnostopwordposreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/noPunctNoStopWordPosReviewFilename.txt'
nopunctnostopwordnegreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/noPunctNoStopWordNegReviewFilename.txt'

#
# removepunctandstopwords(positivereviews, nopunctnostopwordposreviewfilename)
# removepunctandstopwords(negativereviews, nopunctnostopwordnegreviewfilename)
#
# lemmatizeposreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/lemmatizePosReviewFilename.txt'
# lemmatizenegreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/lemmatizeNegReviewFilename.txt'
#
# lemmatize(nopunctnostopwordposreviewfilename, lemmatizeposreviewfilename)
# lemmatize(nopunctnostopwordnegreviewfilename, lemmatizenegreviewfilename)
#
sentencesposreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/sentencesPosReviewFilename.txt'
sentencesnegreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/sentencesNegReviewFilename.txt'
#
# toSentences(positivereviews, sentencesposreviewfilename)
# toSentences(negativereviews, sentencesnegreviewfilename)
#
# sentencessentimentposreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/sentencesSentimentPosReviewFilename.txt'
# sentencessentimentnegreviewfilename = 'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/sentencesSentimentNegReviewFilename.txt'
#
# doSentiment(sentencesposreviewfilename, sentencessentimentposreviewfilename)
# doSentiment(sentencesnegreviewfilename, sentencessentimentnegreviewfilename)


#toSentences(positivereviews, sentencesposreviewfilename)
toSentences(negativereviews, sentencesnegreviewfilename)

textArray = fixContractions(sentencesnegreviewfilename)
print('Fixed Contractions and tokenized:')
print(textArray)

ngrams = getCollocations(textArray)

stop_words = stopwords.words('english')

negative_cleaned_tokens_list = []

#for tokens in tweetTokens:
#        negative_cleaned_tokens_list.append(removeStopwords_removePunctuation_lowercase_lemmatize(tokens, stop_words))

print('remove_stopwords_clean_lemmatize:')
print(negative_cleaned_tokens_list)

#all_neg_words = get_all_words(negative_cleaned_tokens_list)

#freq_dist_pos = FreqDist(all_neg_words)
#print(freq_dist_pos.most_common(10))
#
# negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
#
# positive_dataset = [(tweet_dict, "Positive")
# 					 for tweet_dict in positive_tokens_for_model]
#
# negative_dataset = [(tweet_dict, "Negative")
# 					 for tweet_dict in negative_tokens_for_model]
#
# dataset = positive_dataset + negative_dataset
#
# random.shuffle(dataset)
#
# train_data = dataset[:7000]
# test_data = dataset[7000:]
#
# classifier = NaiveBayesClassifier.train(train_data)
#
# print("Accuracy is:", classify.accuracy(classifier, test_data))
#
# print(classifier.show_most_informative_features(10))
#
# custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
#
# custom_tokens = remove_noise(word_tokenize(custom_tweet))
#
# print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))