# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from numpy import array, uint8
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

#Loading NLTK
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load in the dataframe
#df = pd.read_csv("data/winemag-data-130k-v2.csv", index_col=0)
# Looking at first 5 rows of the dataset
#df.head()

# ----------------------- SENTENCE TOKENIZATION ---------------------------------------------
from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print("SENTENCE TOKENIZATION")
print(tokenized_text)
# ----------------------- SENTENCE TOKENIZATION ---------------------------------------------

# ----------------------- WORD TOKENIZATION ---------------------------------------------
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print("WORD TOKENIZATION")
print(tokenized_word)
# ----------------------- END WORD TOKENIZATION ---------------------------------------------

# ----------------------- FREQUENCY DISTRIBUTION ---------------------------------------------
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print("FREQUENCY DISTRIBUTION")
print(fdist)
fdist.most_common(2)

# Frequency Distribution Plot
fdist.plot(30,cumulative=False)
plt.show()
# ----------------------- END FREQUENCY DISTRIBUTION ---------------------------------------------

# ----------------------- STOPWORDS ---------------------------------------------
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print("STOPWORDS")
print(stop_words)

# remove the stopwords
filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)
# ----------------------- END STOPWORDS ---------------------------------------------

# ----------------------- STEMMING ---------------------------------------------
# Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)
# ----------------------- END STEMMING ---------------------------------------------

# ----------------------- START LEMMATIZATION ---------------------------------------------
#Lexicon Normalization
#performing stemming and Lemmatization

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))
# ----------------------- END LEMMATIZATION ---------------------------------------------

# START POS TAGGING
nltk.download('averaged_perceptron_tagger')
posTaggingSentence = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=nltk.word_tokenize(posTaggingSentence)
print(tokens)
posTagTokens = nltk.pos_tag(tokens)
print(posTagTokens)
# END POS TAGGING

# ------------------ START Sentiment Analysis using Text Classification --------------------

#data = pd.read_csv('data/20220110-echo4-dlRev-allTogetherFromExcel.txt', sep='\n')
#data.head()
#data.info()
#data['sentiment'].value_counts()

# ------------------ END Sentiment Analysis using Text Classification --------------------

# ------------------ START REMOVE STOP WORDS -----------------#

import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# word_tokenize accepts
# a string as an input, not a file.
stop_words = set(stopwords.words('english'))
file1 = open("text.txt")

# Use this to read file content as a stream:
line = file1.read()
words = line.split()
for r in words:
    if not r in stop_words:
        appendFile = open('filteredtext.txt', 'a')
        appendFile.write(" " + r)
        appendFile.close()
# END REMOVE STOP WORDS ----------------- #