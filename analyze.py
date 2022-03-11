import re
import string

from nltk.collocations import *
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter


class Review:
    def __init__(self, review_str):
        self.review_str_working = review_str
        self.review_str_orig = review_str
        self.words = []

    def to_string(self):
        my_str = ''

        token_words = ' '.join([str(elem) for elem in self.words])
        my_str = my_str + token_words

        return my_str


class Summary:
    def __init__(self, reviews, all_tokens, collocation_bigram, collocation_trigram, pos_map, pos_nouns, pos_verbs,
                 pos_adjectives):
        self.reviews = reviews
        self.collocation_bigram = collocation_bigram
        self.collocation_trigram = collocation_trigram
        self.all_tokens = all_tokens
        self.pos_map = pos_map
        self.pos_nouns = pos_nouns
        self.pos_verbs = pos_verbs
        self.pos_adjectives = pos_adjectives


# END DEFINE CLASSES

# USE THE CLASS

def load_reviews(infilename):
    my_array = []

    file1 = open(infilename)

    for line in file1:
        my_review = Review(line)
        my_array.append(my_review)

    return my_array


def fix_contractions(my_review):
    import contractions

    fixed_str = contractions.fix(my_review.review_str_orig)
    my_review.review_str_working = fixed_str


def make_lowercase(my_review):
    fixed_str = my_review.review_str_working
    fixed_str = str(fixed_str).lower()
    my_review.review_str_working = fixed_str


def make_words(my_review):
    review_str = my_review.review_str_working
    words = word_tokenize(review_str)
    my_review.words = words


def remove_stopwords_remove_punctuation_lowercase_lemmatize(my_summary, my_review, my_stopwords):
    words = my_review.words

    cleaned_tokens = []

    for token, tag in pos_tag(words):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        token = token.lower()

        if len(token) > 0 and token not in string.punctuation and token not in my_stopwords:
            cleaned_tokens.append(token)
            my_summary.pos_map[token] = tag
            my_summary.all_tokens.append(token)
            # this the final clean can add to summary
            if tag.startswith("NN"):
                my_summary.pos_nouns.append(token)
            elif tag.startswith('VB'):
                my_summary.pos_verbs.append(token)
            else:
                my_summary.pos_adjectives.append(token)

            # end this the final clean can add to summary

        my_review.words = cleaned_tokens


def get_collocations(my_summary):
    import nltk

    # all_the_tokens = []
    #
    # for my_review in reviews:
    #     words = my_review.words
    #     for word in words:
    #         all_the_tokens.append(word)
    #
    # my_summary.all_tokens = all_the_tokens

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    # change this to read in your data
    bifinder = BigramCollocationFinder.from_words(my_summary.all_tokens)
    trifinder = TrigramCollocationFinder.from_words(my_summary.all_tokens)

    # only bigrams that appear 3+ times
    bifinder.apply_freq_filter(2)
    trifinder.apply_freq_filter(2)

    # return the 10 n-grams with the highest PMI
    ngrams_bi = bifinder.nbest(bigram_measures.pmi, 10)
    ngrams_tri = trifinder.nbest(trigram_measures.pmi, 10)

    my_summary.collocation_bigram = ngrams_bi
    my_summary.collocation_trigram = ngrams_tri


def print_summary(my_summary):
    print('ngrams BI')
    print(my_summary.collocation_bigram)
    print('ngrams Tri')
    print(my_summary.collocation_trigram)
    print('pos_map')
    print(my_summary.pos_map)

    print('PRINTING to_string')
    for my_review in my_summary.reviews:
        print(my_review.to_string())
    print('DONE')


summary = Summary(reviews=[], all_tokens=[], collocation_bigram=[], collocation_trigram=[], pos_map={}, pos_nouns=[],
                  pos_verbs=[], pos_adjectives=[])

sentences_neg_review_filename = \
    'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/20220110-echo4-downloadedReviews-working-negative2.txt'
sentences_pos_review_filename = \
    'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/20220110-echo4-downloadedReviews-working-positive2.txt'
summary.reviews = load_reviews(sentences_neg_review_filename)

print('MAKE LOWERCASE')
for review in summary.reviews:
    make_lowercase(review)

print('FIX CONTRACTIONS')
for review in summary.reviews:
    fix_contractions(review)

print('MAKE WORDS')
for review in summary.reviews:
    make_words(review)

print('LEMMITIZE WORDS')
stop_words = stopwords.words('english')
# stop_words = []

for review in summary.reviews:
    remove_stopwords_remove_punctuation_lowercase_lemmatize(summary, review, stop_words)

get_collocations(summary)

print_summary(summary)

# Pass the split_it list to instance of Counter class.
counter_all = Counter(summary.all_tokens)

counter_nouns = Counter(summary.pos_nouns)
counter_verbs = Counter(summary.pos_verbs)
counter_adjectives = Counter(summary.pos_adjectives)

# most_common() produces k frequently encountered
# input values and their respective counts.
most_occur_all = counter_all.most_common(4)
most_occur_nouns = counter_nouns.most_common(4)
most_occur_verbs = counter_verbs.most_common(4)
most_occur_adjectives = counter_adjectives.most_common(4)

print('most_occur_all')
print(most_occur_all)

print('most_occur_nouns')
print(most_occur_nouns)

print('most_occur_verbs')
print(most_occur_verbs)

print('most_occur_adjectives')
print(most_occur_adjectives)
