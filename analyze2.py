import json
import re
import string

import jsonpickle as jsonpickle
from nltk.collocations import *
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter


class Review:
    def __init__(self, review_str):
        self.review_id = 0
        self.review_str_working = review_str
        self.review_str_orig = review_str
        self.sentences = []
        self.review_tokens = []

    def to_string(self):
        my_str = ''

        for my_token in self.review_tokens:
            my_str = my_str + ' ' + my_token.token_text_working

        return my_str


class Sentence:
    def __init__(self, sentence_id, review_id, sentence_text_orig, sentence_text_working):
        self.sentence_id = sentence_id
        self.review_id = review_id
        self.sentence_text_orig = sentence_text_orig
        self.sentence_text_working = sentence_text_working
        self.sentence_tokens_array = []


class Token:
    def __init__(self, token_id, sentence_id, token_text_orig, token_text_working):
        self.token_id = token_id
        self.sentence_id = sentence_id
        self.token_text_orig = token_text_orig
        self.token_text_working = token_text_working


class Summary:
    def __init__(self):
        self.reviews_array = []
        self.sentences_array = []
        self.sentence_id_to_sentence_map = {}
        self.all_words_array = []
        self.word_to_pos_map = {}
        self.words_that_are_nouns_array = []
        self.words_that_are_verbs_array = []
        self.words_that_are_adjectives_array = []
        self.word_to_sentences_map = {}
        self.collocation_bi_raw_array = []
        self.collocation_tri_raw_array = []
        self.collocation_bi_key_to_sentence_ids_array_map = {}
        self.collocation_tri_key_to_sentence_ids_array_map = {}
        self.collocation_bi_key_to_sentences_map = {}
        self.collocation_tri_key_to_sentences_map = {}
        self.most_occur_all_word_to_sentences_map = {}
        self.most_occur_nouns_word_to_sentences_map = {}
        self.most_occur_verbs_word_to_sentences_map = {}
        self.most_occur_adjectives_word_to_sentences_map = {}

    @staticmethod
    def find_sentences(most_common_word_to_count_map, word_to_sentences_map):
        new_word_to_sentences_map = {}

        for word_to_count in most_common_word_to_count_map:
            the_word = word_to_count[0]
            sentences = word_to_sentences_map[the_word]
            new_word_to_sentences_map[the_word] = sentences

        return new_word_to_sentences_map
        # end most occurences

    @staticmethod
    def print_occur_to_sentence_map(occur_to_sentence_map):
        for occur_to_sentence in occur_to_sentence_map:
            sentences = occur_to_sentence_map[occur_to_sentence]
            print('The word is: ' + occur_to_sentence)
            for sentence_5 in sentences:
                print(sentence_5.sentence_text_working)

    @staticmethod
    def filter_out_most_common(most_occurs):
        new_most_occurs = []
        for most_occur in most_occurs:
            if most_occur[1] >= 2:
                new_most_occurs.append(most_occur)
        return new_most_occurs

    @staticmethod
    def print_collocation_key_to_sentences_map(collocation_key_to_sentences_map):
        for my_key in collocation_key_to_sentences_map:
            print(my_key)
            my_sentences = collocation_key_to_sentences_map[my_key]
            for my_sentence in my_sentences:
                print(my_sentence.sentence_text_working)
        print('END COLLOCATION BI-SENTENCES')


# END DEFINE CLASSES

class SummaryWeb:
    def __init__(self):
        self.collocation_bi_key_to_sentences_map = {}
        self.collocation_tri_key_to_sentences_map = {}
        self.most_occur_all_word_to_sentences_map = {}
        self.most_occur_nouns_word_to_sentences_map = {}
        self.most_occur_verbs_word_to_sentences_map = {}
        self.most_occur_adjectives_word_to_sentences_map = {}

# USE THE CLASS

def load_reviews(infilename):
    my_array = []

    file1 = open(infilename)
    i = 99
    for line in file1:
        i = i + 1
        my_review = Review(line)
        my_review.review_id = i
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


def make_tokens(my_review, my_token_seq, my_stopwords, my_summary):
    review_tokens = []

    for my_sentence2 in my_review.sentences:
        sentence_tokens_array = []
        my_sentence_text = my_sentence2.sentence_text_working

        # review_str = my_review.review_str_working
        words_array = word_tokenize(my_sentence_text)
        pos_tag_array = pos_tag(words_array)

        for word_text, tag in pos_tag_array:
            word_text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                               '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', word_text)
            word_text = re.sub("(@[A-Za-z0-9_]+)", "", word_text)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            word_text = lemmatizer.lemmatize(word_text, pos)
            word_text = word_text.lower()

            word_text = word_text.replace('.', '')

            if len(word_text) > 0 and word_text not in string.punctuation and word_text not in my_stopwords:
                my_token_seq = my_token_seq + 1
                my_token = Token(token_seq, my_sentence2.sentence_id, word_text, word_text)
                sentence_tokens_array.append(my_token)
                review_tokens.append(my_token)
                my_summary.word_to_pos_map[word_text] = tag
                my_summary.all_words_array.append(word_text)

                found_sentences = []
                if my_summary.word_to_sentences_map.__contains__(word_text):
                    found_sentences = my_summary.word_to_sentences_map[word_text]
                # need to make sure the sentence does not already exist in the list
                found_id = 0
                for asdf in found_sentences:
                    if asdf.sentence_id == my_sentence2.sentence_id:
                        found_id = 1
                if not found_id:
                    found_sentences.append(my_sentence2)

                my_summary.word_to_sentences_map[word_text] = found_sentences

                # this the final clean can add to summary
                if tag.startswith("NN"):
                    my_summary.words_that_are_nouns_array.append(word_text)
                elif tag.startswith('VB'):
                    my_summary.words_that_are_verbs_array.append(word_text)
                else:
                    my_summary.words_that_are_adjectives_array.append(word_text)
        my_sentence2.sentence_tokens_array = sentence_tokens_array

    my_review.review_tokens = review_tokens


def make_sentences(my_summary):
    i = 199
    for my_review in my_summary.reviews_array:
        my_review_str = my_review.review_str_working
        sentences_text = sent_tokenize(my_review_str)
        for sentence_text in sentences_text:
            i = i + 1
            my_sentence2 = Sentence(i, my_review.review_id, sentence_text, sentence_text)
            my_summary.sentences_array.append(my_sentence2)
            my_summary.sentence_id_to_sentence_map[my_sentence2.sentence_id] = my_sentence2
            my_review.sentences.append(my_sentence2)


def set_raw_collocations(my_summary):
    import nltk

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    # change this to read in your data
    bifinder = BigramCollocationFinder.from_words(my_summary.all_words_array)
    trifinder = TrigramCollocationFinder.from_words(my_summary.all_words_array)

    # only bigrams that appear 3+ times
    bifinder.apply_freq_filter(3)
    trifinder.apply_freq_filter(2)

    # return the 10 n-grams with the highest PMI
    ngrams_bi = bifinder.nbest(bigram_measures.pmi, 3)
    ngrams_tri = trifinder.nbest(trigram_measures.pmi, 3)

    my_summary.collocation_bi_raw_array = ngrams_bi
    my_summary.collocation_tri_raw_array = ngrams_tri


def set_collocation_maps(my_summary):
    # try now to concatenate words so that i can search the collocations
    collocation_bi_key_to_sentence_ids_array_map = {}
    collocation_tri_key_to_sentence_ids_array_map = {}
    # for idx, my_sentence in enumerate(summary.sentences_array):
    #    print(idx, my_token)
    for sentence in my_summary.sentences_array:
        sentence_tokens_array = sentence.sentence_tokens_array
        for idx, tokens_array in enumerate(sentence_tokens_array):
            tri_str = ''
            bi_str = ''
            length = len(sentence_tokens_array)
            location = length - idx
            if location >= 3:
                tri_str = tokens_array.token_text_working + sentence_tokens_array[idx + 1].token_text_working + sentence_tokens_array[
                    idx + 2].token_text_working
                if collocation_tri_key_to_sentence_ids_array_map.__contains__(tri_str):
                    trimap_existing_sentence_id_list = collocation_tri_key_to_sentence_ids_array_map[tri_str]
                else:
                    trimap_existing_sentence_id_list = []
                trimap_existing_sentence_id_list.append(sentence.sentence_id)
                collocation_tri_key_to_sentence_ids_array_map[tri_str] = trimap_existing_sentence_id_list
            if location >= 2:
                bi_str = tokens_array.token_text_working + sentence_tokens_array[idx + 1].token_text_working
                if collocation_bi_key_to_sentence_ids_array_map.__contains__(bi_str):
                    bimap_existing_sentence_id_list = collocation_bi_key_to_sentence_ids_array_map[bi_str]
                else:
                    bimap_existing_sentence_id_list = []
                bimap_existing_sentence_id_list.append(sentence.sentence_id)
                collocation_bi_key_to_sentence_ids_array_map[bi_str] = bimap_existing_sentence_id_list
    print('BIMAP')
    print(collocation_bi_key_to_sentence_ids_array_map)
    print('TRIMAP')
    print(collocation_tri_key_to_sentence_ids_array_map)

    my_summary.collocation_bi_key_to_sentence_ids_array_map = collocation_bi_key_to_sentence_ids_array_map
    my_summary.collocation_tri_key_to_sentence_ids_array_map = collocation_tri_key_to_sentence_ids_array_map
    # end try now to concatenate words so that i can search the collocations


# now get the sentences for collocations
def get_collocation_key_to_sentences_map(bi_or_tri, my_summary):
    collocation_sent_ids = []
    collocation_key_to_sentences_map = {}
    collocation_keys_to_use = []  # default
    collocation_key_to_sentence_ids_array_map = {}  # default
    if bi_or_tri == 'bi':
        collocation_keys_to_use = my_summary.collocation_bi_raw_array
        collocation_key_to_sentence_ids_array_map = my_summary.collocation_bi_key_to_sentence_ids_array_map
    elif bi_or_tri == 'tri':
        collocation_keys_to_use = my_summary.collocation_tri_raw_array
        collocation_key_to_sentence_ids_array_map = my_summary.collocation_tri_key_to_sentence_ids_array_map
    collocation_key = ''
    for my_gram in collocation_keys_to_use:
        if bi_or_tri == 'bi':
            collocation_key = my_gram[0] + my_gram[1]
        elif bi_or_tri == 'tri':
            collocation_key = my_gram[0] + my_gram[1] + my_gram[2]
        gram_sent_id_array = collocation_key_to_sentence_ids_array_map[collocation_key]
        temp_sent = []
        for sent_id in gram_sent_id_array:
            sentence_3 = my_summary.sentence_id_to_sentence_map[sent_id]
            if not collocation_sent_ids.__contains__(sent_id):
                collocation_sent_ids.append(sent_id)
                temp_sent.append(sentence_3)
        collocation_key_to_sentences_map[collocation_key] = temp_sent

    return collocation_key_to_sentences_map
    # end get the sentences BI


def print_summary(my_summary):
    print('ngrams BI')
    print(my_summary.collocation_bi_raw_array)
    print('ngrams Tri')
    print(my_summary.collocation_tri_raw_array)
    print('word_to_pos_map')
    print(my_summary.word_to_pos_map)

    print('PRINTING to_string')
    for my_review in my_summary.reviews_array:
        print(my_review.to_string())
    print('DONE')


def do_collocations(my_summary):
    set_raw_collocations(my_summary)

    set_collocation_maps(my_summary)

    print('START COLLOCATION BI-SENTENCES')
    my_summary.collocation_bi_key_to_sentences_map = get_collocation_key_to_sentences_map('bi', my_summary)
    my_summary.print_collocation_key_to_sentences_map(my_summary.collocation_bi_key_to_sentences_map)

    print('START COLLOCATION TRI-SENTENCES')
    my_summary.collocation_tri_key_to_sentences_map = get_collocation_key_to_sentences_map('tri', my_summary)
    my_summary.print_collocation_key_to_sentences_map(my_summary.collocation_tri_key_to_sentences_map)


def do_count_occurences(my_summary):
    # Pass the split_it list to instance of Counter class.
    counter_all = Counter(my_summary.all_words_array)

    counter_nouns = Counter(my_summary.words_that_are_nouns_array)
    counter_verbs = Counter(my_summary.words_that_are_verbs_array)
    counter_adjectives = Counter(my_summary.words_that_are_adjectives_array)

    # most_common() produces k frequently encountered
    # input values and their respective counts.
    word_to_numoccur_array_overall = counter_all.most_common(4)
    word_to_numoccur_array_overall = my_summary.filter_out_most_common(word_to_numoccur_array_overall)
    word_to_numoccur_array_nouns = counter_nouns.most_common(4)
    word_to_numoccur_array_nouns = my_summary.filter_out_most_common(word_to_numoccur_array_nouns)
    word_to_numoccur_array_verbs = counter_verbs.most_common(4)
    word_to_numoccur_array_verbs = my_summary.filter_out_most_common(word_to_numoccur_array_verbs)
    word_to_numoccur_array_adjectives = counter_adjectives.most_common(4)
    word_to_numoccur_array_adjectives = my_summary.filter_out_most_common(word_to_numoccur_array_adjectives)

    print('word_to_numoccur_array_overall')
    print(word_to_numoccur_array_overall)

    print('word_to_numoccur_array_nouns')
    print(word_to_numoccur_array_nouns)

    print('word_to_numoccur_array_verbs')
    print(word_to_numoccur_array_verbs)

    print('word_to_numoccur_array_adjectives')
    print(word_to_numoccur_array_adjectives)

    # now do most occurances
    print('occur all sentences')
    my_summary.most_occur_all_word_to_sentences_map = my_summary.find_sentences(word_to_numoccur_array_overall, my_summary.word_to_sentences_map)
    my_summary.most_occur_nouns_word_to_sentences_map = my_summary.find_sentences(word_to_numoccur_array_nouns, my_summary.word_to_sentences_map)
    my_summary.most_occur_verbs_word_to_sentences_map = my_summary.find_sentences(word_to_numoccur_array_verbs, my_summary.word_to_sentences_map)
    my_summary.most_occur_adjectives_word_to_sentences_map = my_summary.find_sentences(word_to_numoccur_array_adjectives, my_summary.word_to_sentences_map)


def print_occurences(my_summary):
    print('all occurences')
    my_summary.print_occur_to_sentence_map(my_summary.most_occur_all_word_to_sentences_map)
    print('noun occurences')
    my_summary.print_occur_to_sentence_map(my_summary.most_occur_nouns_word_to_sentences_map)
    print('verb occurences')
    my_summary.print_occur_to_sentence_map(my_summary.most_occur_verbs_word_to_sentences_map)
    print('adjective occurences')
    my_summary.print_occur_to_sentence_map(my_summary.most_occur_adjectives_word_to_sentences_map)


summary = Summary()

sentences_neg_review_filename = \
    'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/20220110-echo4-downloadedReviews-working-negative2.txt'
sentences_pos_review_filename = \
    'C:/Users/ram82/Google Drive/Freelance/smarthomewarrior/Data/working/20220110-echo4-downloadedReviews-working-positive2.txt'
summary.reviews_array = load_reviews(sentences_pos_review_filename)

print('MAKE LOWERCASE')
for review in summary.reviews_array:
    make_lowercase(review)

print('FIX CONTRACTIONS')
for review in summary.reviews_array:
    fix_contractions(review)

print('MAKE SENTENCES')
make_sentences(summary)

print('MAKE TOKENS')
token_seq = 1000
stop_words = stopwords.words('english')
# stop_words = []
for review in summary.reviews_array:
    make_tokens(review, token_seq, stop_words, summary)

do_collocations(summary)

do_count_occurences(summary)

print_occurences(summary)

summaryWeb = SummaryWeb()
summaryWeb.collocation_bi_key_to_sentences_map = summary.collocation_bi_key_to_sentences_map
summaryWeb.collocation_tri_key_to_sentences_map = summary.collocation_tri_key_to_sentences_map
summaryWeb.most_occur_all_word_to_sentences_map = summary.most_occur_all_word_to_sentences_map
summaryWeb.most_occur_nouns_word_to_sentences_map = summary.most_occur_nouns_word_to_sentences_map
summaryWeb.most_occur_verbs_word_to_sentences_map = summary.most_occur_verbs_word_to_sentences_map
summaryWeb.most_occur_adjectives_word_to_sentences_map = summary.most_occur_adjectives_word_to_sentences_map

jsonStr = jsonpickle.encode(summaryWeb, unpicklable=False)

print('The SummaryWeb Json')
print(jsonStr)

