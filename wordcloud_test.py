import multidict as multidict

import numpy as np

import re
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def getFrequencyDictForText(sentence):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for text in sentence.split(" "):
        if re.match("a|the|an|the|to|in|for|of|or|by|with|is|on|that|be", text):
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])

    fullTermsDict.add('dog food', 5)
    fullTermsDict.add('great buy', 4)
    fullTermsDict.add('good service', 3)
    return fullTermsDict


def makeImage(text):
    wc = WordCloud(background_color="white", max_words=1000)
    # generate word cloud
    wc.generate_from_frequencies(text)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

text = 'expensive cat expensive expensive cow cow expensive expensive expensive expensive expensive'
makeImage(getFrequencyDictForText(text))