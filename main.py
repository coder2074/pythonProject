# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from numpy import array, uint8
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

# Load in the dataframe
df = pd.read_csv("data/winemag-data-130k-v2.csv", index_col=0)

# Looking at first 5 rows of the dataset
df.head()

print("There are {} observations and {} features in this dataset. \n".format(df.shape[0], df.shape[1]))

print("There are {} types of wine in this dataset such as {}... \n".format(len(df.variety.unique()),
                                                                           ", ".join(df.variety.unique()[0:5])))

print("There are {} countries producing wine in this dataset such as {}... \n".format(len(df.country.unique()),
                                                                                      ", ".join(
                                                                                          df.country.unique()[0:5])))
df[["country", "description", "points"]].head()

# Groupby by country
country = df.groupby("country")

# Summary statistic of all countries
country.describe().head()

country.mean().sort_values(by="points", ascending=False).head()

# # ----------------------------------------------------------------------------
# plt.figure(figsize=(15,10))
# country.size().sort_values(ascending=False).plot.bar()
# plt.xticks(rotation=50)
# plt.xlabel("Country of Origin")
# plt.ylabel("Number of Wines")
# plt.show()
#
# # ----------------------------------------------------------------------------
# plt.figure(figsize=(15,10))
# country.max().sort_values(by="points",ascending=False)["points"].plot.bar()
# plt.xticks(rotation=50)
# plt.xlabel("Country of Origin")
# plt.ylabel("Highest point of Wines")
# plt.show()

# # ------------------------- WORDCLOUD -------------------------------------

# Start with one review:
text = df.description[0]

# lower max_font_size, change the maximum number of word and lighten the background:
# wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()
# wordcloud.to_file("data/img/first_review.png")
# # ------------------------- END WORDCLOUD -------------------------------------

# # ------------------------- WORDCLOUD MASK -------------------------------------
wine_mask = np.array(Image.open("data/img/wine_mask.png"))

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val

# Transform your mask into a new one that will work with the function:
transformed_wine_mask = np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)

for i in range(len(wine_mask)):
    transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# Create a word cloud image
wc = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask,
               stopwords=stopwords, contour_width=3, contour_color='firebrick')

# Generate a wordcloud
wc.generate(text)

# store to file
wc.to_file("data/img/wine.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()



# # ------------------------- END WORDCLOUD MASK -------------------------------------
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
