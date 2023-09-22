# -*- coding: utf-8 -*-
"""Assignment 5a.ipynb

# **Assignment 5a**
"""

# !pip install nltk

# Libraries required
import random
import nltk
from nltk.util import bigrams, trigrams

# Download the gutenberg and punkt package
nltk.download('punkt')
nltk.download('gutenberg')

print('>>> Deciding the books that will be used...')
# Create list of books' fileIDs we want
books_raw = ['austen-emma.txt',
                 'austen-sense.txt',
                 'blake-poems.txt',
                 'carroll-alice.txt',
                 'chesterton-ball.txt',
                 'chesterton-brown.txt',
                 'melville-moby_dick.txt',
                 'milton-paradise.txt',
                 'shakespeare-hamlet.txt',
                 'whitman-leaves.txt']

# Store the raw text
texts_raw = ''

print('>>> Getting raw texts from the books...')
for book in books_raw:
    books_text_raw = nltk.corpus.gutenberg.raw(book)
    texts_raw += books_text_raw
print('>>> Retrieved all raw texts from the books...')

print('>>> Getting sentences from the texts...')
sentences = nltk.sent_tokenize(texts_raw)
print('>>> Retrieved sentences from the texts...')

# List to store tokens
tokens = []

print('>>> Tokenizing the sentences...')
for sentence in sentences:
    tokens += nltk.word_tokenize(sentence)
print('>>> Sentences tokenized...')

# Create lists of unigrams, bigrams, trigrams
print('>>> Creating lists of unigrams, bigrams and trigrams...')
unigrams = list(nltk.ngrams(tokens, 1))
bigrams = list(nltk.ngrams(tokens, 2))
trigrams = list(nltk.ngrams(tokens, 3))
print('>>> Lists created...')

# Calculate frequencies
print('>>> Calculating the frequencies in each list...')
unigrams_freqs = nltk.FreqDist(unigrams)
bigrams_freqs = nltk.FreqDist(bigrams)
trigrams_freqs = nltk.FreqDist(trigrams)
print('>>> Frequencies calculated...')

# Lists to store words in bigrams
print('>>> Making two lists for each bigram: First list contains the first word, second list contains the second word...')
first_of_bigrams = []
second_of_bigrams = []
for i in bigrams_freqs:
    first_of_bigrams.append(i[0])
    second_of_bigrams.append(i[1])

# Lists to store words in trigrams
print('>>> Making three lists for each trigram: First list contains the first word, second list contains the second word, third list contains the third word...')
first_of_trigrams = []
second_of_trigrams = []
third_of_trigrams = []
for i in trigrams_freqs:
    first_of_trigrams.append(i[0])
    second_of_trigrams.append(i[1])
    third_of_trigrams.append(i[2])

# Function to create sentence from bigrams
def bigram_sentence(words_max):

    # List to store the bigrams created
    my_bigrams = []

    # First word can be any word, it is picked randomly
    next_word = ' '
    while not next_word.isalpha():
        next_word = random.choice(first_of_bigrams)
    my_bigrams.append(next_word)

    # Sentence ends with full stop
    end_token = '.'

    while len(my_bigrams) < words_max:
        # List to store the second words to choose from
        second_word = []
        for i in range(len(first_of_bigrams)):
            # For each word, we will get the word following after it
            if first_of_bigrams[i] == next_word:
                second_word.append(second_of_bigrams[i])

        # Choose the second word randomly
        next_word = random.choice(second_word)
        my_bigrams.append(next_word)

        if next_word == end_token:
            break

    # The sentence created returned with spaces seperating the words
    return (' '.join(my_bigrams))

# Function to create sentence from trigrams
def trigram_sentence(words_max):

    # List to store the trigrams created
    my_trigrams = []

    # Define the two starting words
    first_word = 'We'
    second_word = 'will'
    my_trigrams.append(first_word)
    my_trigrams.append(second_word)

    # Sentence ends with full stop
    end_token = '.'

    while len(my_trigrams) < words_max:
        # List to store the third words to choose from
        third_word = []
        for i in range(len(first_of_trigrams)):
            # For each word (1st word), we will get the word following after it (2nd word)
            # and then again the following word (3rd word)
            if first_of_trigrams[i] == first_word:
                for j in range(len(second_of_trigrams)):
                    if second_of_trigrams[j] == second_word:
                        third_word.append(third_of_trigrams[j])

        # Choose the third word randomly
        next_word = random.choice(third_word)
        my_trigrams.append(next_word)

        if next_word == end_token:
            break

    # The sentence created returned with spaces seperating the words
    return (' '.join(my_trigrams))

max_words = 20

print('>>> Creating sentences using bigrams...')
for i in range(10):
    print(f'Sentence [{i+1}]: {bigram_sentence(max_words)}')

print('\n>>> Creating sentences using trigrams (this may take a while)...')
for i in range(10):
    print(f'Sentence [{i+1}]: {trigram_sentence(max_words)}')
