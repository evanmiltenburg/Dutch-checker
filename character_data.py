"""
This is where the magic happens! This file has all definitions for data prep which
char_lstm.py and use_model.py can use. This file is not supposed to be run by itself,
but you can import it to make use of everything defined below.
"""

import random
import string
from collections import Counter
from math import ceil
import numpy as np

random.seed(123456789)
TABLE = table = str.maketrans("","",string.punctuation)
DUTCH = 'nl_wiki_plain.txt'
ENGLISH = 'en_wiki_plain.txt'
all_characters = string.ascii_lowercase + string.digits
index = {char: num for num, char in enumerate(list(all_characters)+[' ','UNK'], start=1)}

def normalize(line):
    "Normalize the line so that we only focus on the words and spaces."
    return line.strip().lower().translate(TABLE)

def char2index(char):
    "Map characters to their respective index."
    try:
        return index[char]
    except KeyError:
        return index['UNK']

def vectorizing_generator(filename, maxlen=100):
    "Load the data, converting it to vector format."
    with open(filename) as f:
        for line in f:
            yield [char2index(char) for char in normalize(line)][:maxlen]

def filter_unknowns(data, max_unk=5):
    "Remove vectors that have more than 5 unknown characters."
    unk = index['UNK']
    for vector in data:
        if vector.count(unk) > max_unk:
            continue
        else:
            yield vector

def load_filtered(filename, maxlen=100, max_unk=5):
    dutch_gen = vectorizing_generator(filename)
    return list(filter_unknowns(dutch_gen, max_unk))


def shuffled_sample(data, num=10000):
    "Generate a number of shuffled samples."
    ss = []
    for i, vector in enumerate(data, start=1):
        sample = random.sample(vector, len(vector))
        ss.append(sample)
        if i == num:
            break
    random.shuffle(ss)
    return ss

def train_index(arrays, test_split):
    "Compute the index up to where the training data goes."
    return int(len(arrays) * (1 - test_split))

def load_data(maxlen=100, max_unk=5, num_english=20000, num_shuffled=20000, test_split=0.2):
    "Load all the data."
    dutch_arrays = load_filtered(DUTCH, maxlen, max_unk)
    english_arrays = list(vectorizing_generator(ENGLISH, maxlen))[:num_english]
    # Randomize:
    random.shuffle(dutch_arrays)
    random.shuffle(english_arrays)
    
    # And get the shuffled arrays:
    shuffled_arrays = shuffled_sample(dutch_arrays, num=num_shuffled)
    
    # Compute indices:
    dutch_train_index = train_index(dutch_arrays, test_split)
    english_train_index = train_index(english_arrays, test_split)
    shuffled_train_index = train_index(shuffled_arrays, test_split)
    
    # Prepare arrays:
    train_arrays = []
    test_arrays = []
    train_targets = [1] * dutch_train_index + [0] * (english_train_index + shuffled_train_index)
    
    dutch_test_values = [1] * (len(dutch_arrays) - dutch_train_index)
    english_test_values = [0] * (len(english_arrays) - english_train_index)
    shuffled_test_values = [0] * (len(shuffled_arrays) - shuffled_train_index)
    
    test_targets = dutch_test_values + english_test_values + shuffled_test_values
    
    # Extend train:
    train_arrays.extend(dutch_arrays[:dutch_train_index])
    train_arrays.extend(english_arrays[:english_train_index])
    train_arrays.extend(shuffled_arrays[:shuffled_train_index])
    
    # Extend test:
    test_arrays.extend(dutch_arrays[dutch_train_index:])
    test_arrays.extend(english_arrays[english_train_index:])
    test_arrays.extend(shuffled_arrays[shuffled_train_index:])
    
    assert len(train_arrays) == len(train_targets)
    assert len(test_arrays) == len(test_targets)
    
    train_targets = np.array(train_targets)
    test_targets = np.array(test_targets)
    # Return the data
    return (train_arrays, train_targets), (test_arrays, test_targets)
