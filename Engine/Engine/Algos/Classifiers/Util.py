import os
from collections import Counter
import numpy as np


def make_dictionary(train_dir):

    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2:  # Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words)
    # Paste code for non-word removal here(code snippet is given below)
    list_to_remove = dictionary.keys()
    list_remove = []
    for item in list_to_remove:
        if item.isalpha() is False:
            # del dictionary[item]
            list_remove.append(item)
        elif len(item) == 1:
            # del dictionary[item]
            list_remove.append(item)
    # print(list_remove)
    for item in list_remove:
        del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary


def extract_features(mail_dir, dictionary):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 3000))
    doc_i_d = 0
    for fil in files:
        with open(fil) as fi:
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        word_i_d = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                word_i_d = i
                                features_matrix[doc_i_d, word_i_d] = words.count(word)
            doc_i_d = doc_i_d + 1
    return features_matrix


def extract_features_predict(data, dictionary):
    features_matrix = np.zeros((1, 3000))
    words = data.split()
    for word in words:
        word_i_d = 0
        for i, d in enumerate(dictionary):
            if d[0] == word:
                word_i_d = i
                features_matrix[0, word_i_d] = words.count(word)
    return features_matrix
