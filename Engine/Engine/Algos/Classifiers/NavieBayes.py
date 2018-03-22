import numpy as np
from sklearn.naive_bayes import MultinomialNB

from Engine.Algos.Classifiers import Util


class NaviesBayes:

    def __init__(self):
        self.train_data = ''
        self.model1 = MultinomialNB()
        self.dictionary = {"": ""}

    def train(self):
        print('Started training the engine')
        train_dir = 'D:/adi/ML/ling-spam/train-mails'
        dictionary = Util.make_dictionary(train_dir)
        self.dictionary = dictionary
        train_labels = np.zeros(702)
        train_labels[351:701] = 1
        train_matrix = Util.extract_features(train_dir, dictionary)
        self.model1.fit(train_matrix, train_labels)
        print('Training finished !!!')

    def predict(self, features):
        result = self.model1.predict(features)
        return result

    def get_dictionary(self):
        return self.dictionary


#nb = NaviesBayes()
#nb.train()
#predict_data = Util.extract_features_predict("")
#print(nb.predict(predict_data))
