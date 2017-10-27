from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk import FreqDist, NaiveBayesClassifier, pos_tag
# from nltk.classify import accuracy
from nltk.classify import ClassifierI
# from nltk.classify.scikitlearn import SklearnClassifier
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.svm import SVC, LinearSVC, NuSVC
# import random
import pickle
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self.classifiers = classifiers

    def classify(self, features):
        votes = []
        for classifier in self.classifiers:
            v = classifier.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for classifier in self.classifiers:
            v = classifier.classify(features)
            votes.append(v)
        choice_vote = votes.count(mode(votes))
        return choice_vote / len(votes)

with open("word_features.pkl", "rb") as f:
	word_features = pickle.load(f)

def find_features(doc):
    words = set(word_tokenize(doc))
    features = dict()
    for w in word_features:
        features[w] = (w in words)
    return features


with open("NaiveBayes.pkl", "rb") as f:
    NaiveBayes_model = pickle.load(f)

with open("MultinomialNB_trained_Model.pkl", "rb") as f:
    MultinomialNB_model = pickle.load(f)

with open("BernoulliNB_trained_Model.pkl", "rb") as f:
    BernoulliNB_model = pickle.load(f)

with open("LogisticRegression_trained_Model.pkl", "rb") as f:
    LogisticRegression_model = pickle.load(f)

with open("SGDClassifier_trained_Model.pkl", "rb") as f:
    SGDClassifier_model = pickle.load(f)

with open("LinearSVC_trained_Model.pkl", "rb") as f:
	LinearSVC_model = pickle.load(f) 

with open("NuSVC_trained_Model.pkl", "rb") as f:
	NuSVC_model = pickle.load(f) 

voted_model = VoteClassifier(NaiveBayes_model,
							MultinomialNB_model,
							BernoulliNB_model,
							LogisticRegression_model,
							SGDClassifier_model,
							LinearSVC_model,
							NuSVC_model)

def sentiment(text):
	features = find_features(text)
	return voted_model.classify(features), voted_model.confidence(features)

