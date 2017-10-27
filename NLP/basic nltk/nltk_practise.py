from nltk_some_basic import VoteClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist, NaiveBayesClassifier, pos_tag
from nltk.classify import accuracy
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import random
import pickle
import sys

#read in pos and neg data
#save file in utf-8 then we can use encoding utf-8, otherwise use iso-8859-1
with open("positive.txt", "r", encoding="ISO-8859-1") as f:
	short_pos = f.read()
with open("negative.txt", "r", encoding="ISO-8859-1") as f:
	short_neg = f.read()

#create label for each sentance
documents = []
for r in short_pos.split('\n'):
#for r in sent_tokenize(short_pos):
	documents.append((r.lower(), 'pos'))
for r in short_neg.split('\n'):
#for r in sent_tokenize(short_neg):
	documents.append((r.lower(), 'neg'))

#print(documents)

#label each words
all_words = []
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)
for w in short_pos_words:
	all_words.append(w.lower())
for w in short_neg_words:
	all_words.append(w.lower())

stop_words = set(stopwords.words("english"))
all_words = list(filter(lambda w: w not in stop_words, all_words))
all_words = FreqDist(all_words)

word_features = list(all_words.keys())[:5000]
def find_features(doc, word_features):
    words = set(word_tokenize(doc))
    features = dict()
    for w in word_features:
        features[w] = (w in words)
    return features

features_sets = list((find_features(rev, word_features), cat) for rev, cat in documents)
print(features_sets[0])
sys.exit(0)

with open("features_sets.pkl", "wb") as f:
    pickle.dump(features_sets, f) 
random.shuffle(features_sets)

cut = int(0.8 * len(features_sets))
training_set = features_sets[:cut]
testing_set = features_sets[cut:]

classifier = NaiveBayesClassifier.train(training_set)
print("accuracy: ", accuracy(classifier, testing_set))
#classifier.show_most_informative_features(3)

# save trained model as piclke obj
with open("NaiveBayes1.pkl", "wb") as f:
    pickle.dump(classifier, f) 

#load training model
with open("NaiveBayes1.pkl", "rb") as f:
    cf = pickle.load(f)
print("new accuracy: ", accuracy(cf, testing_set))
#cf.show_most_informative_features(3)

#using classifying method from sklearn
MNB_cf = SklearnClassifier(MultinomialNB())
MNB_cf.train(training_set)
print("MultinomialNB accuracy: ", accuracy(MNB_cf, testing_set))

with open("MultinomialNB_trained_Model.pkl", "wb") as f:
    pickle.dump(MNB_cf, f) 

# GNB_cf = SklearnClassifier(GaussianNB())
# GNB_cf.train(training_set)
# print("GaussianNB accuracy: ", accuracy(GNB_cf, testing_set))

BNB_cf = SklearnClassifier(BernoulliNB())
BNB_cf.train(training_set)
print("BernoulliNB accuracy: ", accuracy(BNB_cf, testing_set))

with open("BernoulliNB_trained_Model.pkl", "wb") as f:
    pickle.dump(BNB_cf, f) 

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC
LR_cf = SklearnClassifier(LogisticRegression())
LR_cf.train(training_set)
print("LogisticRegression accuracy: ", accuracy(LR_cf, testing_set))

with open("ogisticRegression_trained_Model.pkl", "wb") as f:
    pickle.dump(LR_cf, f) 

SGDClassifier_cf = SklearnClassifier(SGDClassifier())
SGDClassifier_cf.train(training_set)
print("SGDClassifier accuracy: ", accuracy(SGDClassifier_cf, testing_set))

with open("SGDClassifier_trained_Model.pkl", "wb") as f:
    pickle.dump(SGDClassifier_cf, f) 

SVC_cf = SklearnClassifier(SVC())
SVC_cf.train(training_set)
print("SVC accuracy: ", accuracy(SVC_cf, testing_set))

LinearSVC_cf = SklearnClassifier(LinearSVC())
LinearSVC_cf.train(training_set)
print("LinearSVC accuracy: ", accuracy(LinearSVC_cf, testing_set))

with open("LinearSVC_trained_Model.pkl", "wb") as f:
    pickle.dump(LinearSVC_cf, f) 

NuSVC_cf = SklearnClassifier(NuSVC())
NuSVC_cf.train(training_set)
print("NuSVC accuracy: ", accuracy(NuSVC_cf, testing_set))

with open("NuSVC_trained_Model.pkl", "wb") as f:
    pickle.dump(NuSVC_cf, f) 

vote_cf = VoteClassifier(classifier, MNB_cf, BNB_cf, LR_cf, SGDClassifier_cf, SVC_cf, LinearSVC_cf, LinearSVC_cf, NuSVC_cf)
print("voted classifier accuracy: ", accuracy(vote_cf, testing_set))

#some example one single records
print("Classification: ", vote_cf.classify(testing_set[1][0]), "confidence: ", vote_cf.confidence(testing_set[1][0]))
print("Classification: ", vote_cf.classify(testing_set[2][0]), "confidence: ", vote_cf.confidence(testing_set[2][0]))
print("Classification: ", vote_cf.classify(testing_set[3][0]), "confidence: ", vote_cf.confidence(testing_set[3][0]))
print("Classification: ", vote_cf.classify(testing_set[5][0]), "confidence: ", vote_cf.confidence(testing_set[5][0]))
print("Classification: ", vote_cf.classify(testing_set[8][0]), "confidence: ", vote_cf.confidence(testing_set[8][0]))


#######################################################################################################################
#similar as above but only using words that are verbs or adjectives 
#######################################################################################################################
with open("positive.txt", "r", encoding="ISO-8859-1") as f:
	short_pos = f.read()
with open("negative.txt", "r", encoding="ISO-8859-1") as f:
	short_neg = f.read()
#create label for each sentance
documents = []
all_words = []
# types can be J-adj; V-verb
allowed_word_types = ["J", "V"]
for r in short_pos.split('\n'):
#for r in sent_tokenize(short_pos):
	documents.append((r.lower(), 'pos'))
	words = word_tokenize(r.lower())
	pos_tags = pos_tag(words)
	for tag in pos_tags:
		if tag[1][0] in allowed_word_types:
			all_words.append(tag[0].lower()) 
for r in short_neg.split('\n'):
#for r in sent_tokenize(short_neg):
	documents.append((r.lower(), 'neg'))
	words = word_tokenize(r.lower())
	pos_tags = pos_tag(words)
	for tag in pos_tags:
		if tag[1][0] in allowed_word_types:
			all_words.append(tag[0].lower()) 

stop_words = set(stopwords.words("english"))
all_words = list(filter(lambda w: w not in stop_words, all_words))
all_words = FreqDist(all_words)

with open("documents.pkl", "wb") as f:
    pickle.dump(documents, f)

word_features = list(all_words.keys())[:5000]

with open("word_features.pkl", "wb") as f:
    pickle.dump(word_features, f)

def find_features(doc, word_features):
    words = set(word_tokenize(doc))
    features = dict()
    for w in word_features:
        features[w] = (w in words)

    return features

features_sets = list((find_features(rev, word_features), cat) for rev, cat in documents)
random.shuffle(features_sets)

cut = int(0.8 * len(features_sets))
training_set = features_sets[:cut]
testing_set = features_sets[cut:]

classifier = NaiveBayesClassifier.train(training_set)
print("accuracy: ", accuracy(classifier, testing_set))
#classifier.show_most_informative_features(3)

# save trained model as piclke obj
with open("NaiveBayes.pkl", "wb") as f:
    pickle.dump(classifier, f) 

MNB_cf = SklearnClassifier(MultinomialNB())
MNB_cf.train(training_set)
print("MultinomialNB accuracy: ", accuracy(MNB_cf, testing_set))

with open("MultinomialNB_trained_Model.pkl", "wb") as f:
    pickle.dump(MNB_cf, f) 

# GNB_cf = SklearnClassifier(GaussianNB())
# GNB_cf.train(training_set)
# print("GaussianNB accuracy: ", accuracy(GNB_cf, testing_set))

BNB_cf = SklearnClassifier(BernoulliNB())
BNB_cf.train(training_set)
print("BernoulliNB accuracy: ", accuracy(BNB_cf, testing_set))

with open("BernoulliNB_trained_Model.pkl", "wb") as f:
    pickle.dump(BNB_cf, f) 

# LogisticRegression, SGDClassifier
# SVC, LinearSVC, NuSVC
LR_cf = SklearnClassifier(LogisticRegression())
LR_cf.train(training_set)
print("LogisticRegression accuracy: ", accuracy(LR_cf, testing_set))

with open("ogisticRegression_trained_Model.pkl", "wb") as f:
    pickle.dump(LR_cf, f) 

SGDClassifier_cf = SklearnClassifier(SGDClassifier())
SGDClassifier_cf.train(training_set)
print("SGDClassifier accuracy: ", accuracy(SGDClassifier_cf, testing_set))

with open("SGDClassifier_trained_Model.pkl", "wb") as f:
    pickle.dump(SGDClassifier_cf, f) 

SVC_cf = SklearnClassifier(SVC())
SVC_cf.train(training_set)
print("SVC accuracy: ", accuracy(SVC_cf, testing_set))

LinearSVC_cf = SklearnClassifier(LinearSVC())
LinearSVC_cf.train(training_set)
print("LinearSVC accuracy: ", accuracy(LinearSVC_cf, testing_set))

with open("LinearSVC_trained_Model.pkl", "wb") as f:
    pickle.dump(LinearSVC_cf, f) 

NuSVC_cf = SklearnClassifier(NuSVC())
NuSVC_cf.train(training_set)
print("NuSVC accuracy: ", accuracy(NuSVC_cf, testing_set))

with open("NuSVC_trained_Model.pkl", "wb") as f:
    pickle.dump(NuSVC_cf, f) 