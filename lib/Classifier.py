# -*- encoding: utf-8 -*-

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from nltk.classify.scikitlearn import SklearnClassifier

class Classifier:
    def __init__(self, algorithm='multinomial_naive_bayes'):
        if algorithm == 'multinomial_naive_bayes':
            self._classifier = SklearnClassifier(MultinomialNB())
        elif algorithm == 'support_vector_machine':
            self._classifier = SklearnClassifier(LinearSVC())
        elif algorithm == 'multinomial_naive_bayes_pipeline':
            pipeline = Pipeline([('tfidf', TfidfTransformer()),
                                 ('chi2', SelectKBest(chi2, k=500)),
                                 ('nb', MultinomialNB())])
            self._classifier = SklearnClassifier(pipeline)
        else:
            exit('FATAL: unsupported algorithm: \"%s\"' % algorithm)
    
    def train(self, train_set):
        self._classifier.train(train_set)
    
    def classify(self, test_set):
        return self._classifier.classify_many(test_set)
