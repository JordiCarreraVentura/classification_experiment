import copy
import csv
import nltk
import random
#import pattern
from nltk import sent_tokenize
from nltk import wordpunct_tokenize

from collections import defaultdict
#from pattern.en import parsetree

from FeatureEngine import FeatureEngine


def csv_fixer(row):
    if len(row) > 2:
        return [row[0], '\''.join(row[1:])]
    else:
        return row


class Dataset:
    """This class 1) reads the dataset from disk, 2) extracts
    features from each input example (using an instance of the 
    FeatureEngine class), 3) provides methods to query for the
    examples and/or their labels, and 4) stores the records as
    quadruples (tag, sent, tokens, features) with the follow-
    ing elements:
    
        tag			the sentiment label (string)
        
        sent		an original constituent sentence  from a 
        			tweet
        
        tokens		a tokenized copy of an original sentence
        
        features	the features extracted for an original
        			sentece.
    
    The dataset must be given as the parameter 'path' at ini-
    tialization. Optionally, the Dataset class can also take
    a second parameter, 'exceptions', pointing also to a sys-
    tem file and containing a list of features to be avoided
    for classification (e.g., stopwords).
    
    In addition, this class also provides a method to perform
    frequency-based pruning of features over a certain thres-
    hold, .prune_features(). The method takes an integer as
    input and operates over the actual list of dataset exampl-
    es.
    """
    def __init__(self, path, exceptions=None):
        self._records = []
        self.feature_engine = FeatureEngine(exceptions)
        self.origin = path
        self.__parse_load()
    
    def __parse_load(self):
        rd = csv.reader(open(self.origin, 'r'))
        for row in rd:
            row = csv_fixer(row)
            tag, example = tuple([field.decode('utf-8') for field in row])
            parses = self.__parse(example)
            for sent, tokens, features in parses:
                record = (tag, sent, tokens, features)
            self._records.append(record)
        random.shuffle(self._records)
    
    def __parse(self, example):
        parsed = []
        for sent in sent_tokenize(example):
            tokens = wordpunct_tokenize(sent.lower())
            features = self.feature_engine(tokens)
            parsed.append((sent, tokens, features))
        return parsed
    
    def records(self, i=None):
        if i == None:
            return copy.deepcopy(self._records)
        else:
            return copy.deepcopy(self._records[i])
    
    def feature_sets(self, i=None):
        if i == None:
            return copy.deepcopy([record[2] for record in self._records])
        else:
            return copy.deepcopy(self._records[i][2])
    
    def tags(self, i=None):
        if i == None:
            return copy.deepcopy([record[0] for record in self._records])
        else:
            return copy.deepcopy(self._records[i][0])
    
    def examples(self, i=None):
        if i == None:
            return copy.deepcopy([record[1] for record in self._records])
        else:
            return copy.deepcopy(self._records[i][1])
    
    def prune_features(self, min_freq):
        new_records = []
        for tag, sent, tokens, features in self.records():
            pruned = tag, sent, tokens, self.__prune(features, min_freq)
            new_records.append(pruned)
        self._records = new_records
    
    def __prune(self, features, min_freq):
        pruned = dict([])
        for feature, value in features.items():
            if self.feature_engine.frequencies[feature] >= min_freq:
                pruned[feature] = value
        return pruned
            
    
    
