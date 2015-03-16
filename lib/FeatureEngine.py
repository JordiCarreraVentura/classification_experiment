import re

from collections import defaultdict

class FeatureEngine:
    """When an instance of the FeatureEngine class is called
    using the .__call__() method, it returns the featureset
    for the tokens provided as argument (list of strings).
    
    The output is a Python dictionary with features as keys
    (strings) and a boolean as the value (always True).
    
    While extracting the features, it also looks up the list
    of exceptions and excludes any features in that list
    from the featuresets.
    
    Features used can be broswed in self.features. Each of
    the feature tuples there contains four elements, in 
    the order below:
        
        name		the name of the feature (string)
        
        state		whether a feature is active or not (Boolean)
        
        function	the method in the FeatureEngine to be used
        			to extract the features (method)
        			
        tuple		arguments for the function (tuple)
    """
    def __init__(self, exceptions):
        self.features = [
            #  name, state, function, args
            ('unigrams', True, self.__ngrams, (1, False)),
            ('bigrams', True, self.__ngrams, (2, False)),
            ('trigrams', False, self.__ngrams, (3, False)),
            ('skip-trigrams', True, self.__ngrams, (3, True)),
            ('skip-fourgrams', True, self.__ngrams, (4, True)),
            ('skip-fivegrams', False, self.__ngrams, (5, True))
        ]
        self.exceptions = self.__load_exceptions(exceptions)
        self.frequencies = defaultdict(int)
    
    def __load_exceptions(self, exceptions):
        exception_list = []
        if exceptions:
            for l in open(exceptions, 'r'):
                w = l.decode('utf-8').strip()
                exception_list.append(w)
        return exception_list
    
    def __call__(self, tokens):
        features = dict([])
        for name, state, function, args in self.features:
            if state:
                features[name] = function(tokens, args)
        return self.__features2featureset(features)
    
    def __features2featureset(self, features):
        featureset = dict([])
        for name, annotation in features.items():
            for feature, value in annotation:
                featureset[feature] = value
                self.frequencies[feature] += 1
        return featureset
    
    def __ngrams(self, tokens, args):
        order, skip = args
        if order > len(tokens):
            return []
        elif order == len(tokens):
            gram = ' '.join(tokens)
            if not self.__is_exception(gram):
                return [(gram, True)]
        grams = []
        for i in range(len(tokens) - (order - 1)):
            gram = tokens[i:i + order]
            g1 = gram[0]
            g2 = gram[-1]
            #if gram == ['!']:
            #    print sorted(self.exceptions)
            #    print tokens, gram, self.__is_exception(' '.join(gram))
            if self.__is_exception(' '.join(gram)):
                continue
            if not (self.__is_word(g1) \
                    and self.__is_word(g2) \
                    and (not self.__is_exception(g1) or \
                         not self.__is_exception(g2))
            ):
                continue
            if skip:
                gram = [g1] + ['*'] + [g2]
            grams.append((' '.join(gram), True))
        return grams
    
    def __is_word(self, gram):
        letters = re.compile('[a-z]{2,}')
        if letters.match(gram):
            return True
        else:
            return False
    
    def __is_exception(self, word):
        if word in self.exceptions:
            return True
        else:
            return False
        
