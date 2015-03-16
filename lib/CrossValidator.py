from __future__ import division

from collections import defaultdict
from collections import Counter

class CrossValidator:
    """This class handles the classification workflow by stor-
    ing the input records, the training and test datasets, and
    the classifier instance.
    
    Input records are quadruples (tag, sent, tokens, features)
    with the following elements:
    
        tag			the sentiment label (string)
        
        sent		an original constituent sentence  from a 
        			tweet
        
        tokens		a tokenized copy of an original sentence
        
        features	the features extracted for an original
        			sentece.
    
    If initialized with a value higher than 0 for the parame-
    ter 'folds', an instance of the CrossValidator class will 
    automatically split the input records into a training set 
    and a test set.
    
    If initialized with a value 0 for the parameter 'folds',
    all the elements in the parameter 'records' will be stored
    as training data in the attribute .train and the class
    will expect test data to be explicitly added at a later 
    time by using the method .set_test()
    
    In any cross-validation run (1 for the case in which folds=
    0), training and test data tuples (i.e., folds) will be 
    created and stored in the self.folds attribute to be used
    as input when the method .run() is used.
    
    The class additionally provides support for results evalua-
    tion through the .evaluation() method, and for writing the
    error analysis to disk (using the .err_out() method).
    """
    def __init__(self, records, classifier, folds=0):
        self.folds = []
        self.runs = []
        self.records = records
        self.classifier = classifier
        self.train = []
        self.test = []
        if folds:
            self.__split_test_train_sets(records, folds)
        else:
            self.records += records
            self.train = range(len(records))
    
    def set_test(self, test_set):
        size = len(self.records)
        self.records += test_set
        self.test = range(size, size + len(test_set))
    
    def __split_test_train_sets(self, records, folds):
        test_set_size = int(len(records) / folds)
        test_set, train_set = [], []
        for i, example in enumerate(records):
            test_set.append(i)
            if len(test_set) == test_set_size:
                train_set = self.__get_every_other_example(records, test_set)
                fold = (train_set, test_set)
                self.folds.append(fold)
                test_set = []

    def __get_every_other_example(self, records, test_set):
        return [i for i in range(len(records)) if i not in test_set]
    
    def run(self):
        if not self.folds:
            self.folds = [(self.train, self.test)]
        for i, fold in enumerate(self.folds):
            train, test, ground_truth = self.__prepare_classifier_inputs(fold)
            print 'fold-%d: training...' % (i + 1)
            self.classifier.train(train)
            print 'fold-%d: classifying...' % (i + 1)
            classification = self.classifier.classify(test)
            print 'fold-%d: evaluation...' % (i + 1)
            results = self.__evaluate(classification, ground_truth, test)
            self.runs.append(results)
    
    def __prepare_classifier_inputs(self, fold):
        train_input, test_input, ground_truth = [], [], []
        train, test = fold
        for i in train:
            tag, sent, tokens, features = self.records[i]
            train_input.append((features, tag))
        for i in test:
            tag, sent, tokens, features = self.records[i]
            test_input.append(features)
            ground_truth.append(tag)
        return train_input, test_input, ground_truth
    
    def __evaluate(self, classification, ground_truth, test):
        total, tp, tn, fp, fn = 0, 0, 0, 0, 0
        guesses = zip(classification, ground_truth)
        error_analysis = defaultdict(list)
        for i, guess in enumerate(guesses):
            hypothesis, expectation = guess
            total += 1
            if expectation == 'negative':
                if hypothesis == 'negative':
                    tn += 1
                elif hypothesis == 'positive':
                    fp += 1
                    error_analysis = \
                        self.__track_errors(error_analysis, hypothesis, 
                                            expectation, test[i])
            elif expectation == 'positive':
                if hypothesis == 'negative':
                    fn += 1
                    error_analysis = \
                        self.__track_errors(error_analysis, hypothesis, 
                                            expectation, test[i])
                elif hypothesis == 'positive':
                    tp += 1
        return self.__create_results_report(total, tp, tn, fp, fn, error_analysis)
    
    def __track_errors(self, error_analysis, hypothesis, expectation, featureset):
        type = (hypothesis, expectation)
        for feature, value in featureset.items():
            error_analysis[type].append(feature)
        return error_analysis

    def __create_results_report(self, total, tp, tn, fp, fn, error_analysis):
        precision, recall, accuracy, fmeasure = 0.0, 0.0, 0.0, 0.0
        if tp:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / total
            fmeasure = 2 * ((precision * recall) / (precision + recall))
        results = {
            'confusion_matrix': {
                'total': total,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'f-measure': fmeasure
            },
            'error_analysis': {key: Counter(values).most_common()
                               for key, values in error_analysis.items()}
        }
        return results

    def evaluation(self):
        avg = sum([run['confusion_matrix']['precision']
                   for run in self.runs]) / len(self.runs)
        print '--- START OF EVALUATION ---'
        for i, run in enumerate(self.runs):
            print '\t----- RUN %d -----' % (i + 1)
            #for key in 'total true_positives true_negatives false_positives false_negatives precision recall accuracy f-measure'.split():
            for key in 'precision accuracy recall f-measure'.split():
                print '\t%s:' % key.upper(), run['confusion_matrix'][key]
            print '\t----- END OF RUN %d -----\n' % (i + 1)
        print 'AVERAGE PRECISION:', avg
        print '\n--- END OF EVALUATION ---'
    
    def err_out(self, outf):
        all_errors = Counter()
        for run in self.runs:
            for case in run['error_analysis'].keys():
                for feature, frequency in run['error_analysis'][case]:
                    error = (case[0], case[1], feature)
                    all_errors[error] += frequency
        table = all_errors.most_common()
        self.__write(table, outf)
    
    def __write(self, table, outf):
        lines = []
        for triple, freq in table:
            hypothesis, expected, feature = triple
            try:
                line = 'EXPECTED=%s\tOBSERVED=%s\tFREQUENCY=%d\tFEATURE=%s' % (
                    expected, hypothesis, freq, feature.encode('utf-8')
                )
            except Exception:
                pass
            lines.append(line)
        txt = '\n'.join(lines)
        wrt = open(outf, 'w')
        wrt.write(txt)
        wrt.close()

