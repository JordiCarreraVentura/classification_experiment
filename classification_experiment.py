import sys

from lib.Dataset import Dataset
from lib.CrossValidator import CrossValidator
from lib.Classifier import Classifier


def argparse(args):
    argdict = dict([])
    argdict['--exceptions'] = None
    argdict['--prune'] = 0
    while args:
        #  --prune int
        #  --train path/to/file
        #  --test path/to/file
        #  --exceptions path/to/file
        key, value = args[0], args[1]
        if key == '--prune':
            argdict[key] = int(value)
        else:
            argdict[key] = value
        args = args[2:]
    argdict['mode'] = 'cross_validate'
    if '--test' in argdict.keys():
        argdict['mode'] = 'dev'
    return argdict

def main(args):
    
    args = argparse(args[1:])
    
    #  run with one dataset (and cross-validation over that dataset):
    if args['mode'] == 'cross_validate':
        
        #  prepare dataset:
        dataset = Dataset(args['--train'], args['--exceptions'])
        dataset.prune_features(args['--prune'])

        #  run classification and evaluate:
        CF = Classifier(algorithm='multinomial_naive_bayes_pipeline')
        CV = CrossValidator(dataset.records(), CF, folds=10)
    
    #    run with two datasets (train and test):
    elif args['mode'] == 'dev':

        #  prepare train and test datasets:
        train_set = Dataset(args['--train'], args['--exceptions'])
        test_set = Dataset(args['--test'], args['--exceptions'])
        train_set.prune_features(args['--prune'])
        test_set.prune_features(args['--prune'])

        #  run classification and evaluate:
        CF = Classifier(algorithm='multinomial_naive_bayes_pipeline')
        CV = CrossValidator(train_set.records(), CF)
        CV.set_test(test_set.records())
    
    CV.run()
    CV.evaluation()
    CV.err_out('error_out.tsv')


if __name__ == '__main__':
    main(sys.argv)
