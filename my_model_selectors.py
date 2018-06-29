import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    """
    lower the BIC, better, why??
    Answer:
    L : Likelihood of the model to fit
    p : parameters = n^2 + 2*n*features - 1
    N : data-points = self.X
    The term −2 log L decreases with increasing model complexity (more parameters), whereas the penalties 2p or p log N increase with increasing complexity
    """


    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        model_bic = float("inf")
        final_model = None

        # try and except for sequences that generate ValueError: rows of transmat_ must sum to 1.0 (got [1. 1. 1. 1. 0. 1.])

        for n in range(self.min_n_components, self.max_n_components + 1):   # BIC score for n between self.min_n_components and self.max_n_components
            base = self.base_model(n)
            try:
                N = self.X
                # BIC = -2 * logL + p * logN
                logL = base.score(N, self.lengths)
                # forum --> p = n^2 + 2*features*n - 1;
                p = n ** 2 + 2 * n * base.n_features - 1
                current_bic = -2 * logL + p * math.log(len(N))

                if current_bic < model_bic:
                    model_bic = current_bic
                    final_model = base
            except Exception:
                pass

        return final_model

        # raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    # DFC approximation where values are replaced by BIC approximation; hence would be similar to BIC calculation
    # Model receives all the words after being trained from the training sequence which trains the model based on feature parameter

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        model_dic = float("-inf")   # the difference needs to be greater so that distinct words can be expressed without increased error
        final_model = None

        # try and except for sequences that generate ValueError: rows of transmat_ must sum to 1.0 (got [1. 1. 1. 1. 0. 1.])

        for n in range(self.min_n_components, self.max_n_components + 1):
            base = self.base_model(n)
            try:
                N = self.X
                # DIC = log(P(X(i)) - 1 / (M - 1) SUM(log(P(X(all but i))
                logP_X = base.score(N, self.lengths)    # like BIC
                anti_logP_X = 0
                word_count = 0

                for word in self.words:
                    # for i
                    if word is self.this_word:
                        pass
                    else:
                        # for all but i: will count (len(word) - 1) times
                        word_X, word_lengths = self.hwords[word]    # self.X, self.lengths = all_word_Xlengths[this_word]
                        anti_logP_X += base.score(word_X, word_lengths)
                        word_count += 1

                anti_logP_X = anti_logP_X / word_count
                current_dic = logP_X - anti_logP_X

                if current_dic > model_dic:
                    model_dic = current_dic
                    final_model = base

            except Exception:
                pass

        return final_model
        # raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        model_cv = float("-inf")    # like DIC; in this case maximum likelihood based on best score
        final_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            base = self.base_model(n)
            logL = 0
            count = 0
            try:
                N = self.X
                n_split = min(3, len(self.sequences))
                if n_split > 1: # for ABC and more, train on AB, test on C and its combination
                    model_split = KFold(n_splits=n_split)
                    for train, test in model_split.split(self.sequences):
                        train_X, train_lengths = combine_sequences(train, self.sequences)
                        test_X, test_lengths = combine_sequences(test, self.sequences)
                        current_model = GaussianHMM(n_components=n, n_iter=1000).fit(train_X, train_lengths)    # Gaussian to train for both, whether sequence greater than 1 or not
                        log = current_model.score(test_X, test_lengths)
                        logL += log
                        count += 1
                else:   # len(self.sequence) == 1 to be handled separately, only one set to train and test, i.e. A to train and test
                    # training and testing on same data
                    current_model = GaussianHMM(n_components=n, n_iter=1000).fit(N, self.lengths)
                    log = current_model.score(N, self.lengths)
                    logL += log
                    count += 1

                logL_avg = logL / count
                if logL_avg > model_cv:
                    model_cv = logL_avg
                    final_model = base

            except Exception:
                pass

        return final_model

        # raise NotImplementedError
