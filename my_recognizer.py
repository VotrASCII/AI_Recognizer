import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []  # has logL value for a word in dictionary format
    guesses = []    # best match

    # TODO implement the recognizer
    for test in range(test_set.num_items):
        value = {}   # dictionary to be passed to probabilities
        word_guess = None
        final_score = float("-inf")
        test_X, test_lengths = test_set.get_item_Xlengths(test)  # returns list, list
        for word in models:
            model = models[word]
            # in case HMM isn't able to train all model
            try:
                score = model.score(test_X, test_lengths)
                value[word] = score
                if score > final_score:
                    final_score = score
                    word_guess = word
            except:
                value[word] = float("-inf")
        probabilities.append(value)
        guesses.append(word_guess)
    return probabilities, guesses
    # raise NotImplementedError
