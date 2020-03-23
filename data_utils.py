import os, re, pickle
from collections import namedtuple
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical

DATA_DIR = 'dialog-bAbI-tasks'
CACHED_DIR = 'cached-data'
CANDIDATES_PATH = os.path.join(DATA_DIR, 'dialog-babi-candidates.txt')
CACHED_DATA_PATH = os.path.join(CACHED_DIR, 'task{}.file')
TOKENS_TO_EXCLUDE = {'a', 'an', 'the', '.', '?', '!'}  # some simple stop-words and punctuation

"""
Silence, turn (user/bot), and time-notion (i) encoding strings.
It's assumed that these special characters won't appear in the dialogs.
"""
SILENCE_ENCODING = '<silence>'
USER_UTTERANCE_ENCODING = '$u'
BOT_RESPONSE_ENCODING = '$r'
TIME_ENCODING = '#{}'  # '#1' --> '#2' --> '#3' --> etc.

"""
Bags of words datatype defined as uint8, instead of float32 or float64, in order to save space.
See space optimization details described in the main.py under the 'Implementation details' section.
"""
BAG_OF_WORDS_DTYPE = 'uint8'    # assumption: the same word won't appear more than 255 times in a single sentence
DEFAULT_KERAS_DTYPE= 'float32'  # consider trying also float16 if half-precision is supported by your gpu


def get_everything_you_need(task_id, use_cache, isOOV=False):
    """
    Prepare the data for the n-th task.
    Returns a named tuple containing:
      - the vocabulary size (the number of unique words, including special characters, inside every dataset of every task)
      - the maximum context size (the memory size)
      - the candidates_matrix (where the candidates are encoded as one-hot-vectors and stacked side by side)
      - the training, validation and testing data
    """

    assert 1 <= task_id <= 5

    if use_cache and os.path.exists(CACHED_DATA_PATH.format(task_id)):
        try:
            with open(CACHED_DATA_PATH.format(task_id), "rb") as cached_file:
                print("Retrieving locally cached data for task {}...".format(task_id))
                useful_data = pickle.load(cached_file)
        except:
            print("Something went wrong, clearing cache and vectorizing data from babi file...")
            os.remove(CACHED_DATA_PATH.format(task_id))
            get_everything_you_need(task_id, use_cache=False, isOOV=False)

    else:  # parse, tokenize and finally vectorize all the data for the selected task
        candidates, tokenized_candidates, cand2idx = _load_candidates()

        files = os.listdir(DATA_DIR)
        files = [os.path.join(DATA_DIR, f) for f in files]
        s = 'dialog-babi-task{}-'.format(task_id)
        train_file = [f for f in files if s in f and 'trn' in f][0]
        if isOOV:
            test_file = [f for f in files if s in f and 'tst-OOV' in f][0]
        else:
            test_file = [f for f in files if s in f and 'tst.' in f][0]
        val_file = [f for f in files if s in f and 'dev' in f][0]
        tokenized_train_data = _load_dialogs(train_file, cand2idx)
        tokenized_val_data = _load_dialogs(val_file, cand2idx)
        tokenized_test_data = _load_dialogs(test_file, cand2idx)

        all_the_data_tokenized = tokenized_train_data + tokenized_val_data + tokenized_test_data
        vocabulary, word2idx = _eval_vocabulary(all_the_data_tokenized, tokenized_candidates)
        max_context_size = _eval_max_context_size(all_the_data_tokenized)

        candidates_matrix = _vectorize_candidates(tokenized_candidates, word2idx)

        print("Vectorizing all the data...")
        useful_data = UsefulData(len(vocabulary),
                                 max_context_size,
                                 candidates_matrix,
                                 _vectorize_data(tokenized_train_data, word2idx, max_context_size),
                                 _vectorize_data(tokenized_val_data, word2idx, max_context_size),
                                 _vectorize_data(tokenized_test_data, word2idx, max_context_size))

        if use_cache:
            print("Saving the vectorized data to disk...")
            pickle.dump(useful_data,
                        open(CACHED_DATA_PATH.format(task_id), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

    return useful_data


"""
All the data needed by the main program to train the model, packed in a handy named tuple.
Consider using a Dataclass if you'll upgrade to Python 3.7+.
"""
UsefulData = namedtuple('UsefulData',
                        ['vocabulary_size',
                         'context_size',
                         'candidates_matrix',
                         'vectorized_training',    # context facts, last user utterance and candidate response (TRAINING)
                         'vectorized_validation',  # context facts, last user utterance and candidate response (VALIDATION)
                         'vectorized_test'])       # context facts, last user utterance and candidate response (TEST)


# ======================================================================================================================
# 'PRIVATE' FUNCTIONS
# ======================================================================================================================


def _tokenize(utterance):
    """
    Given a sentence as string, returns a list of its words, lowering every eventual upper-case letter.
    Everything inside TOKENS_TO_EXCLUDE will be excluded.
    Maybe redundant: if everything is stripped from the sentence, ['<silence>'] will be returned.
    >>> _tokenize('Bob dropped the apple.')
    ['bob', 'dropped', 'apple']
    """
    utterance = utterance.lower()
    tokenized_utterance = [x.strip() for x in re.split('(\W+)?', utterance)
                           if x.strip() and x.strip() not in TOKENS_TO_EXCLUDE]
    return tokenized_utterance if tokenized_utterance else [SILENCE_ENCODING]  # check if tokenized_utterance isn't an empty list


def _load_candidates(path=CANDIDATES_PATH):
    """
    Given the candidates file path, returns:
      - candidates: a list with all the candidates as strings;
      - tokenized_candidates: a list with all the possible utterances that the bot can say,
        where every utterance is encoded as a list of words;
      - cand2idx: a dictionary that given a candidate (as string) returns its index inside the candidates list;
        it's useful to have this dictionary because it's faster to do cand2idx[x] rather than candidates.index(x).
    Example:
    [
      'hello there general kenobi',
      'here is your reservation',
      'you are very welcome',
      ...
    ],
    [
      ['hello', 'there', 'general', 'kenobi'],
      ['here', 'is', 'your', 'reservation'],
      ['you', 'are', 'very', 'welcome'],
      ...
    ],
    {
      'hello there general kenobi': 0,
      'here is your reservation': 1,
      'you are very welcome': 2,
      ...
    }
    Details: "candidates" are all the possible responses that the dialog agent can express,
             including training, validation and test set of all the five bAbI dialog tasks.
    """
    candidates, tokenized_candidates, cand2idx = [], [], {}
    with open(path) as candidates_file:
        for i, line in enumerate(candidates_file):
            candidate = line.strip().split(' ', 1)[1]  # take out the '1' from '1 api_call italian bombay four cheap'
            candidates.append(candidate)
            tokenized_candidates.append(_tokenize(candidate))
            cand2idx[candidate] = i
    return candidates, tokenized_candidates, cand2idx


def _load_dialogs(task_path, cand2idx):
    """
    Returns a list of triplets where each one of them contains:
      - tokenized context: a list of the previous utterances of a dialog (tokenized as a list of words)
      - tokenized utterance: the user's last utterance (tokenized as a list of words)
      - the correct bot response (a) encoded as the corresponding index of all the candidates list.
        NB: this represents the desired Neural Network output given the other two as inputs;
            if you leave it this way (as an integer) you can train your network
            using 'sparse_categorical_crossentropy' as loss function;
            otherwise, if you convert it to its one-hot representation (like [0,0,0, ... 0,1,0, ... 0,0,0])
            you'll need to use 'categorical_crossentropy' as loss function.
    Example:
    [
      (
        [
          ['hi', '$u', '#1'],
          ['hello', 'what', 'can', 'i', 'help', 'you', 'with', 'today', '$r', '#1']
        ],
        ['can', 'you', 'book', 'a', 'table'],
        103
      ),
      ...
    ]
    """
    with open(task_path) as file:
        data = []
        context = []
        u = None  # an utterance said by the user
        r = None  # the bot's response
        for line in file.readlines():
            line = line.strip()
            if line:
                nid, line = line.split(' ', 1)
                nid = int(nid)
                if '\t' in line:  # the line contains the user's utterance, then a tab, then the bot response
                    u, r = line.split('\t')
                    a = cand2idx[r]  # 'a' is already the index of the right candidate
                    u = _tokenize(u)
                    r = _tokenize(r)
                    data.append((context[:], u[:], a))

                    # Speaker and time notion encoded only for the next context
                    u.append(USER_UTTERANCE_ENCODING)
                    u.append(TIME_ENCODING.format(nid))  # '#1' --> '#2' --> '#3' --> '#4' --> etc.
                    r.append(BOT_RESPONSE_ENCODING)
                    r.append(TIME_ENCODING.format(nid))
                    context.append(u)
                    context.append(r)
                else:  # it's an API call, encoded as something said by the bot, and added to the context
                    r = _tokenize(line)
                    r.append(BOT_RESPONSE_ENCODING)
                    r.append(TIME_ENCODING.format(nid))
                    context.append(r)
            else:
                context = []  # clear the context at every empty line
    return data


def _eval_vocabulary(tokenized_dialogs, tokenized_candidates):
    """
    Given:
        - the dialogs with the tokenized context and the tokenized questions;
        - all the possible, tokenized, candidates responses;
    returns:
        - the vocabulary as a list of words;
        - a dict useful to convert every word in its corresponding index inside the memory.
    """
    vocabulary = set()
    for context, user_utterance, _ in tokenized_dialogs:
        flattened_context = [word for utterance in context for word in utterance]  # list of all the words in every memory
        vocabulary |= set(flattened_context + user_utterance)  # update the vocabulary set with all the words in context and user utterances
    for candidate in tokenized_candidates:
        vocabulary |= set(candidate)  # update the vocabulary set with all the words in the candidate bot's responses
    vocabulary = sorted(vocabulary)   # transform the vocabulary to a list and sort it
    word2idx = dict((word, idx) for idx, word in enumerate(vocabulary))
    return vocabulary, word2idx


def _eval_max_context_size(data):
    return max(len(context) for context, _, _ in data)


def _vectorize_candidates(tokenized_candidates, word2idx):
    """
    Returns a numpy 2D matrix representing all the candidates encoded as bags of words stacked side by side.
    """
    candidates_matrix = []
    vocabulary_size = len(word2idx)
    for tokenized_candidate in tokenized_candidates:
        candidate_as_bag = np.zeros(vocabulary_size, dtype=BAG_OF_WORDS_DTYPE)
        for word in tokenized_candidate:
            candidate_as_bag += np.cast[BAG_OF_WORDS_DTYPE](to_categorical(word2idx[word], vocabulary_size))
        candidates_matrix.append(candidate_as_bag)
    candidates_matrix = np.array(candidates_matrix, dtype=BAG_OF_WORDS_DTYPE)
    candidates_matrix = K.transpose(candidates_matrix)
    assert candidates_matrix.shape == (vocabulary_size, len(tokenized_candidates))
    return candidates_matrix


def _vectorize_data(data, word2idx, context_size):
    """
    Vectorize context facts (memories) and queries as bag of words.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.
    Bot's responses won't be changed, other than of course being changed from a list to a numpy array.
    Vectorized context, vectorized user's utterances and bot's answers as indices will be returned as
    three different numpy array.
    Example:
    [
      [
        [0 0 0 1 0 0 1 ... 0 0 1 ...],
        [1 0 0 1 0 0 0 ... 0 0 0 ...],
        [1 0 0 0 0 1 0 ... 0 0 0 ...],
        ...
      ],
      ...
    ],
    [
      [0 0 1 0 1 0 0 ... 1 1 0 ...],
      ...
    ],
    [
      420,
      ...
    ]
    """
    vocabulary_size = len(word2idx)

    contexts_as_bags = []
    user_utterances_as_bags = []
    bot_responses_as_indexes = [r for _, _, r in data]

    for tokenized_context, tokenized_user_utterance, _ in data:

        facts_as_bags = []
        for tokenized_fact in tokenized_context:
            fact_as_bag = np.zeros(vocabulary_size, dtype=BAG_OF_WORDS_DTYPE)
            for word in tokenized_fact:
                fact_as_bag += np.cast[BAG_OF_WORDS_DTYPE](to_categorical(word2idx[word], vocabulary_size))
            facts_as_bags.append(fact_as_bag)
        for _ in range(context_size - len(tokenized_context)):
            facts_as_bags.append(np.zeros(vocabulary_size, dtype=BAG_OF_WORDS_DTYPE))  # pad the context with empty memories
        contexts_as_bags.append(facts_as_bags)

        user_utterance_as_bag = np.zeros(vocabulary_size, dtype=BAG_OF_WORDS_DTYPE)
        for word in tokenized_user_utterance:
            user_utterance_as_bag += np.cast[BAG_OF_WORDS_DTYPE](to_categorical(word2idx[word], vocabulary_size))
        user_utterances_as_bags.append(user_utterance_as_bag)

    return (np.array(contexts_as_bags, dtype=BAG_OF_WORDS_DTYPE),           # [[[0 1 1 0 ...], [1 0 0 0 ...], ...], ...]
            np.array(user_utterances_as_bags, dtype=BAG_OF_WORDS_DTYPE),    # [[0 1 0 1 ...], ...]
            np.array(bot_responses_as_indexes, dtype=DEFAULT_KERAS_DTYPE))  # [420 69 ...]
