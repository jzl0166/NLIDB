#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

import tensorflow as tf
from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_NAN = b"<nan>"
_F0 = b"<field>:int:0"
_F1 = b"<field>:int:1"
_F2 = b"<field>:int:2"
_F3 = b"<field>:int:3"
_F4 = b"<field>:int:4"
_V0 = b"<value>:int:0"
_V1 = b"<value>:int:1"
_V2 = b"<value>:int:2"
_V3 = b"<value>:int:3"
_V4 = b"<value>:int:4"
_F10 = b"<field>:string:0"
_F11 = b"<field>:string:1"
_F12 = b"<field>:string:2"
_F13 = b"<field>:string:3"
_F14 = b"<field>:string:4"
_V10 = b"<value>:string:0"
_V11 = b"<value>:string:1"
_V12 = b"<value>:string:2"
_V13 = b"<value>:string:3"
_V14 = b"<value>:string:4"
_F20 = b"<field>:bool:0"
_F21 = b"<field>:bool:1"
_F22 = b"<field>:bool:2"
_F23 = b"<field>:bool:3"
_F24 = b"<field>:bool:4"
_V20 = b"<value>:bool:0"
_V21 = b"<value>:bool:1"
_V22 = b"<value>:bool:2"
_V23 = b"<value>:bool:3"
_V24 = b"<value>:bool:4"
_F30 = b"<field>:date:0"
_F31 = b"<field>:date:1"
_F32 = b"<field>:date:2"
_F33 = b"<field>:date:3"
_F34 = b"<field>:date:4"
_V30 = b"<value>:date:0"
_V31 = b"<value>:date:1"
_V32 = b"<value>:date:2"
_V33 = b"<value>:date:3"
_V34 = b"<value>:date:4"
_F40 = b"<field>:time:0"
_F41 = b"<field>:time:1"
_F42 = b"<field>:time:2"
_F43 = b"<field>:time:3"
_F44 = b"<field>:time:4"
_V40 = b"<value>:time:0"
_V41 = b"<value>:time:1"
_V42 = b"<value>:time:2"
_V43 = b"<value>:time:3"
_V44 = b"<value>:time:4"

_START_VOCAB = [_PAD, _GO, _EOS, _UNK, _NAN, _F0, _F1, _F2, _F3, _F4, _V0, _V1, _V2, _V3, _V4, \
                _F10, _F11, _F12, _F13, _F14, _V10, _V11, _V12, _V13, _V14, \
                _F20, _F21, _F22, _F23, _F24, _V20, _V21, _V22, _V23, _V24, \
                _F30, _F31, _F32, _F33, _F34, _V30, _V31, _V32, _V33, _V34, \
                _F40, _F41, _F42, _F43, _F44, _V40, _V41, _V42, _V43, _V44]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
NAN_ID = 4
F0_ID = 5
F1_ID = 6
F2_ID = 7
F3_ID = 8
F4_ID = 9
V0_ID = 10
V1_ID = 11
V2_ID = 12
V3_ID = 13
V4_ID = 14
F10_ID = 15
F11_ID = 16
F12_ID = 17
F13_ID = 18
F14_ID = 19
V10_ID = 20
V11_ID = 21
V12_ID = 22
V13_ID = 23
V14_ID = 24
F20_ID = 25
F21_ID = 26
F22_ID = 27
F23_ID = 28
F24_ID = 29
V20_ID = 30
V21_ID = 31
V22_ID = 32
V23_ID = 33
V24_ID = 34
F30_ID = 35
F31_ID = 36
F32_ID = 37
F33_ID = 38
F34_ID = 39
V30_ID = 40
V31_ID = 41
V32_ID = 42
V33_ID = 43
V34_ID = 44
F40_ID = 45
F41_ID = 46
F42_ID = 47
F43_ID = 48
F44_ID = 49
V40_ID = 50
V41_ID = 51
V42_ID = 52
V43_ID = 53
V44_ID = 54

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([,!?\")(])")  # get rid of '.':;'
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path,
                      data_path,
                      max_vocabulary_size,
                      tokenizer=None,
                      normalize_digits=False,
                      crop=False):
    """Create vocabulary file (if it does not exist yet) from data file.
  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.
  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path,
                                                       data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(
                    line)
                for w in tokens:
                    #word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                    if word in _START_VOCAB:  # 0513 Adding field/value constant
                        continue
                    elif word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            if crop == True:
                vocab_list = _START_VOCAB + sorted(
                    vocab, key=vocab.get, reverse=True)
            else:
                vocab_list = _START_VOCAB[:5] + sorted(
                    vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.
  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].
  Args:
    vocabulary_path: path to the file containing the vocabulary.
  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).
  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence,
                          vocabulary,
                          tokenizer=None,
                          normalize_digits=False):
    """Convert a string to list of integers representing token-ids.
  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  Returns:
    a list of integers, the token-ids for the sentence.
  """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    else:
        # Normalize digits by 0 before looking words up in the vocabulary.
        return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path,
                      target_path,
                      vocabulary_path,
                      tokenizer=None,
                      normalize_digits=False):
    """Tokenize data file and turn into token-ids using given vocabulary file.
  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.
  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(
                        tf.compat.as_bytes(line), vocab, tokenizer,
                        normalize_digits)
                    tokens_file.write(" ".join([str(tok)
                                                for tok in token_ids]) + "\n")


def prepare_wmt_data(data_s_dir,
                     data_g_dir,
                     subset,
                     en_vocabulary_size,
                     fr_vocabulary_size,
                     tokenizer=None):
    """Get WMT data into data_dir, create vocabularies and tokenize data.
  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
    # Get data to the specified directory.
    # train_path = os.path.join(data_dir, "rand_train")
    # dev_path = os.path.join(data_dir, "rand_dev")

    #subset = 'restaurants'

    from_train_path = os.path.join(data_s_dir, "%s_train" % subset) + ".qu"
    to_train_path = os.path.join(
        data_g_dir,
        "%s_train" % subset) + ".lox"  # we have a new logical form, 04/20/2017
    tag_train_path = os.path.join(data_g_dir, "%s_train" % subset) + ".ta"
    from_dev_path = os.path.join(data_s_dir, "%s_test" % subset) + ".qu"
    to_dev_path = os.path.join(
        data_g_dir,
        "%s_test" % subset) + ".lox"  # we have a new logical form, 04/20/2017
    tag_dev_path = os.path.join(data_g_dir, "%s_test" % subset) + ".ta"
    return prepare_data(data_s_dir, data_g_dir, from_train_path, to_train_path,
                        tag_train_path, from_dev_path, to_dev_path,
                        tag_dev_path, en_vocabulary_size, fr_vocabulary_size,
                        tokenizer)


def prepare_data(data_s_dir,
                 data_g_dir,
                 from_train_path,
                 to_train_path,
                 tag_train_path,
                 from_dev_path,
                 to_dev_path,
                 tag_dev_path,
                 from_vocabulary_size,
                 to_vocabulary_size,
                 tokenizer=None):
    """Preapre all necessary files that are required for the training.
    Args:
      data_dir: directory in which the data sets will be stored.
      from_train_path: path to the file that includes "from" training samples.
      to_train_path: path to the file that includes "to" training samples.
      from_dev_path: path to the file that includes "from" dev samples.
      to_dev_path: path to the file that includes "to" dev samples.
      from_vocabulary_size: size of the "from language" vocabulary to create and use.
      to_vocabulary_size: size of the "to language" vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
    """
    # Create vocabularies of the appropriate sizes.
    to_vocab_path = os.path.join(data_g_dir, "vocab%d.to" % to_vocabulary_size)
    from_vocab_path = os.path.join(data_s_dir,
                                   "vocab%d.from" % from_vocabulary_size)
    create_vocabulary(
        to_vocab_path, to_train_path, to_vocabulary_size, tokenizer, crop=True)
    create_vocabulary(from_vocab_path, from_train_path, from_vocabulary_size,
                      tokenizer)

    # Create token ids for the training data.
    to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
    from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
    tag_train_ids_path = tag_train_path + (".ids%d" % to_vocabulary_size)
    # tag_train_ids_path = tag_train_path + (".ids%d" % from_vocabulary_size)
    data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path,
                      tokenizer)
    data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path,
                      tokenizer)
    data_to_token_ids(tag_train_path, tag_train_ids_path, to_vocab_path,
                      tokenizer)
    #data_to_token_ids(tag_train_path, tag_train_ids_path, from_vocab_path, tokenizer)

    # Create token ids for the development data.
    to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
    from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
    tag_dev_ids_path = tag_dev_path + (".ids%d" % to_vocabulary_size)
    # tag_dev_ids_path = tag_dev_path + (".ids%d" % from_vocabulary_size)
    data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
    data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path,
                      tokenizer)
    data_to_token_ids(tag_dev_path, tag_dev_ids_path, to_vocab_path, tokenizer)
    #data_to_token_ids(tag_dev_path, tag_dev_ids_path, from_vocab_path, tokenizer)

    return (from_train_ids_path, to_train_ids_path, tag_train_ids_path,
            from_dev_ids_path, to_dev_ids_path, tag_dev_ids_path,
            from_vocab_path, to_vocab_path)
