# Copyright 2016 Stanford University
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import gzip
import os
import re
import tarfile
from os.path import join as pjoin
from six.moves import urllib
import tensorflow as tf
from tensorflow.python.platform import gfile


tf.app.flags.DEFINE_integer("max_vocab_size", 40000, "Vocabulary size limit.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("tokenizer", "CHAR", "BPE / CHAR / WORD.")
tf.app.flags.DEFINE_boolean("gen_voc", False, "whether to generate the vocabulary")
tf.app.flags.DEFINE_string("prefix", None, "which file to tokenize")
FLAGS = tf.app.flags.FLAGS


# Special vocabulary symbols - we always put them at the start.
_PAD = b"<pad>"
_SOS = b"<sos>"
_EOS = b"<eos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]

def char_tokenizer(sentence):
    return list(sentence.strip())

def bpe_tokenizer(sentence):
    tokens = sentence.strip().split()
    tokens = [w + "</w>" if not w.endswith("@@") else w for w in tokens]
    tokens = [w.replace("@@", "") for w in tokens]
    return tokens

def get_tokenizer(tokenizer):
  if tokenizer.lower() == 'bpe':
    return bpe_tokenizer
  elif tokenizer.lower() == 'char':
    return char_tokenizer
  elif tokenizer.lower() == 'word':
    return basic_tokenizer
  else:
    raise
  return tokenizer


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

def create_vocabulary(data_dir, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False, flag_ascii=True):
    data_paths = {pjoin(data_dir, 'train.x.txt'), pjoin(data_dir, 'train.y.txt')}
    ## Create vocabularies of the appropriate sizes.
    vocabulary_path = os.path.join(data_dir, "vocab.dat")
    print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
    vocab = {}
    for path in data_paths:
        with gfile.GFile(path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                # Remove non-ASCII characters
                #if flag_ascii:
                #    line = remove_nonascii(line)
                line = line.strip()
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    print("Vocabulary size: %d" % len(vocab_list))
    if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
            if w in _START_VOCAB:
                vocab_file.write(w + b"\n")
            else:
                print(vocab[w])
                if vocab[w] > 1000:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path, bpe=False):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        # Call ''.join below since BPE outputs split pairs with spaces
        if bpe:
            vocab = dict([(''.join(x.split(' ')), y) for (y, x) in enumerate(rev_vocab)])
        else:
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path, bpe=(tokenizer==bpe_tokenizer))
    with open(data_path, mode="r") as data_file:
        with open(target_path, mode="w") as tokens_file:
            counter = 0
            for line in data_file:
                counter += 1
                if counter % 100000 == 0:
                    print("  tokenizing line %d" % counter)
                #line = remove_nonascii(line).strip()
                line = line.strip()
                token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                  normalize_digits)
                tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def tokenize_data(data_dir, prefix, tokenizer):
    vocab_path = os.path.join(data_dir, "vocab.dat")
    path_x = os.path.join(data_dir,  prefix + ".x.txt")
    path_y = os.path.join(data_dir, prefix + ".y.txt")
    # Create token ids for the training data.
    y_ids_path = os.path.join(data_dir, prefix + ".ids.y")
    x_ids_path = os.path.join(data_dir, prefix + ".ids.x")
    data_to_token_ids(path_x, x_ids_path, vocab_path, tokenizer)
    data_to_token_ids(path_y, y_ids_path, vocab_path, tokenizer)
    return (x_ids_path, y_ids_path)


def main(_):
    if FLAGS.gen_voc:
        create_vocabulary(FLAGS.data_dir, FLAGS.max_vocab_size, tokenizer=get_tokenizer(FLAGS.tokenizer))
    if FLAGS.prefix is not None:
        tokenize_data(FLAGS.data_dir, FLAGS.prefix, tokenizer=get_tokenizer(FLAGS.tokenizer))

if __name__ == "__main__":
    tf.app.run()
