# Copyright 2016 Stanford University
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

import numpy as np
from six.moves import xrange
from flag import FLAGS
import random
from tensorflow.python.platform import gfile
import re

_PAD = b"<pad>"
_SOS = b"<sos>"
_EOS = b"<eos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")



def tokenize(string):
  return [int(s) for s in string.split()]


def pair_iter(fnamex, fnamey, batch_size, num_layers, sort_and_shuffle=True):
  fdx, fdy = open(fnamex), open(fnamey)
  batches = []

  while True:
    if len(batches) == 0:
      refill(batches, fdx, fdy, batch_size, sort_and_shuffle=sort_and_shuffle)
    if len(batches) == 0:
      break

    x_tokens, y_tokens = batches.pop(0)
    y_tokens = add_sos_eos(y_tokens)
    x_padded, y_padded = padded(x_tokens, num_layers), padded(y_tokens, 1)

    source_tokens = np.array(x_padded).T
    source_mask = (source_tokens != PAD_ID).astype(np.float32)
    target_tokens = np.array(y_padded).T
    target_mask = (target_tokens != PAD_ID).astype(np.float32)

    yield (source_tokens, source_mask, target_tokens, target_mask)

  return


def refill(batches, fdx, fdy, batch_size, sort_and_shuffle=True):
  line_pairs = []
  linex, liney = fdx.readline(), fdy.readline()

  while linex and liney:
    x_tokens, y_tokens = tokenize(linex), tokenize(liney)

    if len(x_tokens) < FLAGS.max_seq_len and len(y_tokens) < FLAGS.max_seq_len:
      line_pairs.append((x_tokens, y_tokens))
    # if len(line_pairs) == batch_size * 16:
    #   break
    linex, liney = fdx.readline(), fdy.readline()

  if sort_and_shuffle:
    random.shuffle(line_pairs)
    line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))

  for batch_start in xrange(0, len(line_pairs), batch_size):
    x_batch, y_batch = zip(*line_pairs[batch_start:batch_start+batch_size])
#    if len(x_batch) < batch_size:
#      break
    batches.append((x_batch, y_batch))

  if sort_and_shuffle:
    random.shuffle(batches)
  return


def add_sos_eos(tokens):
  return map(lambda token_list: [SOS_ID] + token_list + [EOS_ID], tokens)


def padded(tokens, depth):
  maxlen = max(map(lambda x: len(x), tokens))
  align = pow(2, depth - 1)
  padlen = maxlen + (align - maxlen) % align
  return map(lambda token_list: token_list + [PAD_ID] * (padlen - len(token_list)), tokens)


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
