# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Data generators for NLIDB data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_WIKI_ALL = [
    [
        "wiki",
        ("wiki/train.qu","wiki/train.lon")
    ],
    [
        "wiki",
        ("wiki/test.qu","wiki/test.lon")
    ],
    [
        "wiki",
        ("wiki/dev.qu","wiki/dev.lon")
    ]

]

_WIKI_TRAIN_LARGE_DATA = [
    [
        "wiki",
        ("wiki/train.qu", "wiki/train.lon")
    ],
]

_WIKI_TEST_LARGE_DATA = [
    [
        "wiki",
        ("wiki/dev.qu", "wiki/dev.lon")
    ],
]

#wenlu add
class DetTextEncoder():
    """Add this class to help decoding, and build embedding matrix"""
    def __init__(self, vocfile):
        import codecs
	# just stupid code that builds a vocabulary dict
        self.v = {}
        self.id2w = {}
        with codecs.open(vocfile,"r","utf-8") as f:
            for i,l in enumerate(f):
                s=l.strip()
                if len(s)>0:
                    s=s.replace('\'','').replace('_','')
                    self.v[s] = i
                    self.id2w[i] = s

    def encode(self, sentence):
        """Converts a space-separated string of tokens to a list of ids."""
        ret = []
        for tok in sentence.strip().split():
            if tok in self.v: ret.append(self.v[tok])
            else: ret.append(self.v['UNK'])
        if self._reverse: ret = ret[::-1]
        return ret

    def decode(self, ids):
        """decode a list of indexes to words"""
        toks=[]
        for i in ids:
            w = self.id2w[i]
            toks.append(w)
        return ' '.join(toks)

    def decode_one(self, i):
        """decode single index to word"""
        return self.id2w[i]

    @property
    def vocab_size(self):
        return len(self.v)


@registry.register_problem
class NlidbWiki(translate.TranslateProblem):

  @property
  def approx_vocab_size(self):
     return 10048

  @property
  def vocab_filename(self):
     return "vocab.wiki.%d" % self.approx_vocab_size

  def source_data_files(self, dataset_split):
     train = dataset_split == problem.DatasetSplit.TRAIN
     datasets = _WIKI_TRAIN_LARGE_DATA if train else _WIKI_TEST_LARGE_DATA
     return datasets

  def vocab_data_files(self):
     return _WIKI_ALL

  @property
  def vocabulary(self):
      wvoc = DetTextEncoder("/home/gongzhitaao/t2t_data/vocab.wiki.10048")
      return wvoc



