#!/usr/bin/env python2

# Generating a tagged sentence for input query

import os
from os import path
from os.path import join
import sys

import gflags
from nltk.parse import stanford
from nltk import tree

import codebase.tagger
import codebase.tag_utils as tu

gflags.DEFINE_string(
    'stanford_parser',
    '../deep_parser',
    'env STANFORD_PARSER')
gflags.DEFINE_string(
    'stanford_models',
    '../deep_parser',
    'env STANFORD_MODELS')
gflags.DEFINE_string(
    'stanford_model_path',
    '../deep_parser/englishPCFG.ser.gz',
    'Stanford Parser\'s model_path')
gflags.DEFINE_string('data_root', '/home/wzw0022/NLIDB/etc/overnight-tagger/data_root/', 'path to data_root')
gflags.DEFINE_boolean("recover", False,
                      "Set to True for interactive decoding.")

FLAGS = gflags.FLAGS


def mainON(field2word, subset):
    ''' process data, from .qu, .lo, and .fi
        to .ta, .lox, .qux
        and .ficorr, .vacorr
    '''

    sub_folder = subset.split('_')[0]
    data_root = FLAGS.data_root
    os.environ['STANFORD_PARSER'] = FLAGS.stanford_parser
    os.environ['STANFORD_MODELS'] = FLAGS.stanford_models
    parser = stanford.StanfordParser(model_path=FLAGS.stanford_model_path)
    schema = ' '.join(field2word.keys())

    if not path.isdir(join(data_root, 'overnight_generated')):
        os.makedirs(join(data_root, 'overnight_generated'))

    (f_ta, f_lox, f_qux, f_ficorr, f_vacorr) = [
        open(
            join(data_root, 'overnight_generated', '%s.%s' % (subset, suffix)),
            'w') for suffix in ['ta', 'lox', 'qux', 'ficorr', 'vacorr']
    ]

    with open(data_root + 'overnight_source/%s/%s.qu' % (sub_folder,subset)) as f_qu, open(
            data_root + 'overnight_source/%s/%s.lon' % (sub_folder,subset)) as f_lo:
        query, logic = f_qu.readline(), f_lo.readline()
        idx = 0
        while query and logic:
            idx += 1
            print '### example: %d ###' % idx
            print query
            print logic
            tagged2, field_corr, value_corr, newquery, newlogical = codebase.tagger.sentTagging_treeON3(
                parser, field2word, query, schema, logic)
            print field_corr
            print value_corr
            print tagged2
            print newquery
            print newlogical
            print '\n'
            f_qux.write(newquery + '\n')
            f_lox.write(newlogical + '\n')
            f_ficorr.write(field_corr + '\n')
            f_vacorr.write(value_corr + '\n')
            f_ta.write(tagged2 + '\n')
            query, logic = f_qu.readline(), f_lo.readline()

    f_ta.close()
    f_lox.close()
    f_qux.close()
    f_vacorr.close()
    f_ficorr.close()
    return


def tag():
    config = tu.Config()

    mainON(config.housing_dict, 'housing_train')
    mainON(config.housing_dict, 'housing_test')
    mainON(config.housing_dict, 'housing_aug')
    mainON(config.recipes_dict, 'recipes_train')
    mainON(config.recipes_dict, 'recipes_aug')
    mainON(config.recipes_dict, 'recipes_test')
    mainON(config.basketball_dict, 'basketball_train')
    mainON(config.basketball_dict, 'basketball_test')
    mainON(config.basketball_dict, 'basketball_aug')
    mainON(config.calendar_dict, 'calendar_train')
    mainON(config.calendar_dict, 'calendar_aug')
    mainON(config.calendar_dict, 'calendar_test')
    mainON(config.restaurants_dict, 'restaurants_train')
    mainON(config.restaurants_dict, 'restaurants_aug')
    mainON(config.restaurants_dict, 'restaurants_test')
    return


def recover():
    return


if __name__ == "__main__":
    FLAGS(sys.argv)
    if FLAGS.recover:
        recover()
    else:
        tag()
