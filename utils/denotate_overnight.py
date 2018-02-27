import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

path = '/home/wzw0022/all_contrib_dev_test/data/DATA/overnight_source/all'

fdict = defaultdict(list)
def denotate(s='test'):
    ta_file = os.path.join(path, 'all_%s.ta'%s)
    qu_file = os.path.join(path, 'all_%s.qu'%s)
    question_file = os.path.join(path, 'new_all_%s.qu'%s)
    with gfile.GFile(ta_file, mode='r') as t, gfile.GFile(qu_file, mode='r') as q, gfile.GFile(question_file, mode='w') as re:
        templates = t.readlines()
        questions = q.readlines()
        assert len(templates)==len(questions)
        for template,question in zip(templates,questions):
            t_tokens = template.split()
            q_tokens = question.split()
            assert len(t_tokens)==len(q_tokens)
            new = ''
            for t_token,q_token in zip(t_tokens,q_tokens):
                if t_token=='<nan>' or t_token=='<count>':
                    new += q_token
                else:
                    words = t_token.split(':')
                    new += (words[0][1]+words[2])
                new += ' '
            re.write(new+'\n')
    print('question file done.')
    lox_file = os.path.join(path, 'all_%s.lox'%s)
    lon_file = os.path.join(path, 'all_%s.lon'%s)
    lo_file = os.path.join(path, 'new_all_%s.lon'%s)
    with gfile.GFile(lox_file, mode='r') as lox,gfile.GFile(lon_file, mode='r') as lon, gfile.GFile(lo_file, mode='w') as re:
        loxs = lox.readlines()
        lons = lon.readlines()
        assert len(lons)==len(loxs)
        for lox, lon in zip(loxs,lons):
            lo_tokens = lox.split()
            lon_tokens = lon.split()
            new = ''
            for lo_token,lon_token in zip(lo_tokens,lon_tokens):
                if ':' in lo_token and len(lo_token.split(':'))==3:
                    words = lo_token.split(':')
                    new += (words[0][1]+words[2])
                elif lo_token=='<count>':
                    new += lon_token
                else:
                    new += lo_token
                new += ' '
            re.write(new+'\n')
    print('logic file done.')


denotate('train')
denotate('test')
