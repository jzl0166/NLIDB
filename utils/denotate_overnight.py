import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from redenote import get_fields,renotate


fdict = defaultdict(list)
def denotate(s='test'):

    all_fields = get_fields()[sub]

    ta_file = os.path.join(path, '%s_%s.ta'%(sub,s))
    qu_file = os.path.join(path, '%s_%s.qu'%(sub,s))
    question_file = os.path.join(path, '%s.qu'%s)
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
                    new += ('<'+words[0][1]+words[2]+'>')
                    if words[0][1]=='f' or words[0][1]=='v':
                        #new += ('<'+words[0][1]+words[2]+'>')
                        new += ' '
                        new += q_token
                        new += ' '
                        new += '<eof>'  
                new += ' '

            
            new += '<eos> '
            for i,f in enumerate(all_fields):
                new += '<c'+str(i)+'> '+f+' <eoc> '
            

            re.write(new+'\n')

    print('question file done.')

    lox_file = os.path.join(path, '%s_%s.lox'%(sub,s))
    lon_file = os.path.join(path, '%s_%s.lon'%(sub,s))
    lo_file = os.path.join(path, '%s.lon'%s)
    new_file = os.path.join(path, 'new_%s.lon'%s)
    with gfile.GFile(lox_file, mode='r') as lox,gfile.GFile(lon_file, mode='r') as lon, gfile.GFile(lo_file, mode='w') as re:
        loxs = lox.readlines()
        lons = lon.readlines()
        n = len(lons)
        error = 0
       
       
        assert len(lons)==len(loxs)
        #newline is redenotdated file
        for lox, lon in zip(loxs,lons):
            lo_tokens = lox.split()
            lon_tokens = lon.split()

   

            t_tokens = template.split()
            q_tokens = question.split()
            assert len(t_tokens)==len(q_tokens)
     
            for t_token,q_token in zip(t_tokens,q_tokens):
                if t_token=='<nan>' or t_token=='<count>':
                    pass
                else:
                    words = t_token.split(':')
                    note = ('<'+words[0][1]+words[2]+'>')
       


            new = ''
            for lo_token,lon_token in zip(lo_tokens,lon_tokens):
                if ':' in lo_token and len(lo_token.split(':'))==3:
                    words = lo_token.split(':')
                    if False and words[0][1]=='f':
                        new += ('<'+words[0][1]+words[2]+'>')
                        new += ' '
                        new += lon_token
                        new += ' '
                        new += '<eof>'
                    else:
                        new += ('<'+words[0][1]+words[2]+'>')
                elif lo_token=='<count>':
                    new += lon_token
                else:
                    new += lo_token
                new += ' '
            re.write(new+'\n')
            
           
    print('logic file done.')

if __name__ == "__main__":
    for sub in ['basketball','calendar','housing','recipes','restaurants']:
        path = os.path.abspath(__file__)
        path = os.path.dirname(path).replace('utils','data/DATA/overnight_source/%s'%sub)
        denotate('train')
        denotate('test')
        renotate('train',sub)
        renotate('test',sub)

