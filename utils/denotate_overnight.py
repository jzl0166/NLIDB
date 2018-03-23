import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

subset = 'all'
path = '/home/wzw0022/forward_wiki/data/DATA/overnight_source/%s'%subset

fdict = defaultdict(list)
def denotate(s='test'):
    ta_file = os.path.join(path, '%s_%s.ta'%(subset,s))
    qu_file = os.path.join(path, '%s_%s.qu'%(subset,s))
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
                    if words[0][1]=='f':
                        #new += ('<'+words[0][1]+words[2]+'>')
                        new += ' '
                        new += q_token
                        new += ' '
                        new += '<eof>'  
                new += ' '
            re.write(new+'\n')
    print('question file done.')
    lox_file = os.path.join(path, '%s_%s.lox'%(subset,s))
    lon_file = os.path.join(path, '%s_%s.lon'%(subset,s))
    lo_file = os.path.join(path, '%s.lon'%s)
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

def redenotate(s='test'):
    basketball_fs = [ 'season','position','team','player']
    error = 0
    qu_file = os.path.join(path, '%s.qu'%s)
    lon_file = os.path.join(path, '%s.lon'%s)
    new_lon_file = os.path.join(path, 'new_%s.lon'%s)
    new_qu_file = os.path.join(path, 'new_%s.qu'%s)
    ori_lon_file = os.path.join(path, '%s_%s.lon'%(subset,s))
    with gfile.GFile(qu_file, mode='r') as qu, gfile.GFile(lon_file, mode='r') as lon,gfile.GFile(new_lon_file, mode='w') as new_lon,gfile.GFile(new_qu_file, mode='w') as new_qu, gfile.GFile(ori_lon_file, mode='r') as ori_lon:
        qu_lines = qu.readlines()
        lon_lines = lon.readlines()
        ori_lons = ori_lon.readlines()
        assert len(qu_lines) == len(lon_lines)
        for Q, S, S0 in zip(qu_lines,lon_lines,ori_lons):
            syms = ['<f0>','<f1>','<f2>','<f3>'] 
            for sym in syms:
                if sym in S:
                    if sym not in Q:
                        idx = S.split().index(sym)
                        newp = S0.split()[idx]
                        print(newp)
                        S = S.replace(sym,newp)
                        error += 1
                        #print('------------')
                        #print(Q)
                        #print(S)
                        #print(S0)

            '''
            for i,f in enumerate(basketball_fs):
                Q = Q.replace('\n',' <eos>')
                Q += ( ' <c'+str(i)+'> '+f+' <eoc> ' )
            
            for i,f in enumerate(basketball_fs):
                S = S.replace(f,'<c'+str(i)+'>')
            new_qu.write(Q+'\n')
            '''
            new_lon.write(S)
            
    print(error*1.0/len(qu_lines))

if __name__ == "__main__":
    denotate('train')
    denotate('test')
    redenotate('train')
    redenotate('test')
