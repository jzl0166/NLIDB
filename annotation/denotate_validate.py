# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import sys
import json
from argparse import ArgumentParser
from tqdm import tqdm
from lib.query import Query
from lib.common import count_lines
import numpy as np
import re
import collections
from collections import defaultdict
import copy
import editdistance as ed
from scipy import spatial
from utils.glove import Glove
path = os.path.abspath(__file__)
data_path = os.path.dirname(path).replace('annotation','data')
glove = Glove()

maps = defaultdict(list)
stop_words = ['a','of','the','in']
def _match_field(name_pairs,candidates):

    if len(name_pairs)!=len(candidates):
        return False
    return name_pairs==candidates
  

def _preclean(Q):
    Q = re.sub('#([0-9])',r'# \1', Q)
    Q = Q.replace('€',' €')
    Q = Q.replace('\'',' ')
    Q = Q.replace(',','')
    Q = Q.replace('?','')
    Q = Q.replace('\"','')
    Q = Q.replace('(s)','')
    Q = Q.replace('  ',' ')
    return Q.lower()

def _clean(Q):
    Q = Q.replace('<eof>s','<eof>')
    Q = Q.replace('<eof>d','<eof>')
    return Q

def _check_head(head,Q):
    question_words = ['what','what\'s','whats','whatis','how','which','list','who','who\'s','whos','give','tell','name','where']
    
    Q0 = Q
    idx = Q.index(head)
    Q = Q[:idx]
    tokens = Q.split()


    for i in range(1,8):
        if len(tokens)-i>=0 and tokens[-i] in question_words:
            return True

    return False


def _max_span(ids,tokens):

    if len(ids)<1:
        return '',-1
    
    intervals = []
    for i in ids:
        if not intervals:
            intervals.append([i])
        else:
            ADD = False
            for interval in intervals:
                if i == interval[0]-1:
                    interval.insert(0,i)
                    ADD = True
                elif i == interval[-1]+1:
                    interval.append(i)
                    ADD = True

            if not ADD:
                intervals.append([i])

    #intervals.sort(key=len, reverse=True)

    max_len = -1
    max_interval = []

    for interval in intervals:
        l = 0
        for idx in interval:
            if tokens[idx] not in stop_words:
                l += 1
        if l > max_len:
            max_len = l
            max_interval = interval


    re = ''
    for i in max_interval[:-1]:
        re += tokens[i] 
        re += ' '
    re += tokens[max_interval[-1]]
    
    return re, max_len

def _abbr_match(a,b):
    if a[-1]!='.' and b[-1]!='.':
        return False

    if a[-1]=='.':
        idx = a.index('.')
        if len(b)>idx and a[:idx]==b[:idx]:
            return True
    else:
        idx = b.index('.')
        if len(a)>idx and  a[:idx]==b[:idx]:
            return True

    return False

#imperfect match
def _match_ids(field,Q):
    ids = []
    length = 0
    for token in field.replace('/',' ').split():
        if token == 'no.':
            token = 'number'

        for i,q in enumerate(Q.split()):
            semantic_sim = 1-spatial.distance.cosine(glove.embed_one(q), glove.embed_one(token))
            if q in stop_words or _abbr_match(q,token) or ed.eval(q,token)/len(token) < 0.5 or semantic_sim>=0.7 :
                ids.append(i)           
                if q not in stop_words:
                    length += 1

    return ids

#overlap threshold
def _threshold(f):
    tokens = []
    for t in f.split():
        if t not in stop_words:
            tokens.append(t)
    return len(tokens)/2


def main():
    parser = ArgumentParser()
    parser.add_argument('--din', default=data_path, help='data directory')
    parser.add_argument('--dout', default='annotated', help='output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    
    for split in ['train','test','dev']:
     
        with open('%s.qu'%split, 'w') as qu_file, open('%s.lon'%split, 'w') as lon_file, open('%s.out'%split, 'w') as out,open('%s_sym_pairs.txt'%split, 'w') as sym_file, open('%s_ground_truth.txt'%split, 'w') as S_file:
          
            fsplit = os.path.join(args.din, split) + '.jsonl'
            ftable = os.path.join(args.din, split) + '.tables.jsonl'

            with open(fsplit) as fs, open(ftable) as ft :
                print('loading tables')
                tables = {}

                

                for line in tqdm(ft, total=count_lines(ftable)):
                    d = json.loads(line)
                    tables[d['id']] = d
                    
                print('loading examples')
                n = 0
                acc = 0
                acc_pair = 0
                acc_all = 0
                target = -1
            
                error = 0
                ADD_FIELDS = False
                step2 = True
             
                poss = 0
                for line in tqdm(fs, total=count_lines(fsplit)):
                    ADD_TO_FILE = True
                    d = json.loads(line)
                    
                    Q = d['question']
                    Q = Q.replace(u'\xa0',u' ')
                    rows = tables[d['table_id']]['rows']
                    rows = np.asarray(rows)

                    fs = tables[d['table_id']]['header']

                 
                    l = 0
                    for f in fs:
                        l += ( len(f.split()) + 2)

                    all_fields = []
                    for f in fs:
                        all_fields.append(_preclean(f))

                    all_fields = sorted(all_fields, key=len, reverse=True)
                    
                    if l>40:
                        pass
                        #error += 1

                    #f2v
                    smap = defaultdict(list)
                    #v2f
                    reverse_map = defaultdict(set)
                    for row in rows:
                        for i in range(len(fs)):
                            cur_f = _preclean(str(fs[i])) 
                            cur_row =  _preclean(str(row[i]))  
                            smap[cur_f].append(cur_row)
                            reverse_map[cur_row].add(cur_f)
                  
                    #----------------------------------------------------------                    
                  
                    #all values
                    keys = sorted(reverse_map.keys(), key=len, reverse=True)

                    candidates = []
                    cond_fields = []
        
                    Q = _preclean(Q)
                    Q_ori = Q
            
                    #========================MATCH VALUES & FIELDS=============================                    
                    field2partial = {}
                    field2partial = defaultdict(lambda:'unk', field2partial)

                    for v in keys: 
                        l = len(v.split())
                        PASS = False
                        

                        fs = reverse_map[v]
                        if v.isdigit() and v not in Q.split():
                            v = str(v)+'.0'


                        if ( l>1 and v in Q) or ( l==1 and str(v) in Q.split() ):
                            
                            #PASS if current v is a substring of matched v
                            for field,value in candidates:
                                if v+' ' in value or ' '+v in value or (v in value and abs(len(v)-len(value))<2 ):
                                    PASS = True
                               

                            if PASS == False:        
                                
                                fs = list(fs)
              
                                if len(fs)==1:
                                    f = fs[0]
                                    cond_fields.append(f)
                                    
                                else:
                    
                                    #TODO only partial match while len(fs)==0
                                    fs_ori = copy.copy(fs)
                                    fs_tmp = copy.copy(fs)
                                    for field in fs_tmp:
                                        if field not in Q:
                                            fs.remove(field)

                                    if len(fs)==1:
                                        f = fs[0]
                                        cond_fields.append(f)

                                    if len(fs)>1: #match for field closest to v
                                        fs.sort(key=lambda x: abs(Q_ori.index(x)-Q_ori.index(v)) )
                                        f = fs[0]
                                        cond_fields.append(f)

                                    if len(fs)==0:
                                        fs = copy.copy(fs_ori)
                                        fs_tmp = copy.copy(fs)

                                        for field in fs_tmp:
                                            if field not in Q:
                                                ids = _match_ids(field,Q)
                                                replacement, match_len=_max_span(ids, Q.split())

                                                if match_len >= _threshold(field):
                                                    field2partial[field] = replacement
                                                    #if replacement not in Q:
                                                    #    print('******')
                                                    #    print(Q)
                                                    #    print(replacement)
                                                else:
                                                    fs.remove(field) 
                                            else:
                                                field2partial[field] = field

                                        if len(fs)==0:
                                            f = 'unk'
                                            cond_fields.append(f)
                                        
                                 
                                        if len(fs)==1:
                                            f = fs[0]
                                            cond_fields.append(f)
                                          
                                        
                                        if len(fs)>1: #match for field closest to v
                                            fs.sort(key=lambda x: abs(Q_ori.index(field2partial[x])-Q_ori.index(v)) )
                                            f = fs[0]
                                            cond_fields.append(f)
                                 
                                        
                                if f==v:   
                                    cond_fields.remove(f)
                                    pass   
                                    #candidates.append([_preclean(f),'true'])
                                else:
                                    candidates.append([_preclean(f),_preclean(v)])
                                    
                                
                    candidates.sort( key=lambda x: x[1] )
                  
                    #=============================MATCH HEAD====================================
                    head_cands = []
                    heads = smap.keys()
                    heads = sorted(heads, key=len, reverse=True)

                    for f in heads:
                        ADD = True
                        for h in head_cands:
                            if f in h:
                                ADD = False
                                break

                        if ADD and f in Q_ori and _check_head(f, Q_ori) and f not in cond_fields:
                            head_cands.append(f)


                    if len(head_cands)>1:
                        head_cands.sort(key=lambda x: Q_ori.index(x) )
                        head = head_cands[0]
                    elif len(head_cands)==1:
                        head = head_cands[0]
                    elif len(heads)==1:
                        head = heads[0]
                    else:
                        head = ''

                              

                    #========================MATCH PARTIAL/VARIATION HEAD=====================
                 
                    head2partial = {}
                    head2partial = defaultdict(lambda:'unk', head2partial)

                    if head == '' :
                        max_len = -1
                        for field in all_fields:

                            if field not in cond_fields:

                                ids = _match_ids(field,Q)
                                replacement, match_len=_max_span(ids, Q.split()) 
                                if match_len >= _threshold(field) and match_len>=max_len:
                                    head2partial[field] = replacement
                                   
                                    if match_len==max_len:
                                        head_cands.append(field)
                                    else:
                                        head = field
                                        head_cands = [head]
                                        max_len = match_len

                        if len(head_cands) >= 2:
                            head_cands.sort(key=lambda x: Q_ori.index(head2partial[x]) )
                            head = head_cands[0]
                           
                            


                    Q_head = head
                    #---------------------------------------------------------------------------------------------------
                    
                    new_candidates = [] 
                    i = 1
                    for field, value in candidates:
                        new_candidates.append(['<f'+str(i)+'>','<v'+str(i)+'>'])
                        i += 1
                    
                    #annonate Q(query)
                    Qpairs = []
                    for (f,v),(new_f,new_v) in zip(candidates,new_candidates):
                        Qpairs.append((f,new_f,'f'))
                        Qpairs.append((v,new_v,'v'))

                    Qpairs.sort( key=lambda x: 100-len(x[0]) )

                    ##===========================MATCH FIELD VAIRATION==================================
                    #partial match
                    if True:
                        p2partial = {}
                        p2partial = defaultdict(lambda:'unk', p2partial)

                        for f,v in candidates:
                            if f not in Q :
                                ids = _match_ids(f,Q)
                                replacement, match_len=_max_span(ids, Q.split()) 
                                if match_len >= _threshold(f) :
                                      
                                    if n==target:
                                        print(match_len)
                                        print(ids)
                                        print(replacement)


                                    p2partial[f] = replacement
                                else:
                                    # if v is found, and f is not seen in Q.
                                    if step2:
                                        Q = Q.replace(v, f+' '+v)
                                    else:
                                        pass
                                   
                                    #print('********')
                                    #print(Q_ori)
                                    #print(Q)


                    # field covered by value f = street, v = ryan street 
                    for f,v in candidates:
                        if ((f in v and Q.count(f)==1) or (p2partial[f] in v and Q.count(p2partial[f])==1)) and Q.count(v)==1:
                            Q = Q.replace(v,f+' '+v)

                    for p,new_p,t in Qpairs:     
                        cp = p.replace('\\','\\\\').replace('(', r'\(').replace(')', r'\)').replace('+', r'\+').replace('-', r'\-').replace('*', r'\*').replace('?', r'\?')
                        
                        if t=='f':   
                            if p not in Q:
                                p = p2partial[p]            
                            Q0 = re.sub('(\s|^)'+cp+'(\s|$|s|\?|,|.)',' '+new_p+' '+p+' <eof> ',Q)
                            Q = Q.replace(p,new_p+' '+p+' <eof> ') if Q==Q0 else Q0 
                        else:              
                            Q0 = re.sub('(\s|^)'+cp+'(\s|$|s|\?|,|.)',' '+new_p+' ',Q)
                            Q = Q.replace(p,new_p+' ') if Q==Q0 else Q0 
                       
                    f0 = Q_head
                    if len(f0)>=1:
                        if f0 not in Q:
                            f0 = head2partial[f0]

                        if f0 in Q:
                            Q = Q.replace(f0,'<f0> '+f0+' <eof>')
                        else:
                            
                            tokens = f0.split()
                            while tokens[0] in stop_words:
                                tokens = tokens[1:]
                            while tokens[-1] in stop_words:
                                tokens = tokens[:-1]
                            f0 = ' '.join(tokens)
                            Q = Q.replace(f0,'<f0> '+f0+' <eof>')
                            

                    
                    if ADD_FIELDS:
                        Q += ' <eos> ' 
                        for i,f in enumerate(all_fields):
                            Q += (' <c'+str(i)+'> '+f+' <eoc> ')

                    Q = _clean(Q)
                    Q = re.sub(r'(<f[0-9]>)(s)(\s|$)',r'\1\3', Q)
                    qu_file.write(Q+'\n')

                    validation_pairs = copy.copy(Qpairs)
                    validation_pairs.append((Q_head,'<f0>','head'))
                    for i,f in enumerate(all_fields):
                        validation_pairs.append((f,'<c'+str(i)+'>','c'))
                   
                    #######################################
                    # Annotate SQL
                    #
                    #######################################

                    q_sent = Query.from_dict(d['sql'])
                    S, col_names, val_names = q_sent.to_sentence(tables[d['table_id']]['header'],rows,tables[d['table_id']]['types'])
                    S_noparen = q_sent.to_sentence_noparenthesis(tables[d['table_id']]['header'],rows,tables[d['table_id']]['types'])
                    S_noparen = _preclean(S_noparen)
                    S = _preclean(S)
                    S_ori = S

                    
                    new_col_names = []
                    for col_name in col_names:
                        new_col_names.append(_preclean(col_name))
                    col_names = new_col_names

                    new_val_names = []
                    for val_name in val_names:
                        new_val_names.append(_preclean(val_name))
                    val_names = new_val_names

                    HEAD = col_names[-1]
                    S_head = _preclean(HEAD)

                    if n==target:
                        print(Q_ori)
                        print(Q)
                        print(S)
                        print(S_head)
                        print(Q_head)
                        print(candidates)
                        print(name_pairs)
                        print(p2partial)

                    #annotate for SQL
                    name_pairs = []
                    for col_name, val_name in zip(col_names, val_names):
                        if col_name == val_name: 
                            name_pairs.append([_preclean(col_name),'true'])
                        else:
                            name_pairs.append([_preclean(col_name),_preclean(val_name)])

                    name_pairs.sort( key=lambda x: x[1]  )

                    new_name_pairs = []
                    i = 1
                    for field, value in name_pairs:
                        new_name_pairs.append(['<f'+str(i)+'>','<v'+str(i)+'>'])
                        i += 1

                    if _match_field(name_pairs,candidates):
                        #true pairs
                        pairs = []
                        for (f,v),(new_f,new_v) in zip(name_pairs,new_name_pairs):

                            pairs.append((f,new_f,'f'))
                            pairs.append((v,new_v,'v'))


                        pairs.sort( key=lambda x: 100-len(x[0]) )
                        # true pairs
                        for p, new_p, t in pairs:

                            cp = p.replace('\\','\\\\').replace('(', r'\(').replace(')', r'\)').replace('+', r'\+').replace('-', r'\-').replace('*', r'\*').replace('?', r'\?')

                            if new_p in Q:
                                if t == 'v':
                                    S = S.replace(p + ' )',new_p+' )') 
                                if t == 'f':
                                    S=re.sub('\( '+cp+' (equal|less|greater)', '( '+new_p+r' \1', S)
                          

                    if  S_head == Q_head and '<f0>' in Q:
                        S = S.replace(S_head ,'<f0>')
                       

                    # denote unseen fields
                    
                    if ADD_FIELDS:
                        for i,f in enumerate(all_fields):
                            cf = f.replace('\\','\\\\').replace('(', r'\(').replace(')', r'\)').replace('+', r'\+').replace('-', r'\-').replace('*', r'\*').replace('?', r'\?').replace('$', r'\$').replace('[', r'\[').replace(']', r'\]')
                            S = re.sub('(\s|^)'+cf+'(\s|$|s)',' <c'+str(i)+'> ',S)
                
                    S = _clean(S)
                    S = re.sub(r'(<f[0-9]>)(s)(\s|$)',r'\1\3', S)
                    lon_file.write(S+'\n')


                    #--------------------------------------------------------------------------------
                    #################################################################################
                    ################################## VALIDATION ###################################
                    #################################################################################

                    recover_S = S
                    for word,sym,t in validation_pairs:
                        recover_S = recover_S.replace(sym,word)
                        sym_file.write(sym+'=>'+word+'<>')
                    sym_file.write('\n')

                    S_file.write(S_noparen+'\n')
                    if recover_S != S_ori:
                        print(S_ori)
                        print(S)
                        print(recover_S)

                    #--------------------------------------------------------------------------------
                    if _match_field(name_pairs,candidates):      
                        acc_pair += 1
                        

                    if Q_head == S_head:     
                        acc += 1
                        
                    if _match_field(name_pairs,candidates) and Q_head == S_head:
                        acc_all += 1

                
                    full_anno = True
                    for s in S.split():
                        if s[0]!='<' and s not in ['(',')','where','less','greater','equal','max','min','count','sum','avg','and','true']:
                            error += 1  
                            full_anno = False  
                            if False and not _match_field(name_pairs,candidates) and n%100==0:
                                print('----------')
                                '''
                                out.write('----------'+str(n)+'-------------'+'\n')
                                out.write(Q+'\n')
                                out.write(S+'\n')
                                out.write('(f,v) pairs: '+' ; '.join(f+','+v for f,v in candidates)+'\n')
                                out.write('True head:'+S_head+'\n')
                                out.write('Matched head:'+Q_head+'\n')
                                out.write('all fields: '+' ; '.join(all_fields)+'\n')
                                out.write(np.fromstring(rows,dtype=str))
                                out.write('\n')
                                '''
                                print(Q)
                                print(S)
                                print('head:')
                                print(head) 
                                print(head2partial)
                                print('true HEAD')
                                print(Q_head)
                                print('fields:')
                                print(candidates)  
                                print(p2partial)
                                print('true fields')
                                print(name_pairs)

                            break

                    if False and not (_match_field(name_pairs,candidates) and head == col_names[-1]):
                        print('--------'+str(n)+'-----------')
                        print(Q_ori)
                        print(Q)
                        print(S_ori)
                        print(S)
                        print('head:')
                        print(head) 
                        print(head2partial)
                        print('true HEAD')
                        print(Q_head)
                        print('fields:')
                        print(candidates)  
                        print(p2partial)
                        print('true fields')
                        print(name_pairs)
                    
                 
                    n += 1
                   
                print(n)
                print('fully snnotated:' + str(1-error*1.0/n))
                print('accurate all percent:'+str(acc_all*1.0/n))
                print('accurate head match percent:'+str(acc*1.0/n))
                print('accurate fields pair match percent:'+str(acc_pair*1.0/n))

                print(poss*1.0/n)

if __name__ == '__main__':
    main()
   
