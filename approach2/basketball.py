import numpy as np
from copy import copy
from collections import defaultdict
import itertools

all_dicts =  np.load('all_dicts.npy').item()
#------------------------------------------------------
aggregates = ['max', 'count', 'min', 'avg', 'sum']
cmps = ['nl', 'ng', 'greater', 'equal', 'less', 'neq']
cmp_strs = ['not less than', 'not more than', 'more than', 'equal', 'less than', 'not']

'''
time_query_words = ['when','how long']
location_query_words = ['where','in which']
general_query_words = ['what', 'which']

time_fields = ['season', 'start_time', 'end_time', 'date', 'posting_date']
location_fields = ['']
'''

#------------------------------------------------------
subset = 'basketball'
subset_dict = all_dicts[subset]

def _build_pairs(smap, conditions, cmps):
    
    for cond in conditions:
        if cond=='player':
            smap[cond].append([(' does #v have '),('( #f equal #v )')])

        if cond=='team':
            smap[cond].append([(' for #v '),('( #f equal #v )')])

        if cond=='season':
            smap[cond].append([(' in #v '),('( #f equal #v )')])

        if cond=='position':
            smap[cond].append([(' as #v '),('( #f equal #v )')])

        if cond.startswith('number'):
            for cmp_op,comp in zip(cmps,cmp_strs):
                if cmp_op == 'neq':
                    smap[cond].append([(' not #v '+cond.split('_')[-1]),('( #f '+cmp_op+' #v )')])
                elif cmp_op == 'equal':
                    smap[cond].append([(' #v '+cond.split('_')[-1]),('( #f '+cmp_op+' #v )')])
                else:
                    smap[cond].append([(' '+comp+' #v '+cond.split('_')[-1]),('( #f '+cmp_op+' #v )')])
    return smap
#--------------------------------------------------------------------
'''
build smap that holds (sub-clause sentence, sub-clause sql) pairs
'''
def build_smap():
    smap = defaultdict(list)
    conditions = copy(subset_dict.keys())
    return _build_pairs(smap, conditions, cmps)
#------------------------------------------------------------------
'''
combine two sub-clauses
select HEAD where ( ( A [comparator] a ) [conjunction] ( B [comparator] b ) )
'''
def _build_2(smap, switch=False):
    s_head = 'which f0 has'
    sql_head = 'f0 where' 
    all_fields = copy(smap.keys())
    conds = itertools.permutations(all_fields,r=2)

    def _replace(s,i):
        return s.replace('#f','f'+str(i)).replace('#v','v'+str(i))

    sents = []
    sqls = []
    for c1,c2 in conds:
        list1 = smap[c1]
        list2 = smap[c2]
        if len(list2) > len(list1):
            tmp = list2
            list2 = list1
            list1 = tmp

        pairs = []
        for x in itertools.permutations(list1,len(list2)):
            pairs.extend(zip(x,list2))
        print(len(pairs))
        for s1,s2 in pairs: 

            s_tmp = _replace(s1[0],1)+_replace(s2[0],2)
            sql_tmp ='( ' + _replace(s1[1],1)+' and '+_replace(s2[1],2)+ ' )'
            sents.append(s_head+s_tmp)
            sqls.append(sql_head+sql_tmp)
            s_tmp = _replace(s1[0],0)+_replace(s2[0],1)
            sql_tmp ='( ' + _replace(s1[1],0)+' and '+_replace(s2[1],1)+ ' )'
            sents.append(s_head+s_tmp)
            sqls.append(sql_head+sql_tmp)
            s_tmp = _replace(s1[0],1)+_replace(s2[0],0)
            sql_tmp ='( ' + _replace(s1[1],1)+' and '+_replace(s2[1],0)+ ' )'
            sents.append(s_head+s_tmp)
            sqls.append(sql_head+sql_tmp)

        return sents,sqls
#-----------------------------------------------------------------
'''
TODO:
select HEAD where ( A [comparator] ( select ( A where ( B [comparator] b ) ) ) )
'''
def _build_3(smap, switch=False):
    s_head = 'which f0 has'
    sql_head = 'f0 where' 
    all_fields = copy(smap.keys())
    conds = itertools.permutations(all_fields,r=2)

    def _replace(s,i):
        return s.replace('#f','f'+str(i)).replace('#v','v'+str(i))

    sents = []
    sqls = []
    for c1,c2 in conds:
        list1 = smap[c1]
        list2 = smap[c2]
        if len(list2) > len(list1):
            tmp = list2
            list2 = list1
            list1 = tmp

        pairs = []
        for x in itertools.permutations(list1,len(list2)):
            pairs.extend(zip(x,list2))
        print(len(pairs))
        for s1,s2 in pairs: 

            s_tmp = _replace(s1[0],1)+_replace(s2[0],2)
            sql_tmp ='( ' + _replace(s1[1],1)+' and '+_replace(s2[1],2)+ ' )'
            sents.append(s_head+s_tmp)
            sqls.append(sql_head+sql_tmp)
            s_tmp = _replace(s1[0],0)+_replace(s2[0],1)
            sql_tmp ='( ' + _replace(s1[1],0)+' and '+_replace(s2[1],1)+ ' )'
            sents.append(s_head+s_tmp)
            sqls.append(sql_head+sql_tmp)
            s_tmp = _replace(s1[0],1)+_replace(s2[0],0)
            sql_tmp ='( ' + _replace(s1[1],1)+' and '+_replace(s2[1],0)+ ' )'
            sents.append(s_head+s_tmp)
            sqls.append(sql_head+sql_tmp)

        return sents,sqls
#-----------------------------------------------------------------
def build():
    smap = build_smap() #build base pairs
    sents, sqls = _build_2(smap)
    assert len(sents)==len(sqls)
    with open('basketball.txt', 'a') as output:
        for sent, sql in zip(sents,sqls):
            output.write(sent+'\n')
            output.write(sql+'\n')

build()




