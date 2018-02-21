'''
subset: basketball
generate [SQL structure (2)]
'''
import numpy as np
from copy import copy
from collections import defaultdict
import itertools
from tensorflow.python.platform import gfile
#------------------------------------------------------
#load field value of the table
all_dicts =  np.load('all_dicts.npy').item()
subset = 'basketball'
subset_dict = all_dicts[subset]
#------------------------------------------------------
aggregates = ['max', 'count', 'min', 'avg', 'sum']
cmps = ['nl', 'ng', 'greater', 'equal', 'less', 'neq']
cmp_strs = ['not less than', 'not more than', 'more than', 'equal', 'less than', 'not']

time_query_words = ['when','how long']
location_query_words = ['where','in which']
general_query_words = ['what', 'which']

time_fields = ['season', 'start_time', 'end_time', 'date', 'posting_date']
location_fields = ['']
#------------------------------------------------------
'''
Generate base template
return : dictionary { field : list of (Sc,Qc) tuples }
'''
def _iterate_all(smap, query_words, field, conditions, cmps):
    new_sents = []
    sents = []
    sqls = []
    
    for query_word in query_words:
        sent = query_word
        sql = field+' where '
    
        if query_word.startswith('how many'):
            sent+=(' '+field.split('_')[-1])    
        elif query_word.startswith('when'):
            pass
        elif query_word=='who':
            sent+=(' played')
        else:
            sent+=(' '+field)

        for cond in conditions:
            s1 = copy(sent)
            sql1 = copy(sql)

            values = subset_dict[cond]
            if cond=='player':
                for val in values:
                    s = copy(s1)
                    stmp = (' does '+val+' played')
                    s += stmp
                    new_sents.append( s )
                    sq = copy(sql1)
                    sqtmp = ('( '+cond+' equal '+val+' )')
                    sq += sqtmp
                    sqls.append(sq)

                    smap[cond].append([stmp,sqtmp])

            if cond=='team':
                for val in values:
                    s = copy(s1)
                    stmp = (' for '+val)
                    s += stmp
                    new_sents.append( s )
                    sq = copy(sql1)
                    sqtmp = ('( '+cond+' equal '+val+' )')
                    sq += sqtmp
                    sqls.append(sq)

                    smap[cond].append([stmp,sqtmp])

            if cond=='season':
                for val in values:
                    s = copy(s1)
                    stmp = (' in '+val)
                    s += stmp
                    new_sents.append( s )
                    sq = copy(sql1)
                    sqtmp = ('( '+cond+' equal '+val+' )')
                    sq += sqtmp
                    sqls.append(sq)

                    smap[cond].append([stmp,sqtmp])

            if cond=='position':
                for val in values:
                    s = copy(s1)
                    stmp = (' as '+val)
                    s += stmp
                    new_sents.append( s )
                    sq = copy(sql1)
                    sqtmp = ('( '+cond+' equal '+val+' )')
                    sq += sqtmp
                    sqls.append(sq)

                    smap[cond].append([stmp,sqtmp])

            if cond.startswith('number'):
                for val in values:
                    for cmp_op,comp in zip(cmps,cmp_strs):
                        if cmp_op == 'neq':
                            continue
                        s = copy(s1)
                        stmp = (' '+comp+' '+val+' '+cond.split('_')[-1])
                        stmp = stmp.replace(' equal','')
                        s += stmp
                        new_sents.append( s )
                        sq = copy(sql1)
                        sqtmp = ('( '+cond+' '+cmp_op+' '+val+' )')
                        sq += sqtmp
                        sqls.append(sq)

                        smap[cond].append([stmp,sqtmp])

    return smap
#--------------------------------------------------------------------
'''
combine two sub-clauses
[SQL structure (2)]
'''
def _iterate_smap(smap, head_str, conds, switch=False):
    '''
    Example:
    conds = [ ['player','number_of_turnovers'],['season','player'],['player','team']]
    '''
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
            s_tmp = s1[0]+s2[0]
            sql_tmp ='( ' + s1[1]+' and '+s2[1] + ' )'
            sents.append(head_str[0]+s_tmp)
            #print(head_str[0]+s_tmp)
            sqls.append(head_str[1]+sql_tmp)
            #print(head_str[1]+sql_tmp)

    return sents, sqls

#------------------------------------------------------------------
def build_smap():
    smap = defaultdict(list)
    conditions = copy(subset_dict.keys())
    return _iterate_all(smap,['what'], 'player', conditions, cmps)
#-----------------------------------------------------------------
#generate possible field combination of sub-conditions
def generate_2():
    nums_fields = ['number_of_assists','number_of_blocks','number_of_turnovers','number_of_points',
        'number_of_fouls','number_of_steals','number_of_played_games']
    smap = build_smap()
    all_fields = copy(subset_dict.keys())
    sents, sqls = [], []

    for head in subset_dict.keys():
        print(head)
        if head=='player':
            head_str = ('which player played', 'select '+head+' where ')
            fields = copy(all_fields)
            fields.remove(head)
            field_combinations = []
            field_combinations.extend( [ ('season',x) for x in nums_fields ] )
            field_combinations.extend( [ ('team',x) for x in nums_fields ] )
            field_combinations.extend( [ ('season','team'),('team','position') ] )  
            #field_combinations = list(itertools.combinations(fields, r=2))
            sents0, sqls0 = _iterate_smap(smap, head_str, field_combinations )
            sents.extend(sents0)
            sqls.extend(sqls0)

        if head=='season':
            head_str = ('in which season','select '+head+' where ')
            fields = copy(all_fields)
            fields.remove(head)
            field_combinations = []
            field_combinations.extend( [ ('player',x) for x in nums_fields ] )
            field_combinations.extend( [ ('team',x) for x in nums_fields ] )
            field_combinations.extend( [ ('player','team') ] )  
            #field_combinations = list(itertools.combinations(fields, r=2))
            sents0, sqls0 = _iterate_smap(smap, head_str, field_combinations )
            sents.extend(sents0)
            sqls.extend(sqls0)

        if head=='team':
            head_str = ('which team','select '+head+' where ')
            fields = copy(all_fields)
            fields.remove(head)
            field_combinations = []
            field_combinations.extend( [ ('player',x) for x in nums_fields ] )
            field_combinations.extend( [ (x,'season') for x in nums_fields ] )
            field_combinations.extend( [ ('player','season') ] )    
            #field_combinations = list(itertools.combinations(fields, r=2))
            sents0, sqls0 = _iterate_smap(smap, head_str, field_combinations )
            sents.extend(sents0)
            sqls.extend(sqls0)

        if head=='position':
            head_str = ('which position','select '+head+' where ')
            fields = copy(all_fields)
            fields.remove(head)
            field_combinations = []
            field_combinations.extend( [ ('player','season') ] )    
            #field_combinations = list(itertools.combinations(fields, r=2))
            sents0, sqls0 = _iterate_smap(smap, head_str, field_combinations )
            sents.extend(sents0)
            sqls.extend(sqls0)

        if head.startswith('number'):
            head_str = ('how many '+head.split('_')[2],'select '+head+' where ')
            fields = copy(all_fields)
            fields.remove(head)
            field_combinations = []
            field_combinations.extend( [ ('player','season'), ('player','team')  ] )    
            #field_combinations = list(itertools.combinations(fields, r=2))
            sents0, sqls0 = _iterate_smap(smap, head_str, field_combinations )
            sents.extend(sents0)
            sqls.extend(sqls0)

    return sents, sqls


sents, sqls = generate_2()
with gfile.GFile('basketball_2.txt', mode='w') as out:
    for sent,sql in zip(sents,sqls):
        out.write(sent+'\n')
        out.write(sql+'\n')






