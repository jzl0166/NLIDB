import os,sys,inspect
import random
import numpy as np
import copy

import tagger as tg
import tag_utils as tu
from nltk.parse import stanford
from nltk import tree


lo2count = 'count ( <field>:0 )'
lo2avg = 'avg ( <field>:0 )'
lo4max_1 = 'max ( <field>:0 )'
# lo4min_1 = 'min ( <field>:0 )'
lo4max = '<field>:0 where ( <field>:1 equal max ( <field>:1 ) )'
lo4maxb = '<field>:1 where ( <field>:0 equal max ( <field>:0 ) )'
# lo4min = '<field>:0 where ( <field>:1 equal min ( <field>:1 ) )'

lo5maxcnt = '<field>:0 argmax ( count ( <field>:1 ) )'
# lo5mincnt = '<field>:0 argmin ( count ( <field>:1 ) )'

lo6selecteq = '<field>:0 where ( <field>:1 equal <value>:1 )'
lo6selecteqb = '<field>:1 where ( <field>:0 equal <value>:0 )'
lo6selectneq = '<field>:0 where ( <field>:1 neq <value>:1 )'
lo6selectl = '<field>:0 where ( <field>:1 less <value>:1 )'
lo6selectng = '<field>:0 where ( <field>:1 ng <value>:1 )'

lo7selectcnteq = '<field>:0 where ( count ( <field>:1 ) equal <count> )'
lo7selectcntl = '<field>:0 where ( count ( <field>:1 ) less <count> )'
lo7selectcntng = '<field>:0 where ( count ( <field>:1 ) ng <count> )'
lo7selectcntneq = '<field>:0 where ( count ( <field>:1 ) neq <count> )'

lo8between = '<field>:0 where ( <field>:1 between <value>:1 and <value>:1 )'

lo10select = '<field>:0 where ( ( <field>:1 equal <value>:1 ) and ( <field>:2 equal <value>:2 ) )'
lo10selector = '<field>:0 where ( ( <field>:1 equal <value>:1 ) or ( <field>:1 equal <value>:1 ) )'
lo10selectorb = '<field>:1 where ( ( <field>:0 equal <value>:0 ) or ( <field>:0 equal <value>:0 ) )'
lo10selector1 = '<field>:0 where ( ( <field>:0 equal <value>:0 ) or ( <field>:0 equal <value>:0 ) )'

lo10select102 = '<field>:1 where ( ( <field>:0 equal <value>:0 ) and ( <field>:2 equal <value>:2 ) )'
lo10select201 = '<field>:2 where ( ( <field>:0 equal <value>:0 ) and ( <field>:1 equal <value>:1 ) )'

lo11nestmulti = '<field>:0 where ( <field>:1 equal ( select ( <field>:2 where ( <field>:0 equal <value>:0 ) ) ) )'
lo11nestmultil = '<field>:0 where ( <field>:1 less ( select ( <field>:2 where ( <field>:0 equal <value>:0 ) ) ) )'
lo11nestmulting = '<field>:0 where ( <field>:1 ng ( select ( <field>:2 where ( <field>:0 equal <value>:0 ) ) ) )'
lo11nestmulti021 = '<field>:0 where ( <field>:2 equal ( select ( <field>:2 where ( <field>:1 equal <value>:1 ) ) ) )'
lo11nestmulti012 = '<field>:0 where ( <field>:1 equal ( select ( <field>:1 where ( <field>:2 equal <value>:2 ) ) ) )'

lo11nestfront = '<field>:0 where ( <field>:1 equal ( select ( <field>:1 where ( <field>:0 equal <value>:0 ) ) ) )'
lo11nestfrontl = '<field>:0 where ( <field>:1 less ( select ( <field>:1 where ( <field>:0 equal <value>:0 ) ) ) )'
lo11nestfrontng = '<field>:0 where ( <field>:1 ng ( select ( <field>:1 where ( <field>:0 equal <value>:0 ) ) ) )'
lo11nestfrontneq = '<field>:0 where ( <field>:1 ng ( select ( <field>:1 where ( <field>:0 equal <value>:0 ) ) ) )'
lo11nestneq = '<field>:0 where ( <field>:0 equal ( select ( <field>:0 where ( <field>:1 equal <value>:1 ) ) ) )'

lo11nestback = '<field>:1 where ( <field>:0 equal ( select ( <field>:0 where ( <field>:1 equal <value>:1 ) ) ) )'
lo11nestbackng = '<field>:1 where ( <field>:0 equal ( select ( <field>:0 where ( <field>:1 equal <value>:1 ) ) ) )'
lo11nestbackneq = '<field>:1 where ( <field>:0 neq ( select ( <field>:0 where ( <field>:1 equal <value>:1 ) ) ) )'


def isRepetitive(sequence):
    for element in sequence[:-1]:
        if element == sequence[-1]:
            return True
    return False


def generateFieldCombs(field_corr_dicts):
    ''' If only fields are recombinable'''
    list_of_seqs = []
    if len(field_corr_dicts) == 1:
        # base case:
        for key in field_corr_dicts[0].keys():
            list_of_seqs.append([key])
    else:
        # recursive case:
        former_seqs = generateFieldCombs(field_corr_dicts[:-1])
        for key in field_corr_dicts[-1].keys():
            for seq in former_seqs:
                newseq = [x for x in seq]
                newseq.append(key)
                # check new repetitive elements
                if not isRepetitive(newseq):
                    list_of_seqs.append(newseq)
    return list_of_seqs


def generateValueCombs(field_corr_dicts, field_combination, qu_value):
    ''' Both fields and values are recombinable
        arguments --- field_combination: the selected field combination, where the value are to be decided
    '''
    list_of_seqs = []
    if len(qu_value) == 1:
        # base case:
        _, idx = qu_value[0]  # check position of values
        for value in field_corr_dicts[idx][field_combination[idx]]:
            list_of_seqs.append([value])
    else:
        # recursive case:
        former_seqs = generateValueCombs(field_corr_dicts, field_combination, qu_value[:-1])
        _, idx = qu_value[-1]
        for value in field_corr_dicts[idx][field_combination[idx]]:
            for seq in former_seqs:
                newseq = [x for x in seq]
                newseq.append(value)
                # check new repetitive elements
                if not isRepetitive(newseq):
                    list_of_seqs.append(newseq)
    return list_of_seqs


def augment(field2word, quTemp, loTemp, field_corr, schema_aug):
    ''' Data augmentation from a pair of query template and logical template
        arguments --- field_corr: a list of value_types e.g. ['string','entity','int','bool','date'], each idx should 
                      correspond to the postion in the templates
                      schema_aug: (self) PLURALS HERE! several schemas that the template could augment to.
        return --- collections of queries, logics, and fields
    '''
    queryCollect, logicCollect, fieldCollect = [], [], []
    
    # Step 1: preparation
    print '* step: 1 *'
    print quTemp
    query = quTemp.split()
    logic = loTemp.split()
    qu_field = []  # positions of field in query
    qu_value = []  # positions of value in query
    lo_field = []  # positions of field in logic
    lo_value = []  # positions of value in logic
    for i in range(len(query)):
        reference = query[i].split(':')
        if len(reference) == 1:
            continue
        print reference
        idx = int(reference[1])
        if reference[0] == '<field>':
            qu_field.append((i, idx))
        elif reference[0] == '<value>':
            qu_value.append((i, idx))
    print qu_field, qu_value
    for i in range(len(logic)):
        reference = logic[i].split(':')
        if len(reference) == 1:
            continue
        print reference
        idx = int(reference[1])
        if reference[0] == '<field>':
            lo_field.append((i, idx))
        elif reference[0] == '<value>':
            lo_value.append((i, idx))
    print lo_field, lo_value
    
    # Step 2: augment to different schemas
    print '* step: 2 *'
    for j in range(len(schema_aug)):
        # Step 2.1: for each schema, build correspondence list of dictionarys: [{}, {}, {}]
        field_corr_dicts = []
        # print '=== %d schema ===' %j
        schema = schema_aug[j]
        # because there could be multiple same-type fields in one sentences, we go over field_corr
        for k in range(len(field_corr)):
            field_corr_dict = dict()
            for i in range(len(schema)):
                field = schema[i]
                #print field
                value_type = field2word[schema[i]]['value_type']
                if value_type == field_corr[k]:
                    if value_type == 'entity':
                        #field_corr_dict[field] = config.field2word[schema[i]]['value_range']
                        num_sample = 3
                        if len(field2word[schema[i]]['value_range']) < num_sample:
                            num_sample = len(field2word[schema[i]]['value_range'])
                        field_corr_dict[field] = random.sample(field2word[schema[i]]['value_range'], num_sample)
                    elif value_type == 'string':
                        #field_corr_dict[field] = config.field2word[schema[i]]['value_range']
                        num_sample = 3
                        if len(field2word[schema[i]]['value_range']) < num_sample:
                            num_sample = len(field2word[schema[i]]['value_range'])
                        field_corr_dict[field] = random.sample(field2word[schema[i]]['value_range'], num_sample)
                    elif value_type == 'int':
                        field_corr_dict[field] = random.sample(range(1, 10), 3) 
                    elif value_type == 'date':
                        field_corr_dict[field] = [2004, 2007, 2010]
                    elif value_type == 'time':
                        field_corr_dict[field] = ['10am', '3pm', '5pm', '1pm']
                    elif value_type == 'month':
                        field_corr_dict[field] = ['jan_2nd', 'jan_3rd', 'feb_3rd']
                    elif value_type == 'bool':
                        field_corr_dict[field] = []  #'true'
            field_corr_dicts.append(field_corr_dict)
        # print field_corr_dicts 
        # now the list of dicts [{str_field1:[], str_field2:[], ...}, {int_field1:[], int_field2:[], ...}]
        
        # Step 2.2: Regenerate sentence by filling into the place
        field_combinations = generateFieldCombs(field_corr_dicts)
        for field_combination in field_combinations:
            print field_combination
            newquery = [x for x in query]
            newlogic = [x for x in logic]
            # regenerate query, lower case or query_word
            for (posit, idx) in qu_field:
                field_info = field2word[field_combination[idx]]
                if len(field_info['query_word']) > 1:
                    if posit == 0 and 'who' in field_info['query_word']:
                        pick = 'who'
                    elif posit == 0 and 'when' in field_info['query_word']:
                        pick = 'when'
                    else:
                        pick = random.choice(field_info['query_word'])
                        while pick == 'who' or pick == 'when' or pick == 'city':
                            pick = random.choice(field_info['query_word'])
                    newquery[posit] = pick
                else:
                    newquery[posit] = field_combination[idx].lower()                
            # regenerate logic forms
            for (posit, idx) in lo_field:
                newlogic[posit] = field_combination[idx]
            if len(qu_value) > 0:
                value_combinations = generateValueCombs(field_corr_dicts, field_combination, qu_value)
                for value_combination in value_combinations:
                    morequery = [x for x in newquery]
                    morelogic = [x for x in newlogic]
                    for i in range(len(qu_value)):
                        morequery[qu_value[i][0]] = str(value_combination[i]).lower()
                    for i in range(len(qu_value)):
                        morelogic[lo_value[i][0]] = str(value_combination[i])
                    queryCollect.append(' '.join(morequery))
                    if isRepetitive(queryCollect):
                        del queryCollect[-1]
                        continue
                    logicCollect.append(' '.join(morelogic))
                    fieldCollect.append(' '.join(schema_aug[j]))
                    # newly added for true
                    logicCollect[-1] = logicCollect[-1].replace('<value>:1','true')
                    logicCollect[-1] = logicCollect[-1].replace('<value>:2','true')
                continue
            queryCollect.append(' '.join(newquery))
            # newly added for <count>
            fillin = random.sample(['2','3','two','three'], 1)[0]
            queryCollect[-1] = queryCollect[-1].replace('<count>',fillin)
            if isRepetitive(queryCollect):
                del queryCollect[-1]
                continue
            logicCollect.append(' '.join(newlogic))
            fieldCollect.append(' '.join(schema_aug[j]))
            logicCollect[-1] = logicCollect[-1].replace('<count>',fillin)
            # newly added for true
            logicCollect[-1] = logicCollect[-1].replace('<value>:1','true')
            logicCollect[-1] = logicCollect[-1].replace('<value>:2','true')
    return queryCollect, logicCollect, fieldCollect


def main(parser, field2word, field2word_tag, collect, logic, schema):
    ''' for certain logic form, we have lines from collect files
        return --- queryCollect, logicCollect, fieldCollect
    '''
    queryCollect, logicCollect, fieldCollect = [], [], []
    for query in collect:
        # for each line, we parse the query, schema
        if query == '':
            continue
        print '*** New query ***'
        print query
        #tagging using tag_util's dict
        tagged2, field_corr, value_corr, quTemp, _ = \
                tg.sentTagging_treeON(parser, field2word_tag, query, ' '.join(schema))
        #converting to field_type_corr
        field_corr_old = field_corr.split()
        field_corr_new = ['' for x in field_corr_old]
        for i in range(len(field_corr_old)):
            field_type = field2word[field_corr_old[i]]['value_type']
            field_corr_new[i] = field_type
        schema_aug = [schema]
        
        #augmenting
        queryOne, logicOne, fieldOne = augment(field2word, quTemp, logic, field_corr_new, schema_aug)
        #extending collections
        queryCollect.extend(queryOne)
        logicCollect.extend(logicOne)
        fieldCollect.extend(fieldOne)
    return queryCollect, logicCollect, fieldCollect


# from less to more, equal to neq, argmax to argmin
def expandDatasets(queryCollect, logicCollect, schemaCollect):
    newqueryCollect, newlogicCollect, newschemaCollect = [], [], []
    for i in range(len(queryCollect)):
        query, logic, schema = copy.copy(queryCollect[i]), copy.copy(logicCollect[i]), copy.copy(schemaCollect[i])
        ## ng, no more
        if logic.find(' ng ')!= -1 and query.find('no more')!= -1:
            sample = np.random.rand()
            if sample >= 0.75:
                # stay
                if np.random.rand() >= 0.5:
                    query = query.replace('no more', 'not more')
            elif sample >= 0.5 and sample < 0.75:
                # to less
                logic = logic.replace(' ng ', ' nl ')
                if np.random.rand() >= 0.5:
                    query = query.replace('no more', 'not less')
                else:
                    query = query.replace('no more', 'no less')
            elif sample >= 0.25 and sample < 0.5:
                # to less
                logic = logic.replace(' ng ', ' greater ')
                if np.random.rand() >= 0.5:
                    query = query.replace('no more', 'more')
                else:
                    query = query.replace('no more', 'higher')                
            else:
                # to less
                logic = logic.replace(' ng ', ' less ')
                if np.random.rand() >= 0.5:
                    query = query.replace('no more', 'less')
                else:
                    query = query.replace('no more', 'fewer')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        ## ng, or less
        if logic.find(' ng ')!= -1 and query.find('or less')!= -1:
            if np.random.rand() >= 0.5:
                query = query.replace('or less', 'or more')
                logic = logic.replace(' ng ', ' nl ')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        ## ng, or lower
        if logic.find(' ng ')!= -1 and query.find('or lower')!= -1:
            if np.random.rand() >= 0.5:
                query = query.replace('or lower', 'or higher')
                logic = logic.replace(' ng ', ' nl ')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        ## ng, or fewer
        if logic.find(' ng ')!= -1 and query.find('or fewer')!= -1:
            if np.random.rand() >= 0.5:
                query = query.replace('or fewer', 'or more')
                logic = logic.replace(' ng ', ' nl ')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        ## ng, or bigger
        if logic.find(' ng ')!= -1 and query.find('or bigger')!= -1:
            if np.random.rand() >= 0.5:
                query = query.replace('or bigger', 'or smaller')
                logic = logic.replace(' ng ', ' nl ')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        ## ng, maximum
        if logic.find(' ng ')!= -1 and query.find('maximum')!= -1:
            if np.random.rand() >= 0.5:
                query = query.replace('maximum', 'minimum')
                logic = logic.replace(' ng ', ' nl ')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        # less, less than
        if logic.find('less')!= -1 and query.find('less than')!= -1:
            sample = np.random.rand()
            if sample >= 0.75:
                # stay
                if np.random.rand() >= 0.5:
                    query = query.replace('less than', 'fewer than')
            elif sample >= 0.5 and sample < 0.75:
                # to less
                logic = logic.replace('less', 'greater')
                if np.random.rand() >= 0.5:
                    query = query.replace('less than', 'more than')
                else:
                    query = query.replace('less than', 'larger than')
            elif sample >= 0.25 and sample < 0.5:
                # to less
                logic = logic.replace('less', 'nl')
                if np.random.rand() >= 0.6:
                    query = query.replace('less than', 'no less than')
                elif np.random.rand() >= 0.3 and np.random.rand() < 0.6:
                    query = query.replace('less than', 'at least')
                else:
                    query = query.replace('less than', 'equal or more than')
            else:
                # to less
                logic = logic.replace('less', 'ng')
                if np.random.rand() >= 0.6:
                    query = query.replace('less than', 'no more than')
                elif np.random.rand() >= 0.3 and np.random.rand() < 0.6:
                    query = query.replace('less than', 'at most')
                else:
                    query = query.replace('less than', 'equal or less than')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        # less, lower
        if logic.find('less')!= -1 and query.find('lower')!= -1:
            if np.random.rand() >= 0.5:
                query = query.replace('lower', 'higher')
                logic = logic.replace('less', 'greater')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        # less, smaller
        if logic.find('less')!= -1 and query.find('smaller')!= -1:
            if np.random.rand() >= 0.5:
                query = query.replace('smaller', 'bigger')
                logic = logic.replace('less', 'greater')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        ### If in time-domain
        # Less
        if logic.find('less')!= -1 and query.find('before')!= -1:
            if np.random.rand() >= 0.5:
                query = query.replace('before', 'after')
                logic = logic.replace('less', 'greater')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        if logic.find('less')!= -1 and (query.find('earlier')!= -1 or query.find('shorter')!= -1 \
                                        or query.find('sooner')!= -1):
            if np.random.rand() >= 0.5:
                oppo = random.choice(['later','greater','longer'])
                query = query.replace('earlier', oppo)
                query = query.replace('shorter', oppo)
                query = query.replace('sooner', oppo)
                logic = logic.replace('less', 'greater')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        # Ng
        if logic.find(' ng ')!= -1 and (query.find('no later')!= -1 or query.find('no longer')!= -1):
            if np.random.rand() >= 0.5:
                query = query.replace('no later', 'no earlier')
                query = query.replace('no longer', 'no earlier')
                logic = logic.replace(' ng ', ' nl ')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        if logic.find(' ng ')!= -1 and (query.find('or earlier')!= -1 or query.find('or shorter')!= -1 \
                                        or query.find('or before')!= -1):
            if np.random.rand() >= 0.5:
                oppo = random.choice(['or later','or after','or longer'])
                query = query.replace('or earlier', oppo)
                query = query.replace('or shorter', oppo)
                query = query.replace('or before', oppo)
                logic = logic.replace(' ng ', ' nl ')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        if logic.find(' ng ')!= -1 and (query.find('at most')!= -1 or query.find('at latest')!= -1):
            if np.random.rand() >= 0.5:
                oppo = random.choice(['at least','at earliest'])
                query = query.replace('at most', oppo)
                query = query.replace('at latest', oppo)
                logic = logic.replace(' ng ', ' nl ')
            newqueryCollect.append(query)
            newlogicCollect.append(logic) 
            newschemaCollect.append(schema)
            continue
        # max, min
        if logic.find('max ')!= -1:
            newqueryCollect.append(query)
            newlogicCollect.append(logic)
            newschemaCollect.append(schema)
            
            if query.find('most') != -1:
                newqueryCollect.append(query.replace('most', 'least'))
                newlogicCollect.append(logic.replace('max ', 'min '))
                newschemaCollect.append(schema)
            
                if logic.find('count') != -1:
                    continue
                if np.random.rand() >= 0.5:
                    # maximum
                    newqueryCollect.append(query.replace('most', 'maximum'))
                    newlogicCollect.append(logic)
                    newschemaCollect.append(schema)
                else:
                    # highest
                    newqueryCollect.append(query.replace('most', 'highest'))
                    newlogicCollect.append(logic)
                    newschemaCollect.append(schema)
                
                if np.random.rand() >= 0.5:
                    # minimum
                    newqueryCollect.append(query.replace('most', 'minimum'))
                    newlogicCollect.append(logic.replace('max ', 'min '))
                    newschemaCollect.append(schema)
                else:
                    # smallest
                    newqueryCollect.append(query.replace('most', 'least'))
                    newlogicCollect.append(logic.replace('max ', 'min '))
                    newschemaCollect.append(schema)
            if query.find('latest') != -1:
                newqueryCollect.append(query.replace('latest', 'earliest'))
                newlogicCollect.append(logic.replace('max ', 'min '))
                newschemaCollect.append(schema)
            if query.find('longest') != -1:
                newqueryCollect.append(query.replace('longest', 'shortest'))
                newlogicCollect.append(logic.replace('max ', 'min '))
                newschemaCollect.append(schema)
            continue
        # non-equal to equal
        if logic.find('neq')!= -1 and query.find(' not ') != -1:
            # neq
            newqueryCollect.append(query)
            newlogicCollect.append(logic)
            newschemaCollect.append(schema)
            # equal
            newqueryCollect.append(query.replace(' not ', ' '))
            newlogicCollect.append(logic.replace('neq', 'equal'))
            newschemaCollect.append(schema)
            continue
        newqueryCollect.append(query)
        newlogicCollect.append(logic)
        newschemaCollect.append(schema)
    return newqueryCollect, newlogicCollect, newschemaCollect


def TD_Augmenting(TD, configdict):
    '''Provided certain information of TD, we can generate a test dataset
    '''
    parser = stanford.StanfordParser(model_path='/Users/richard_xiong/Documents/DeepLearningMaster/deep_parser/englishPCFG.ser.gz')
    
    queryCollect, logicCollect, schemaCollect = [], [], []
    for lo in TD['examples']:
        collect = TD['examples'][lo]
        queryTiny, logicTiny, schemaTiny = main(parser, TD['schema'], configdict, \
                                                 collect, lo, TD['schema'].keys())
        queryExpa, logicExpa, schemaExpa = expandDatasets(queryTiny, logicTiny, schemaTiny)
        queryCollect.extend(queryExpa)
        logicCollect.extend(logicExpa)
        schemaCollect.extend(schemaExpa)
    return queryCollect, logicCollect, schemaCollect