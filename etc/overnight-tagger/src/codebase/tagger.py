#!/usr/bin/env python2

import math
import os
import random
import sys
import time
import re
import inspect

import editdistance as ed
import numpy as np
import tag_utils as tu

from nltk.parse import stanford
from nltk import tree


def buildDictionary(schema):
    ''' Build a search dictionary based on a given schema
        tu.Config() contains a huge SCHEMA database, which contains all the fields in schema given
        return --- query_dict, mapping between words and possible field names they refer
                 string_dict, mapping between values and possible field names they refer
    '''
    config = tu.Config()
    query_dict = dict()
    string_dict = dict()
    # num_dict = dict()
    # for key in config.field2word:
    for key in schema:
        # build query_dict: for field names
        if key not in config.field2word:
            continue
        value = config.field2word[key]
        for query_word in value['query_word']:
            if query_word.lower() not in query_dict:
                # one value could potentially corresponds to several field names
                query_dict[query_word.lower()] = []
            query_dict[query_word.lower()].append(key)
        if value['value_type'] == 'string':
            # build string_dict: for field values
            for word in value['value_range']:
                if word.lower() not in string_dict:
                    string_dict[word.lower()] = []
                string_dict[word.lower()].append(key)
        # else:
        #     # build num_dict
    return query_dict, string_dict  #, num_dict


def buildDictionary880(schema):
    ''' Build a search dictionary based on a given schema in GeoQuery880
        tu.Config() contains a huge SCHEMA database, which contains all the fields in schema given
        return --- query_dict, mapping between words and possible field names they refer
                 string_dict, mapping between values and possible field names they refer
    '''
    config = tu.Config()
    query_dict = dict()
    string_dict = dict()
    #num_dict = dict()
    #for key in config.field2word:
    for key in schema:
        # build query_dict: for field names
        if key not in config.geo880_dict:
            continue
        value = config.geo880_dict[key]
        for query_word in value['query_word']:
            if query_word.lower() not in query_dict:
                # one value could potentially corresponds to several field names
                query_dict[query_word.lower()] = []
            query_dict[query_word.lower()].append(key)
        if value['value_type'] == 'string':
            # build string_dict: for field values
            for word in value['value_range']:
                if word.lower() not in string_dict:
                    string_dict[word.lower()] = []
                string_dict[word.lower()].append(key)
        # else:
        #     # build num_dict
    return query_dict, string_dict  #, num_dict


def strSimilarity(word1, word2):
    ''' Measure the similarity based on Edit Distance
    ### Measure how similar word1 is with respect to word2
    '''
    diff = ed.eval(word1.lower(), word2.lower())  #search
    # lcs = LCS(word1, word2)   #search
    length = max(len(word1), len(word2))
    if diff >= length:
        similarity = 0.0
    else:
        similarity = 1.0 * (length - diff) / length
    return similarity


def strIsNum(s):
    '''verify if the word represent a numerical value
    '''
    if not isinstance(s, basestring):
        return 0
    ones = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine"
    }
    tens = {
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
        "ninety"
    }
    teens = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", \
            "seventeen", "eighteen", "nineteen"}
    levels = {"hundred", "thousand", "million", "billion", "trillion"}
    if s in ones:
        return True
    if s in tens:
        return True
    if s in teens:
        return True
    if s in levels:
        return True
    if s.isdigit():
        return True  #return int(s)
    # for float value
    try:
        x = float(s)
        return True
    except ValueError:
        return False
    return False


def basic_tokenizer(sentence):
    '''Very basic tokenizer: split the sentence into a list of tokens.
    '''
    words = []
    WORD_SPLIT = re.compile(b"([,!?\")(])")  # get rid of '.':;'
    for space_separated_fragment in sentence.strip().split():
        words.extend(WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def treeLCAdepth(deptree, value_position, field_position):
    ''' Measure the depth of Lowest Common Ancestor (LCA) in a tree
        arguments --- deptree: dependency tree of a certain parse_query
                      value_position (int): the index of value in original query
                      field_position (int): the index of field in original query
        call function nltk.tree.treeposition_spanning_leaves(start, end), return the sequence of common ancestors
        return --- length of common ancestor sequence, the depth of LCA
    '''
    if value_position > field_position:
        start = field_position
        end = value_position
    else:
        start = value_position
        end = field_position
    treespan = deptree.treeposition_spanning_leaves(start, end)
    return len(treespan)


def buildDictionaryON(field2word, schema):
    ''' Build a search dictionary based on a given schema
        tu.Config() contains a huge SCHEMA database, which contains all the fields in schema given
        return --- query_dict, mapping between words and possible field names they refer
                 string_dict, mapping between values and possible field names they refer
    '''
    #config = tu.Config()
    query_dict = dict()
    string_dict = dict()
    #num_dict = dict()
    #for key in config.field2word:
    for key in schema:
        # build query_dict: for field names
        if key not in field2word:
            continue
        value = field2word[key]
        for query_word in value['query_word']:
            if query_word.lower() not in query_dict:
                # one value could potentially corresponds to several field names
                query_dict[query_word.lower()] = []
            query_dict[query_word.lower()].append(key)
        if value['value_type'] in ['string', 'date', 'time']:
            # build string_dict: for field values
            for word in value['value_range']:
                if word.lower() not in string_dict:
                    string_dict[word.lower()] = []
                string_dict[word.lower()].append(key)
        # else:
        #     # build num_dict
    return query_dict, string_dict  #, num_dict


def rearrange(tag2, field_corr, value_corr):
    ''' tag2 contains the tagging information with tagging id
        purpose --- rearrang the id such that they appear in ascending sequence; 
                    correspondingly change field_corr and value_corr
    '''
    newid = 0
    mapping = dict()  # for rearraging
    re_mapping = []  # for corr files
    for i in range(len(tag2)):
        if tag2[i] == '<nan>' or tag2[i] == '<count>':
            continue
        original_id = int(tag2[i][-1])
        if original_id in mapping:
            continue
        else:
            mapping[original_id] = newid
            re_mapping.append(original_id)
            newid += 1
    newtag2 = ['' for x in tag2]
    newfield_corr = [
        field_corr[re_mapping[idx]] for idx in range(len(field_corr))
    ]
    newvalue_corr = [
        value_corr[re_mapping[idx]] for idx in range(len(value_corr))
    ]
    for i in range(len(tag2)):
        if tag2[i] == '<nan>' or tag2[i] == '<count>':
            newtag2[i] = tag2[i]
            continue
        original_id = int(tag2[i][-1])
        newtag2[i] = tag2[i][:-1] + str(mapping[original_id])
    return newtag2, newfield_corr, newvalue_corr


def sentTagging_treeON(parser, field2word, query, fields, logic=None):
    ''' Tag each word in a query with one of the three possible tokens:
          BASED ON Dependency Tree
          1. <nan>
          2. <field>:<type>:i
          3. <value>:<type>:j
          where i, j are the position according to the schema
        argument --- parser: Stanford Dependency parser using NLTK python interface, output a dependency tree 
                             of a parse_query
        return --- tag
                   field correspondence: a list of the corresponding field names in seuquence [Gold, Nation], 
                                        if <field: 1> is Gold and <field: 2> is Nation
                   value correspondence: a list of the corresponding field values in seuquence ['13', 'China'],
                                        corresponding to the field at the same position in field_corr
                   tagged query: how many <field>:0 the <field>:1 <field>:2 <value>:2 won
                   tagged logical form: where <field>:1 equal <value>:1, select <field>:2
                   .ficorr -- [Gold, Nation]
                   .vacorr -- [14, italy;japan]
        deprecate the previous thought:
                   .ficorr -- [Gold, Nation, Nation]
                   .vacorr -- [14, italy, japan]
    '''
    ### prepare query, schema, and initialize tag with <nan> ###
    query = query.lower()
    words = basic_tokenizer(query)
    #0528 newly added
    if logic is not None:
        schema = []
        tokens = logic.split()
        for i in range(len(tokens)):
            if tokens[i] not in fields.split():
                continue
            else:
                if tokens[i] not in schema:
                    schema.append(tokens[i])
    else:
        schema = fields.split()
    tag = ["<nan>" for x in words]

    #field2vecField, field2vecValue = fromWordtoVecList(schema)
    field_dict, value_dict = buildDictionaryON(field2word, schema)

    filter_words = [',','the','a','an','in','for','of','on','through','with','than','and',\
    'is','are','do','does','did','has','have','had','what','how','many','get','same','as'] #,'number'

    ### TAG WITH <field> & <value>
    for i in range(len(words)):
        # 0th pass eliminate non-sense words, label <num> and standby
        if words[i] in filter_words:
            continue
        if strIsNum(words[i]):
            if len(words[i]) == 4:
                # find Year like fields
                the_one = None
                for j in range(len(schema)):
                    if field2word[schema[j]]['value_type'] == 'date':
                        # find year_like field
                        the_one = schema[j]
                if the_one == None:
                    tag[i] = '<value>:<num>'
                else:
                    tag[i] = '<value>:' + the_one
            else:
                tag[i] = '<value>:<num>'

    # 1st pass exact match of field name
        if tag[i] is not "<nan>":
            continue
        if words[i] in field_dict:
            tag[i] = '<field>:'
            if len(field_dict[words[i]]) == 1:
                tag[i] += field_dict[words[i]][0]
            else:
                tag[i] += ';'.join(field_dict[words[i]])

    # 2nd pass exact match of field values (CURRENTLY assume one word can NOT be both value and name)
    # TO DO: later update to Bloom filter, for strings with high similarity to certain value
        if tag[i] is not "<nan>":
            continue
        if words[i] in value_dict:
            tag[i] = '<value>:'
            if len(value_dict[words[i]]) == 1:
                tag[i] += value_dict[words[i]][0]
            else:
                tag[i] += ';'.join(value_dict[words[i]])

    # 3rd pass find field name according to string similarity (field with numerical values)
        if tag[i] is not "<nan>":
            continue
        baseline = 0.9
        for j in range(len(schema)):
            if strSimilarity(words[i], schema[j].lower()) >= baseline:
                tag[i] = schema[j]
                baseline = strSimilarity(words[i], schema[j].lower())
        if tag[i] is not '<nan>':
            tag[i] = '<field>:' + tag[i]

    # 0706 newly added
    # 4th pass find field values for strings with high similarity to certain value
        if tag[i] is not "<nan>":
            continue
        baseline = 0.85
        temp = None
        for keyvalue in value_dict:  #go over the keys of value dict and find high similarity ones
            if strSimilarity(words[i], keyvalue.lower()) >= baseline:
                baseline = strSimilarity(words[i], keyvalue.lower())
                temp = keyvalue.lower()
        if temp is not None:
            words[i] = temp
            tag[i] = '<value>:'
            if len(value_dict[words[i]]) == 1:
                tag[i] += value_dict[words[i]][0]
            else:
                tag[i] += ';'.join(value_dict[words[i]])

    #tag_sentence = ' '.join(tag)
    tag2 = ["<nan>" for x in tag]
    field_corr = []
    value_corr = []
    # 0706 newly hidden
    # if logic is not None:
    #     field_corr = [x for x in schema]  #0528 newly added
    #     value_corr = ['<nan>' for x in schema]
    # else:
    #     field_corr = []
    #     value_corr = []
    num_field_position = []
    num_value_position = []
    str_field_position = []
    str_value_position = []
    # count = 0

    ### CORRESPOND <field> with <value>
    for i in range(len(tag2)):
        if tag[i] == "<nan>":
            continue
        reference = tag[i].split(':')
        if reference[0] == '<field>':
            print reference
            if reference[1] in field_corr:
                idx = field_corr.index(reference[1])
                tag2[i] = '<field>:' + str(idx)
            else:
                field_corr.append(reference[1])
                value_corr.append("<nan>")
                idx = len(field_corr) - 1
                tag2[i] = '<field>:' + str(idx)

            refers = reference[1].split(';')
            # 0716 newly revised
            if field2word[refers[0]]['value_type'] in [
                    'string', 'time'
            ]:  #,'date' exclude for its uniqueness
                if len(refers) > 1:  #multiple fields, did not add into
                    continue
                str_field_position.append((i, idx))
            else:
                num_field_position.append((i, idx))

    moderation = set()  #0526 newly added
    for i in range(len(tag2)):
        if tag[i] == "<nan>":
            continue
        reference = tag[i].split(':')
        if reference[0] == '<value>':
            print reference
            # check reference[1] == '<num>'
            if reference[1] in field_corr:
                if len(reference[1].split(';')) > 1:
                    # Overlapping value range for field with 'string' type (e.g. 1st_venue)
                    str_value_position.append(i)
                    continue
                idx = field_corr.index(reference[1])
                tag2[i] = '<value>:' + str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';' + words[i]

            else:
                if reference[1] == '<num>' or reference[1] == '<order>':
                    num_value_position.append(i)
                    continue
                elif len(reference[1].split(';')) > 1:
                    # Overlapping value range for field with 'string' type (e.g. 1st_venue)
                    str_value_position.append(i)
                    #0526 newly added
                    for refer in reference[1].split(';'):
                        moderation.add(refer)
                    continue
                field_corr.append(reference[1])
                value_corr.append(words[i])
                idx = len(field_corr) - 1
                tag2[i] = '<value>:' + str(idx)

    print num_field_position  #[(4, 1), (7, 2)]
    print num_value_position  #[9]
    print str_field_position  #[(4, 1), (7, 2)]
    print str_value_position  #[9]
    print moderation
    # (dependency tree LCA) build parse query for str_position and num_position
    if len(num_value_position) > 0:
        parsequery_num = [x for x in words]
        for (j, m) in num_field_position:
            parsequery_num[j] = '<field:' + str(m) + '>'
        #for i in num_value_position:
        #  parsequery_num[i] = '<value>'
        dependency_tree_num = parser.raw_parse_sents(
            ('Hello, My name is Melroy', ' '.join(parsequery_num)))
        # find corresponding field (dependency tree LCA)
        # only take dependency_tree[1]
        for i in num_value_position:
            # find largest ancestor depth for (value, field) pair in the tree
            idx = None
            largest_depth = 0
            for (j, m) in num_field_position:
                depth = treeLCAdepth(dependency_tree_num[1], i, j)
                if depth > largest_depth:
                    idx = m
                    largest_depth = depth
            if idx is not None:
                tag2[i] = '<value>:' + str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';' + words[i]
            # 0711 newly added, coping with count numbers
            else:
                if tag[i] == '<value>:<num>':
                    tag2[i] = '<count>'

    ### 0508 newly added, to deal with overlap range in string-type field
    if len(str_value_position) > 0:
        parsequery_str = [x for x in words]
        for (j, m) in str_field_position:
            parsequery_str[j] = '<field:' + str(m) + '>'
        #for i in str_value_position:
        #  parsequery_str[i] = '<value>'
        dependency_tree_str = parser.raw_parse_sents(
            ('Hello, My name is Melroy', ' '.join(parsequery_str)))
        for i in str_value_position:
            # find largest ancestor depth for (value, field) pair in the tree
            idx = None
            largest_depth = 0
            for (j, m) in str_field_position:
                if field_corr[m] not in moderation:  #0526 newly added
                    continue
                depth = treeLCAdepth(dependency_tree_str[1], i, j)
                if depth > largest_depth:
                    idx = m
                    largest_depth = depth
            if idx is not None:
                tag2[i] = '<value>:' + str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';' + words[i]
            else:  #0526 newly added
                #new correspondence found
                reference = tag[i].split(':')
                refers = reference[1].split(';')
                if refers[0] in field_corr:
                    idx = field_corr.index(refers[0])
                    if value_corr[idx] is "<nan>":
                        value_corr[idx] = words[i]
                    else:
                        value_corr[idx] += ';' + words[i]
                else:
                    field_corr.append(refers[0])
                    idx = len(field_corr) - 1
                    value_corr.append(words[i])
                tag2[i] = '<value>:' + str(idx)

    # 0713 newly added: final rearrange of sequence
    tag2, field_corr, value_corr = rearrange(tag2, field_corr, value_corr)
    field_corr_sentence = ' '.join(field_corr)
    value_corr_sentence = ' '.join(value_corr)
    tag2_sentence = ' '.join(tag2)

    newquery = [x for x in basic_tokenizer(query)]
    for i in range(len(tag2)):
        if tag2[i] == '<nan>':
            continue
        newquery[i] = tag2[i]
    newquery_sentence = ' '.join(newquery)

    # further change the logical forms to new_logical forms
    if logic is not None:
        #0528 newly added
        #tokens = logic.split()
        newlogic = ['<nan>' for x in tokens]
        for i in range(len(tokens)):
            if tokens[i] in field_corr:
                idx = field_corr.index(tokens[i])
                newlogic[i] = '<field>:' + str(idx)
                continue
            for idx in range(len(value_corr)):
                if tokens[i].lower() in value_corr[idx].split(';'):
                    newlogic[i] = '<value>:' + str(idx)
                    continue
            if newlogic[i] == '<nan>':
                newlogic[i] = tokens[i]
        # 0711 newly added
        if len(newlogic) == 10 and newlogic[3] == 'count':
            newlogic[8] = '<count>'
        newlogic_sentence = ' '.join(newlogic)
    else:
        newlogic_sentence = None
    return tag2_sentence, field_corr_sentence, value_corr_sentence, newquery_sentence, newlogic_sentence


def sentTagging_treeON3(parser, field2word, query, fields, logic=None):
    ''' Tag each word in a query with one of the three possible tokens:
          BASED ON Dependency Tree
          1. <nan>
          2. <field>:<type>:i
          3. <value>:<type>:j
          where i, j are the position according to the schema
        argument --- parser: Stanford Dependency parser using NLTK python interface, output a dependency tree 
                             of a parse_query
        return --- tag
                   field correspondence: a list of the corresponding field names in seuquence [Gold, Nation], 
                                        if <field: 1> is Gold and <field: 2> is Nation
                   value correspondence: a list of the corresponding field values in seuquence ['13', 'China'],
                                        corresponding to the field at the same position in field_corr
                   tagged query: how many <field>:0 the <field>:1 <field>:2 <value>:2 won
                   tagged logical form: where <field>:1 equal <value>:1, select <field>:2
                   .ficorr -- [Gold, Nation]
                   .vacorr -- [14, italy;japan]
        deprecate the previous thought:
                   .ficorr -- [Gold, Nation, Nation]
                   .vacorr -- [14, italy, japan]
    '''
    ### prepare query, schema, and initialize tag with <nan> ###
    query = query.lower()
    words = basic_tokenizer(query)
    #0528 newly added
    if logic is not None:
        schema = []
        tokens = logic.split()
        for i in range(len(tokens)):
            if tokens[i] not in fields.split():
                continue
            else:
                if tokens[i] not in schema:
                    schema.append(tokens[i])
    else:
        schema = fields.split()
    tag = ["<nan>" for x in words]

    #field2vecField, field2vecValue = fromWordtoVecList(schema)
    field_dict, value_dict = buildDictionaryON(field2word, schema)

    filter_words = [',','the','a','an','in','for','of','on','through','with','than','and',\
    'is','are','do','does','did','has','have','had','what','how','many','get','same','as'] #,'number'

    ### TAG WITH <field> & <value>
    for i in range(len(words)):
        # 0th pass eliminate non-sense words, label <num> and standby
        if words[i] in filter_words:
            continue
        if strIsNum(words[i]):
            if len(words[i]) == 4:
                # find Year like fields
                the_one = None
                for j in range(len(schema)):
                    if field2word[schema[j]]['value_type'] == 'date':
                        # find year_like field
                        the_one = schema[j]
                if the_one == None:
                    # adding type
                    tag[i] = '<value>:<num>'
                else:  # adding type
                    tag[i] = '<value>:' + the_one
            else:
                tag[i] = '<value>:<num>'
    # 1st pass exact match of field name
        if tag[i] is not "<nan>":
            continue
        if words[i] in field_dict:
            tag[i] = '<field>:'
            if len(field_dict[words[i]]) == 1:
                tag[i] += field_dict[words[i]][0]
            else:
                tag[i] += ';'.join(field_dict[words[i]])

    # 2nd pass exact match of field values (CURRENTLY assume one word can NOT be both value and name)
    # TO DO: later update to Bloom filter, for strings with high similarity to certain value
        if tag[i] is not "<nan>":
            continue
        if words[i] in value_dict:
            tag[i] = '<value>:'
            if len(value_dict[words[i]]) == 1:
                tag[i] += value_dict[words[i]][0]
            else:
                tag[i] += ';'.join(value_dict[words[i]])

    # 3rd pass find field name according to string similarity (field with numerical values)
        if tag[i] is not "<nan>":
            continue
        baseline = 0.75
        for j in range(len(schema)):
            if strSimilarity(words[i], schema[j].lower()) >= baseline:
                tag[i] = schema[j]
                baseline = strSimilarity(words[i], schema[j].lower())
        if tag[i] is not '<nan>':
            tag[i] = '<field>:' + tag[i]

    # 0706 newly added
    # 4th pass find field values for strings with high similarity to certain value
        if tag[i] is not "<nan>":
            continue
        baseline = 0.85
        temp = None
        for keyvalue in value_dict:  #go over the keys of value dict and find high similarity ones
            if strSimilarity(words[i], keyvalue.lower()) >= baseline:
                baseline = strSimilarity(words[i], keyvalue.lower())
                temp = keyvalue.lower()
        if temp is not None:
            words[i] = temp
            tag[i] = '<value>:'
            if len(value_dict[words[i]]) == 1:
                tag[i] += value_dict[words[i]][0]
            else:
                tag[i] += ';'.join(value_dict[words[i]])

    #tag_sentence = ' '.join(tag)
    tag2 = ["<nan>" for x in tag]
    field_corr = []
    value_corr = []
    # 0706 newly hidden
    # if logic is not None:
    #     field_corr = [x for x in schema]  #0528 newly added
    #     value_corr = ['<nan>' for x in schema]
    # else:
    #     field_corr = []
    #     value_corr = []
    num_field_position = []
    num_value_position = []
    str_field_position = []
    str_value_position = []
    # count = 0

    ### CORRESPOND <field> with <value>
    for i in range(len(tag2)):
        if tag[i] == "<nan>":
            continue
        reference = tag[i].split(':')
        if reference[0] == '<field>':
            print reference
            if reference[1] in field_corr:
                idx = field_corr.index(reference[1])
                tag2[i] = '<field>:' + str(idx)
            else:
                field_corr.append(reference[1])
                value_corr.append("<nan>")
                idx = len(field_corr) - 1
                tag2[i] = '<field>:' + str(idx)

            refers = reference[1].split(';')
            # 0716 newly revised
            if field2word[refers[0]]['value_type'] in [
                    'string', 'time'
            ]:  #,'date' exclude for its uniqueness
                if len(refers) > 1:  #multiple fields, did not add into
                    continue
                str_field_position.append((i, idx))
            else:
                num_field_position.append((i, idx))

    moderation = set()  #0526 newly added
    for i in range(len(tag2)):
        if tag[i] == "<nan>":
            continue
        reference = tag[i].split(':')
        if reference[0] == '<value>':
            print reference
            # check reference[1] == '<num>'
            if reference[1] in field_corr:
                if len(reference[1].split(';')) > 1:
                    # Overlapping value range for field with 'string' type (e.g. 1st_venue)
                    str_value_position.append(i)
                    continue
                idx = field_corr.index(reference[1])
                tag2[i] = '<value>:' + str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';' + words[i]

            else:
                if reference[1] == '<num>' or reference[1] == '<order>':
                    num_value_position.append(i)
                    continue
                elif len(reference[1].split(';')) > 1:
                    # Overlapping value range for field with 'string' type (e.g. 1st_venue)
                    str_value_position.append(i)
                    #0526 newly added
                    for refer in reference[1].split(';'):
                        moderation.add(refer)
                    continue
                field_corr.append(reference[1])
                value_corr.append(words[i])
                idx = len(field_corr) - 1
                tag2[i] = '<value>:' + str(idx)

    print 'Field position (num type): ', num_field_position  #[(4, 1), (7, 2)]
    print 'Value position (num type): ', num_value_position  #[9]
    print 'Field position (str type): ', str_field_position  #[(4, 1), (7, 2)]
    print 'Value position (str type): ', str_value_position  #[9]
    print 'Unresolved f-v pairs: ', moderation
    # (dependency tree LCA) build parse query for str_position and num_position
    if len(num_value_position) > 0:
        parsequery_num = [x for x in words]
        for (j, m) in num_field_position:
            parsequery_num[j] = '<field:' + str(m) + '>'
        #for i in num_value_position:
        #  parsequery_num[i] = '<value>'
        dependency_tree_num = parser.raw_parse_sents(
            ('Hello, My name is Melroy', ' '.join(parsequery_num)))
        # find corresponding field (dependency tree LCA)
        # only take dependency_tree[1]
        for i in num_value_position:
            # find largest ancestor depth for (value, field) pair in the tree
            idx = None
            largest_depth = 0
            for (j, m) in num_field_position:
                depth = treeLCAdepth(dependency_tree_num[1], i, j)
                if depth > largest_depth:
                    idx = m
                    largest_depth = depth
            if idx is not None:
                tag2[i] = '<value>:' + str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';' + words[i]
            # 0711 newly added, coping with count numbers
            else:
                if tag[i] == '<value>:<num>':
                    tag2[i] = '<count>'

    ### 0508 newly added, to deal with overlap range in string-type field
    if len(str_value_position) > 0:
        parsequery_str = [x for x in words]
        for (j, m) in str_field_position:
            parsequery_str[j] = '<field:' + str(m) + '>'
        #for i in str_value_position:
        #  parsequery_str[i] = '<value>'
        dependency_tree_str = parser.raw_parse_sents(
            ('Hello, My name is Melroy', ' '.join(parsequery_str)))
        for i in str_value_position:
            # find largest ancestor depth for (value, field) pair in the tree
            idx = None
            largest_depth = 0
            for (j, m) in str_field_position:
                if field_corr[m] not in moderation:  #0526 newly added
                    continue
                depth = treeLCAdepth(dependency_tree_str[1], i, j)
                if depth > largest_depth:
                    idx = m
                    largest_depth = depth
            if idx is not None:
                tag2[i] = '<value>:' + str(idx)
                if value_corr[idx] is "<nan>":
                    value_corr[idx] = words[i]
                else:
                    value_corr[idx] += ';' + words[i]
            else:  #0526 newly added
                #new correspondence found
                reference = tag[i].split(':')
                refers = reference[1].split(';')
                if refers[0] in field_corr:
                    idx = field_corr.index(refers[0])
                    if value_corr[idx] is "<nan>":
                        value_corr[idx] = words[i]
                    else:
                        value_corr[idx] += ';' + words[i]
                else:
                    field_corr.append(refers[0])
                    idx = len(field_corr) - 1
                    value_corr.append(words[i])
                tag2[i] = '<value>:' + str(idx)
    # 0711 newly added, expand semantic representation
    for i in range(len(tag2)):
        if tag2[i] == '<nan>' or tag2[i] == '<count>':
            continue
        reference = tag2[i].split(':')
        idx = reference[1]
        if strIsNum(idx):
            idx = int(idx)
            tag2[
                i] = reference[0] + ':' + field2word[field_corr[idx]]['value_type'] + ':' + reference[1]

    # 0713 newly added: final rearrange of sequence
    tag2, field_corr, value_corr = rearrange(tag2, field_corr, value_corr)
    field_corr_sentence = ' '.join(field_corr)
    value_corr_sentence = ' '.join(value_corr)
    tag2_sentence = ' '.join(tag2)

    newquery = [x for x in basic_tokenizer(query)]
    for i in range(len(tag2)):
        if tag2[i] == '<nan>':
            continue
        newquery[i] = tag2[i]
    newquery_sentence = ' '.join(newquery)

    # further change the logical forms to new_logical forms
    if logic is not None:
        # 0528 newly added
        # tokens = logic.split()
        newlogic = ['<nan>' for x in tokens]
        for i in range(len(tokens)):
            if tokens[i] in field_corr:
                idx = field_corr.index(tokens[i])
                newlogic[
                    i] = '<field>:' + field2word[field_corr[idx]]['value_type'] + ':' + str(
                        idx)
                continue
            for idx in range(len(value_corr)):
                if tokens[i].lower() in value_corr[idx].split(';'):
                    newlogic[
                        i] = '<value>:' + field2word[field_corr[idx]]['value_type'] + ':' + str(
                            idx)
                    continue
            if newlogic[i] == '<nan>':
                newlogic[i] = tokens[i]
        # 0711 newly added
        if len(newlogic) == 10 and newlogic[3] == 'count':
            newlogic[8] = '<count>'
        newlogic_sentence = ' '.join(newlogic)
    else:
        newlogic_sentence = None
    return tag2_sentence, field_corr_sentence, value_corr_sentence, newquery_sentence, newlogic_sentence


def templateToLogicalfrom(field_corr_sentence, value_corr_sentence,
                          newlogic_sentence):
    ''' given newlogical template with field_corr and value_corr
        output the declarative logical form
    '''

    logic = []
    newlogical = newlogic_sentence.split()
    field_corr = field_corr_sentence.split()
    value_corr = value_corr_sentence.split()
    # go over each token in newlogical and replace with corresponding field name
    checkcount = dict(
    )  # used for keep track field with multiple values appeared
    for i in range(len(newlogical)):
        reference = newlogical[i].split(':')
        if len(reference) == 1:
            logic.append(newlogical[i])
            continue
        if reference[0] == '<field>':
            print reference
            idx = int(reference[1])
            logic.append(field_corr[idx])
            if field_corr[idx] not in checkcount:
                checkcount[field_corr[idx]] = 0
            else:
                checkcount[field_corr[idx]] += 1
        else:
            print reference
            idx = int(reference[1])
            # check whether value_corr[idx] is single value
            value_choice = value_corr[idx].split(';')
            if len(value_choice) == 1:
                logic.append(value_corr[idx])
                continue
            pick = checkcount[field_corr[idx]]
            logic.append(value_choice[pick])
    #print logic
    logic_sentence = ' '.join(logic)
    return logic_sentence
