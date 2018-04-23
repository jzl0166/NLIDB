import numpy as np
import random
import copy
from config_factory import *
from utils import *
from keras.preprocessing.sequence import pad_sequences
import cPickle
import sys

# seed = 1000
# np.random.seed(seed)

randint = np.random.randint
def rand_choice(l):
    if type(l) is not list: l = [l]
    return l[randint(len(l))]

def select(f1, f2, data):
    return [{'rid': rid, f1: row[f1], f2: row[f2]} for rid, row in enumerate(data)]


def where(f, op, val, data):
    if op == '>':
        return [r for r in data if r[f] > val]
    elif op == '<':
        return [r for r in data if r[f] < val]
    elif op == '=':
        return [r for r in data if r[f] == val]


def next_row(r1, table):
    if len(r1) == 0:
        return []
    r1 = r1[0]
    row_num = len(table)
    if r1['rid'] == row_num - 1:
        return []
    return [r for r in table if r['rid'] == r1['rid'] + 1]


def argmax_min(arg1, arg2, op, data):
    if op == 'max':
        try:
            val = max(data, key=lambda r: r[arg2])[arg2]
            rows = [r for r in data if r[arg2] == val]
        except:
            rows = []
        #row = max(data, key=lambda r: r[arg2])
        #return row['rid'], row[arg1]

        return [(r['rid'], r[arg1]) for r in rows]
    elif op == 'min':
        try:
            val = min(data, key=lambda r: r[arg2])[arg2]
            rows = [r for r in data if r[arg2] == val]
        except:
            rows = []

        return [(r['rid'], r[arg1]) for r in rows]
        #row = min(data, key=lambda r: r[arg2])
        #return row['rid'], row[arg1]


def max_min(arg, op, data):
    if op == 'max':
        try:
            val = max(data, key=lambda r: r[arg])[arg]
            rows = [r for r in data if r[arg] == val]
        except:
            rows = []
        #row = max(data, key=lambda r: r[arg2])
        #return row['rid'], row[arg1]

        return [(r['rid'], r[arg]) for r in rows]
    elif op == 'min':
        try:
            val = min(data, key=lambda r: r[arg])[arg]
            rows = [r for r in data if r[arg] == val]
        except:
            rows = []

        return [(r['rid'], r[arg]) for r in rows]
        #row = min(data, key=lambda r: r[arg2])
        #return row['rid'], row[arg1]


class ToyWorld(object):

    oov_fields = {'host_city', 'country'}

    def __init__(self):

        self.sample_range = 60  # TODO: increase size!!!
        self.world = {
            # world_size = 60
            'city': ['Amsterdam', 'Antwerp', 'Athens', 'Atlanta', 'Bangkok', 'Barcelona', 'Beijing', 'Berlin',
                          'Budapest', 'Buenos_Aires', 'Cairo', 'Cape_Town', 'Chamonix', 'Chicago', 'Cortina_Ampezzo',
                          'Dallas', 'Delhi', 'Dubai', 'Dublin', 'Florence', 'Grenoble', 'Helsinki', 'Hong_Kong',
                          'Innsbruck', 'Istanbul', 'Jerusalem', 'Lake_Placid', 'Las_Vegas', 'London', 'Los_Angeles',
                          'Madrid', 'Melbourne', 'Mexico_City', 'Milan', 'Montreal', 'Moscow', 'Mumbai', 'Munich',
                          'New_York', 'Oslo', 'Paris', 'Philadelphia', 'Prague', 'Rio_de_Janeiro', 'Rome', 'San_Diego',
                          'Sapporo', 'Sarajevo', 'Seattle', 'Seoul', 'Macau', 'Squaw_Valley', 'St._Moritz',
                          'Stockholm', 'Sydney', 'Tokyo', 'Toronto', 'Venice', 'Vienna', 'Warsaw'],
            'country': ['Brazil', 'Canada', 'Qatar', 'Italy', 'Peru', 'Kuwait', 'New_Zealand', 'Luxembourg', 'France',
                        'HK', 'Slovakia', 'Ireland', 'Nigeria', 'Norway', 'Argentina', 'South_Korea', 'Israel',
                        'Australia', 'Iran', 'Indonesia', 'West_Germany', 'Iceland', 'Slovenia', 'China', 'Chile',
                        'Belgium', 'Germany', 'Iraq', 'Philippines', 'Poland', 'Spain', 'Ukraine', 'Hungary',
                        'Netherlands', 'Denmark', 'Turkey', 'Finland', 'Sweden', 'Vietnam', 'Thailand', 'Switzerland',
                        'Russia', 'Pakistan', 'Romania', 'Portugal', 'Mexico', 'Egypt', 'Soviet_Union', 'Singapore',
                        'India', 'Liechtenstein', 'US', 'Czech', 'Austria', 'Yugoslavia', 'Saudi_Arabia', 'UK',
                        'Greece', 'Japan', 'Taiwan'],
            'continent': ['Americas', 'Asia', 'Europe', 'Oceania'],
            'gdp': range(10, 10 * self.sample_range + 10, 10),
            'population': range(10, 10 * self.sample_range + 10, 10),
            'size': range(10, 10 * self.sample_range + 10, 10),
        }

        self.depending_fields = {'country_population': 'country', 'country_size': 'country', 'country_gdp': 'country'}
        self.depended_fields = set(self.depending_fields[f] for f in self.depending_fields)

        world_toks = set()
        for f in self.world:
            for v in self.world[f]:
                world_toks.add(v)

        self.world_tok_count = len(world_toks)

        self.location_set = set(self.world['host_city']).union(set(self.world['country'])) #.union(set(self.world['continent']))

        ToyWorld.fixed_embed_words = set()
        # self.fixed_embed_words.update(i for i in self.world['host_city'])
        # ToyWorld.fixed_embed_words.update(['host_city', 'year', '#_participate', '#_medals', '#_duration'])

    def sample_table(self, row_num, fields=None):
        order_keep_fields = []  # ['year']
        sample_no_replace_fields = ['year'] # ['year', 'host_city']
        table_col_view = {}
        depended_fields_value_dict = {}

        if fields is None:
            fields = [f for f in self.world]

        for field in fields:
            values = np.random.choice(self.world[field],
                                      size=row_num,
                                      replace=False if field in sample_no_replace_fields else True)
            np.random.shuffle(values)
            if field in order_keep_fields:
                values = sorted(values)

            table_col_view[field] = values

        table = []
        for i in range(row_num):
            row = {f: table_col_view[f][i] for f in fields}

            if len(fields) == len(self.world):
                for f in self.depended_fields:
                    if f not in depended_fields_value_dict:
                        depended_fields_value_dict[f] = dict()

                    if row[f] not in depended_fields_value_dict[f]:
                        depended_fields_value_dict[f][row[f]] = dict()

                    for f_k, f_v in self.depending_fields.iteritems():
                        if f_v == f:
                            if f_k in depended_fields_value_dict[f][row[f]]:
                                row[f_k] = depended_fields_value_dict[f][row[f]][f_k]
                            else:
                                depended_fields_value_dict[f][row[f]][f_k] = row[f_k]

            table.append(row)

        return table

    def sample_table_with_oov(self, row_num, p_oov):
        table = self.sample_table(row_num)

        for row in table:
            for field in row:
                if field in ToyWorld.oov_fields:
                    is_new_word = np.random.choice([True, False], p=[p_oov, 1 - p_oov])

                    if is_new_word:
                        row[field] = row[field] + '_new'

        return table

    def get_utterance(self, token):
        if token == 'max':
            return 'maximum'
        elif token == 'min':
            return 'minimum'
        elif token == '<':
            return 'is smaller than'
        elif token == '>':
            return 'is larger than'
        elif token == '=':
            return 'is'
        else:
            token = str(token)

            # keep entity as it is
            if self.is_location(token): return token

            token = token.replace('_', ' ')
            token = token.replace('#', 'number of')

            return token

    def is_location(self, entity):
        return entity in self.location_set

    def get_prep_phrase(self, field, val=None, op='=', default_prep='with', style='POS'):
        assert op in {'=', '>', '<'}

        if val is None: val = ''
        val = str(val)
        utter = {}

        def comp_phrase(be=False, larger=False, count=True, than=True):
            if op == '=': _u = ''

            if op == '>':
                if larger: _u = 'larger'
                else: _u = 'more'
            if op == '<':
                if larger: _u = 'smaller'
                else: _u = 'fewer' if count else 'less'

            if than and op in {'<', '>'}: _u += ' than'

            if be: _u = 'is ' + _u

            return _u.strip()

        # default_prep = 'with' if allow_leading_with else 'which has'
        if type(default_prep) is list:
            default_prep = rand_choice(default_prep)

        if self.is_location(val):
            assert op == '='
            assert val

            _utter = 'in ' + self.get_utterance(val)
            utter = {'SPO': _utter, 'POS': _utter}
        elif field == 'year':
            if op == '=': prep = 'in'
            elif op == '>': prep = 'after'
            elif op == '<': prep = 'before'

            _utter = prep + ' ' + val
            utter = {'SPO': _utter, 'POS': _utter}
        elif field == '#_duration':
            prep = rand_choice(['that lasts for', 'lasting for'])

            _utter_SPO = _utter_POS = ' '.join([prep, comp_phrase(), val, 'days'])

            if not val:
                _utter_SPO = ' '.join([prep, comp_phrase(than=False), 'days than'])

            utter = {'SPO': _utter_SPO, 'POS': _utter_POS}
        elif field == '#_participants':
            _utter_POS = ' '.join([default_prep, comp_phrase(), val, 'participants'])
            _utter_SPO = ' '.join(['whose', self.get_utterance(field), comp_phrase(True, larger=True), val])
            if not val:
                _utter_SPO_2 = ' '.join(['which has', comp_phrase(than=False), 'participants than'])
                _utter_SPO_3 = ' '.join(['with', 'participants', comp_phrase(than=True)])
                _utter_SPO = [_utter_SPO, _utter_SPO_2, _utter_SPO_3]

            utter = {'SPO': _utter_SPO, 'POS': _utter_POS}
        elif field == '#_audience':
            _utter_POS = ' '.join([default_prep, comp_phrase(), val, 'audience'])
            _utter_SPO = ' '.join(['whose', self.get_utterance(field), comp_phrase(True, larger=True), val])
            if not val:
                _utter_SPO_2 = ' '.join(['which has', comp_phrase(than=False), 'audience than'])
                _utter_SPO_3 = ' '.join(['with', 'audience', comp_phrase(than=True)])
                _utter_SPO = [_utter_SPO, _utter_SPO_2, _utter_SPO_3]

            utter = {'SPO': _utter_SPO, 'POS': _utter_POS}
        elif field == '#_medals':
            _utter_POS = ' '.join([default_prep, comp_phrase(), val, 'medals'])
            _utter_SPO = ' '.join(['whose', self.get_utterance(field), comp_phrase(True, larger=True), val])
            if not val:
                _utter_SPO_2 = ' '.join(['which has', comp_phrase(than=False), 'medals than'])
                _utter_SPO_3 = ' '.join(['with', 'medals', comp_phrase(than=True)])
                _utter_SPO = [_utter_SPO, _utter_SPO_2, _utter_SPO_3]

            utter = {'SPO': _utter_SPO, 'POS': _utter_POS}
        elif field == 'country_population':
            _utter_POS = _utter_SPO = ' '.join(['whose host country population', comp_phrase(True, larger=True, count=False), val])
            if not val:
                _utter_SPO_2 = ' '.join(['which has', comp_phrase(than=False, count=False, larger=True), 'host country population than'])
                _utter_SPO_3 = ' '.join(['whose host country has population', comp_phrase(count=False, larger=True, than=True)])
                _utter_SPO = [_utter_SPO, _utter_SPO_2, _utter_SPO_3]

            utter = {'SPO': _utter_SPO, 'POS': _utter_POS}
        elif field == 'country_size':
            if op == '=':
                _utter_POS = _utter_SPO = ' '.join(['whose host country is of size', val])
            else:
                _utter_POS = _utter_SPO = ' '.join(['whose host country size is', comp_phrase(be=False, larger=True), val])

            if not val:
                _utter_SPO_2 = ' '.join(['which has', comp_phrase(than=False, count=False, larger=True), 'country size than'])
                _utter_SPO = [_utter_SPO, _utter_SPO_2]

            utter = {'SPO': _utter_SPO, 'POS': _utter_POS}
        elif field == 'country_gdp':
            _utter_POS = _utter_SPO = ' '.join(['whose host country gdp', comp_phrase(True, larger=True, count=False), val])
            if not val:
                if op == '>':
                    com_nl_phrase = 'wealthier'
                elif op == '<':
                    com_nl_phrase = 'less wealthy'

                _utter_SPO_2 = ' '.join(['whose host country is', com_nl_phrase, 'than'])
                _utter_SPO = [_utter_SPO, _utter_SPO_2]

            utter = {'SPO': _utter_SPO, 'POS': _utter_POS}

        style = rand_choice(style)
        utter = utter[style]

        return rand_choice(utter)

    def get_select_field_utterance(self, select_field):
        u1 = u2 = None
        if select_field == 'host_city':
            u1 = ('which city hosted the game', 'attr/prep')
            u2 = ('in which city was the game hosted', 'prep')
        elif select_field == 'country':
            u1 = ('which country hosted the game', 'attr/prep')
            u2 = ('in which country was the game hosted', 'prep')
        elif select_field == '#_participants':
            u1 = ('how many people participated in the game', 'attr/prep')
            u2 = ('how many participants are in the game', 'attr/prep')
        elif select_field == '#_audience':
            u1 = ('how many people watched the game', 'attr/prep')
            u2 = ('how many audience members are in the game', 'attr/prep')
        elif select_field == 'year':
            u1 = ('when was the game', 'attr/prep')
            u2 = ('when was the game hosted', 'prep')
        elif select_field == '#_medals':
            u1 = ('how many medals are in the game', 'attr/prep')
        elif select_field == 'country_size':
            u1 = ('what is the size of the host country of the game', 'attr/prep')
            u2 = ('how big is the country which hosted the game', 'attr/prep')
        elif select_field == 'country_population':
            u1 = ('what is the population of the host country of the game', 'attr/prep')
            u2 = ('how many people are in the host country of the game', 'attr/prep')
        elif select_field == '#_duration':
            u1 = ('how long is the game', 'attr/prep')
        elif select_field == 'continent':
            u1 = ('what is the continent of the host country of the game', 'attr/prep')
        elif select_field == 'country_gdp':
            # u1 = ('what is the gdp of the host country of the game', 'attr/prep')
            u1 = ('how wealthy is the host country of the game', 'attr/prep')
        else:
            raise RuntimeError('unknown select field: ' + select_field)

        if u2 is None:
            u = u1
        else: u = [u1, u2][np.random.randint(2)]

        return u[0], u[1].split('/')

    def get_superlative_utterance(self, field, polarity):
        utter = []
        utter_type = 'utter'

        assert polarity == 'max' or polarity == 'min', 'wrong polarity! %s' % polarity

        if polarity == 'max':
            if field == 'year':
                utter = ['last', 'latest']
                utter_type = 'adjective'
            elif field == '#_participants':
                utter = ['with the most participants']
            elif field == '#_medals':
                utter = ['with the most medals']
            elif field == '#_duration':
                utter = ['longest']
                utter_type = 'adjective'
            elif field == 'country_population':
                utter = ['with the largest host country population']
            elif field == 'country_size':
                utter = ['with the largest host country size', 'hosted by the largest country']
            elif field == 'country_gdp':
                # utter = ['with the largest gdp']
                utter = ['in the richest country']
            elif field == '#_audience':
                utter = ['with the most audience', 'watched by the most people']
            else: raise RuntimeError('unknown field! [%s]' % field)
        elif polarity == 'min':
            if field == 'year':
                utter = ['first', 'earliest']
                utter_type = 'adjective'
            elif field == '#_participants':
                utter = ['with the fewest participants']
            elif field == '#_medals':
                utter = ['with the fewest medals']
            elif field == '#_duration':
                utter = ['shortest']
                utter_type = 'adjective'
            elif field == 'country_population':
                utter = ['with the smallest host country population']
            elif field == 'country_size':
                utter = ['with the smallest host country size', 'hosted by the smallest country']
            elif field == 'country_gdp':
                # utter = ['with the smallest gdp']
                utter = ['in the poorest country']
            elif field == '#_audience':
                utter = ['with the fewest audience', 'watched by the fewest people']
            else: raise RuntimeError('unknown field! [%s]' % field)

        utter = rand_choice(utter)

        return utter, utter_type


def sample_table(rows):
    _tb_col_val = []
    for i in range(len(captions)):
        _tb_col_val.append([])

    for i in range(len(captions)):
        for j in range(len(rows)):
            _tb_col_val[i].append(rows[j][i])

        # for next query - keep the chronological order of year field!
        if captions[i] != 'year':
            np.random.shuffle(_tb_col_val[i])

    _rows = []
    for i in range(len(rows)):
        row = [_tb_col_val[j][i] for j in range(len(captions))]
        _rows.append(row)

    _table = []
    pos_caption_map = dict((i, field) for i, field in enumerate(captions))
    for row in _rows:
        r = dict((pos_caption_map[i], field) for i, field in enumerate(row))
        _table.append(r)

    return _table

utterence_pattern_multi_field_with_select = 'select {field1} , {field2} pair , where {comp_filed1} {comp1} {comp_val1} ' \
                                            'and {comp_filed2} {comp2} {comp_val2} , arg {max_min} {arg1} {arg2}'

utterance_pattern_multi_field = 'where {comp_field} {comp} {comp_val} ' \
                                ', arg {max_min} {arg1} {arg2}'

utterance_pattern_where_superlative_indeptfield = 'where {comp_field} {comp} {comp_val} ' \
                                                  ', arg {max_min} {arg1} {arg2}'

utterance_pattern_where_superlative_deptfield = 'where {comp_field} {comp} {comp_val} ' \
                                                ', arg {max_min} {arg1} {arg2}'


# utterance_pattern_where_superlative_deptfield = '{arg1} of the game whose {comp_field} {comp} {comp_val} ' \
#                                                 'and has the {max_min} {arg2}'


utterance_pattern_single_select = 'where {comp_field} = {comp_val} , ' \
                                  'select {select_field}'

utterance_multi_query = 'where {query1_comp_field} {query1_comp} {query1_comp_val} , ' \
                        'select {query1_project_field} as A , ' \
                        'where {query1_project_field} {query2_comp} A , ' \
                        'arg {max_min} {arg1} {arg2}'

###utterance_nested_query_4field_shortened = '{query1_comp_field} {query1_comp} {query1_comp_val} , ' \
###                                '{query1_project_field} {query2_comp} , ' \
###                                '{max_min} {arg1} {arg2}'

utterance_nested_query_4field = 'where {query1_comp_field} {query1_comp} {query1_comp_val} , ' \
                                          'select {query1_project_field} as A , ' \
                                          'where {query1_project_field} {query2_comp} A , ' \
                                          'arg {max_min} {arg1} {arg2}'

# utterance_nested_query_4field = 'where {query1_project_field} {query2_comp} A , ' \
#                                 'arg {max_min} {arg1} {arg2}'

utterance_next_query = 'where {query1_comp_field} {query1_comp} {query1_comp_val} , ' \
                       'next , ' \
                       'select {query2_project_field}'

utterance_superlative_query = 'arg {max_min} {arg1} {arg2}'


# ==============generate NL query examples==============

def post_process_nl_query(query):
    # when was the game hosted hosted by the largest country lasting for more than 380 days <eos>
    query = query.replace('hosted hosted', 'hosted')
    query = query.replace('  ', ' ')

    return query


def span_multiple_sets(test_set, sets):
    is_subset = True
    for s in sets:
        is_subset = is_subset and (not test_set.issubset(s))
    return is_subset


def generate_examples_single_select_nl_query(world, sample_size, sample_fields=None, query_cross_field_sets=None):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = 10

    while True:
        if sample_fields is None:
            fields = [f for f in world.world]
        else: fields = list(sample_fields)
        np.random.shuffle(fields)

        # sample a table
        table = world.sample_table(row_num, fields)

        field_pos_map = dict((field, i) for i, field in enumerate(fields))

        col_ids_attention = []

        # samble a query

        # step 1
        comp_field = np.random.choice(fields)

        # if not isint(table[0][comp_field]):
        #     continue

        comp_val = table[np.random.randint(row_num)][comp_field]

        col_ids_attention.append(field_pos_map[comp_field])

        # step 2
        select_field = np.random.choice(fields)

        if comp_field == select_field:
            continue

        col_ids_attention.append(field_pos_map[select_field])

        # query must span all field sets
        if query_cross_field_sets:
            if not span_multiple_sets({comp_field, select_field}, query_cross_field_sets):
                continue

        logical_form = utterance_pattern_single_select.format(comp_field=comp_field,
                                                           comp_val=comp_val,
                                                           select_field=select_field)

        # prefix, suffix = [('what is the', '?'), ('show me the', ''), ('tell me the', ''), ('give me the', ''), ('the', '')][np.random.randint(5)]

        # utterance = [prefix, world.get_utterance(select_field), 'of the game']

        sel_field_utterance, sel_clause_types = world.get_select_field_utterance(select_field)

        # sel_clause_type = np.random.choice(sel_clause_types)

        # def prep():
        #     predict_tokens = ['in' if world.is_location(comp_val) or comp_field == 'year' else 'with',
        #                       world.get_prep_phrase(comp_val, comp_field)]
        #     return ' '.join(predict_tokens)
        #
        # def attr():
        #     predict_tokens = ['whose', world.get_utterance(comp_field), 'is', world.get_utterance(comp_val)]
        #     return ' '.join(predict_tokens)

        # utterance = [sel_field_utterance, eval(sel_clause_type + '()')]

        # utterance.append(suffix)

        utterance = ' '.join([sel_field_utterance,
                              world.get_prep_phrase(comp_field, comp_val, default_prep=['with', 'that has'], style=['SPO', 'POS'])]).strip()
        utterance = post_process_nl_query(utterance)

        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        data = where(comp_field, '=', comp_val, data)

        if len(data) != 1:
            continue

        denotation = data[0]

        queries.add(utterance)

        total_inst_num += 1

        examples.append({'utterance': utterance,
                         'logical_form': logical_form,
                         'row_id': denotation['rid'],
                         'col_id': field_pos_map[select_field],
                         'denotation': denotation[select_field],
                         'caption_pos_map': field_pos_map,
                         'tokens': utterance.split(' '),
                         'table': table,
                         'meta': {'type': 'select_where', 'logical_form': logical_form},
                         'col_ids_attention': col_ids_attention})

        if len(examples) == sample_size:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


def generate_examples_superlative_nl_query(world, sample_size, sample_fields=None, query_cross_field_sets=None):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = 10

    while True:
        if sample_fields is None:
            fields = [f for f in world.world]
        else: fields = list(sample_fields)
        np.random.shuffle(fields)

        # sample a table
        table = world.sample_table(row_num, fields)

        field_pos_map = dict((field, i) for i, field in enumerate(fields))

        col_ids_attention = []

        # samble a query
        max_min = np.random.choice(['max', 'min'])
        arg1, arg2 = np.random.choice(fields, 2)

        col_ids_attention.append(field_pos_map[arg2])
        col_ids_attention.append(field_pos_map[arg1])

        if not isint(table[0][arg2]):
            continue

        # query must span all field sets
        if query_cross_field_sets:
            if not span_multiple_sets({arg1, arg2}, query_cross_field_sets):
                continue

        logical_form = utterance_superlative_query.format(max_min=max_min,
                                                       arg1=arg1,
                                                       arg2=arg2)

        sel_field_utterance, _ = world.get_select_field_utterance(arg1)
        superlative_clause, superlative_clause_type = world.get_superlative_utterance(arg2, max_min)

        # superlative_clause = ' '.join(['with', world.get_utterance(max_min), world.get_utterance(arg2)])

        if superlative_clause_type == 'adjective':
            utterance = sel_field_utterance.replace('the game', 'the ' + superlative_clause + ' game')
        else:
            utterance = ' '.join([sel_field_utterance, superlative_clause]).strip()

        utterance = post_process_nl_query(utterance)

        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        denotation = argmax_min(arg1, arg2, max_min, data)

        if len(denotation) != 1:
            continue

        queries.add(utterance)

        total_inst_num += 1

        # print utterance + '\t' + `denotation[0][1]`

        tokens = {arg1, arg2 + '_2ndarg'}

        examples.append({'utterance': utterance,
                         'logical_form': logical_form,
                         'row_id': denotation[0][0],
                         'col_id': field_pos_map[arg1],
                         'denotation': denotation[0][1],
                         'caption_pos_map': field_pos_map,
                         'tokens': utterance.split(' '),
                         'table': table,
                         'meta': {'type': 'superlative', 'logical_form': logical_form},
                         'col_ids_attention': col_ids_attention})

        if len(examples) == sample_size:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


def generate_examples_where_superlative_indeptfield_nl_query(world, sample_size, sample_fields=None, query_cross_field_sets=None):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = config.get('KB.row_num')

    while True:
        if sample_fields is None:
            fields = [f for f in world.world]
        else: fields = list(sample_fields)
        np.random.shuffle(fields)

        # sample a table
        table = world.sample_table(row_num, fields)

        field_pos_map = dict((field, i) for i, field in enumerate(fields))

        col_ids_attention = []

        # sample a query

        # step 1
        comp_field = np.random.choice(fields)

        if not isint(table[0][comp_field]):
            continue

        comp_val = table[np.random.randint(row_num)][comp_field]
        comp = np.random.choice(['<', '>'])

        col_ids_attention.append(field_pos_map[comp_field])

        # step 2 & 3
        max_min = np.random.choice(['max', 'min'])
        arg1, arg2 = np.random.choice(fields, 2)

        col_ids_attention.append(field_pos_map[arg2])
        col_ids_attention.append(field_pos_map[arg1])

        if not isint(table[0][arg2]):
            continue

        # remove signs of the same direction
        if (comp == '<' and max_min == 'min' or comp == '>' and max_min == 'max') and \
                        comp_field == arg2:
            continue

        # query must span all field sets
        if query_cross_field_sets:
            if not span_multiple_sets({arg1, arg2, comp_field}, query_cross_field_sets):
                continue

        logical_form = utterance_pattern_where_superlative_indeptfield.format(comp_field=comp_field,
                                                                           comp=comp,
                                                                           comp_val=comp_val,
                                                                           max_min=max_min,
                                                                           arg1=arg1,
                                                                           arg2=arg2)

        sel_field_utterance, _ = world.get_select_field_utterance(arg1)
        superlative_clause, superlative_clause_type = world.get_superlative_utterance(arg2, max_min)

        if superlative_clause_type == 'adjective':
            sel_field_utterance = sel_field_utterance.replace('the game', 'the ' + superlative_clause + ' game')
            superlative_clause = ''

        # superlative_clause = ' '.join(['with', world.get_utterance(max_min), world.get_utterance(arg2)])

        # def where_clause_prefix(comp_field, comp):
        #     if comp_field == 'year':
        #         if comp == '<': _prefix = 'before'
        #         if comp == '>': _prefix = 'after'
        #     else:
        #         _prefix = ' '.join(['whose', world.get_utterance(comp_field), world.get_utterance(comp)])
        #
        #     return _prefix

        # where_clause = where_clause_prefix(comp_field, comp) + ' ' + world.get_utterance(comp_val)

        where_clause = world.get_prep_phrase(comp_field, comp_val, op=comp,default_prep=['with', 'that has'], style=['SPO', 'POS'])

        utterance = ' '.join([sel_field_utterance, superlative_clause, where_clause]).strip()
        utterance = post_process_nl_query(utterance)

        # print utterance

        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        data1 = where(comp_field, comp, comp_val, data)

        if len(data1) < 2:
            continue

        denotation = argmax_min(arg1, arg2, max_min, data1)

        if len(denotation) != 1:
            continue

        # no easy example
        # denotation_short_cut = argmax_min(arg1, arg2, max_min, data)
        #
        # if denotation[0][1] == denotation_short_cut[0][1]:
        #     continue

        queries.add(utterance)

        total_inst_num += 1

        # print utterance + '\t' + `denotation[0][1]`

        # tokens = {comp_field, comp_val, arg1, arg2 + '_2ndarg'}

        examples.append({'utterance': utterance,
                         'logical_form': logical_form,
                         'row_id': denotation[0][0],
                         'col_id': field_pos_map[arg1],
                         'denotation': denotation[0][1],
                         'caption_pos_map': field_pos_map,
                         'tokens': utterance.split(' '),
                         'table': table,
                         'meta': {'type': 'where_superlative', 'logical_form': logical_form},
                         'col_ids_attention': col_ids_attention})

        if len(examples) == sample_size:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


def generate_examples_nested_query_4field_nl_query(world, sample_size):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = config.get('KB.row_num')

    while True:
        # sample a table
        table = world.sample_table(row_num)

        # shuffle fields
        fields = [f for f in world.world]
        np.random.shuffle(fields)
        field_pos_map = dict((field, i) for i, field in enumerate(fields))

        col_ids_attention = []

        # sample a query!

        # query1_comp_field, query1_project_field, arg1, arg2 = np.random.choice(fields, size=4, replace=False)

        # step 1
        query1_comp_field = np.random.choice(fields)
        query1_comp = '='
        query1_comp_val = table[np.random.randint(row_num)][query1_comp_field]

        col_ids_attention.append(field_pos_map[query1_comp_field])

        # step 2
        query1_project_field = np.random.choice(fields)

        if query1_comp_field == query1_project_field:
            continue

        if not isint(table[0][query1_project_field]):
            continue

        col_ids_attention.append(field_pos_map[query1_project_field])

        # step 3
        query2_comp = np.random.choice(['<', '>'])

        col_ids_attention.append(field_pos_map[query1_project_field])

        # step 4 5
        max_min = np.random.choice(['max', 'min'])
        arg1, arg2 = np.random.choice(fields, 2)

        if not isint(table[0][arg2]):
            continue

        col_ids_attention.append(field_pos_map[arg2])
        col_ids_attention.append(field_pos_map[arg1])

        # remove signs of the same direction
        if (query2_comp == '<' and max_min == 'min' or query2_comp == '>' and max_min == 'max') and \
                        query1_project_field == arg2:
            continue

        logical_form = utterance_nested_query_4field.format(query1_comp_field=query1_comp_field,
                                                         query1_comp=query1_comp,
                                                         query1_comp_val=query1_comp_val,
                                                         query1_project_field=query1_project_field,
                                                         query2_comp=query2_comp,
                                                         max_min=max_min,
                                                         arg1=arg1,
                                                         arg2=arg2)

        sel_field_utterance, _ = world.get_select_field_utterance(arg1)
        superlative_clause, superlative_clause_type = world.get_superlative_utterance(arg2, max_min)

        if superlative_clause_type == 'adjective':
            sel_field_utterance = sel_field_utterance.replace('the game', 'the ' + superlative_clause + ' game')
            superlative_clause = ''

        # superlative_clause = ' '.join(['with', world.get_utterance(max_min), world.get_utterance(arg2)])

        # def where_clause_prefix(comp_field, comp):
        #     if comp_field == 'year':
        #         if comp == '<': _prefix = 'before'
        #         if comp == '>': _prefix = 'after'
        #     else:
        #         _prefix = ' '.join(['whose', world.get_utterance(comp_field), world.get_utterance(comp)])
        #
        #     return _prefix

        # where_clause = ' '.join([where_clause_prefix(query1_project_field, query2_comp), 'the game',
        #                          'in' if world.is_location(query1_comp_val) or query1_comp_field == 'year' else 'with',
        #                          world.get_prep_phrase(query1_comp_field, query1_comp_val)])

        comparative_clause = world.get_prep_phrase(query1_project_field, None, op=query2_comp, default_prep=['with', 'that has'], style='SPO')
        where_clause = world.get_prep_phrase(query1_comp_field, query1_comp_val, default_prep=['with'], style=['POS', 'SPO'])

        utterance = ' '.join([sel_field_utterance, superlative_clause, comparative_clause, 'the game', where_clause]).strip()
        utterance = post_process_nl_query(utterance)

        utterance_length = len(utterance.split())
        if utterance_length > 20:
            discard = np.random.choice([True, False], p=[0.8, 0.2])
            if discard:
                continue

        # print utterance

        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        data1 = where(query1_comp_field, query1_comp, query1_comp_val, data)

        if len(data1) != 1:
            continue

        query1_project_field_val = data1[0][query1_project_field]
        data1 = where(query1_project_field, query2_comp, query1_project_field_val, data)

        if len(data1) < 2:
            continue

        denotation = argmax_min(arg1, arg2, max_min, data1)

        if len(denotation) != 1:
            continue

        # no easy example
        # denotation_short_cut = argmax_min(arg1, arg2, max_min, data)
        #
        # if denotation[0][1] == denotation_short_cut[0][1]:
        #     continue

        queries.add(utterance)

        total_inst_num += 1

        # print utterance + '\t' + `denotation[0][1]`

        tokens = {query1_comp_field, arg1, arg2, denotation[0][1], query1_comp_val}

        examples.append({'utterance': utterance,
                         'logical_form': logical_form,
                         'row_id': denotation[0][0],
                         'col_id': field_pos_map[arg1],
                         'denotation': denotation[0][1],
                         'caption_pos_map': field_pos_map,
                         'tokens': utterance.split(' '),
                         'table': table,
                         'meta': {'type': 'nest_query', 'logical_form': logical_form},
                         'col_ids_attention': col_ids_attention})

        # if len(examples) == 222223:
        if len(examples) == sample_size:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


# ==============generate OOV examples==============

def generate_examples_single_select_oov(world, sample_size, row_num = 10, p_oov = 0.5):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples_train = []
    examples_test = []

    for examples, sample_table, sample_size in zip((examples_train, examples_test),
                                                   (world.sample_table, lambda r: world.sample_table_with_oov(r, p_oov)),
                                                   (sample_size * 2, sample_size * 3)):
        while True:
            # sample a table
            table = sample_table(row_num)
            fields = [f for f in world.world]
            np.random.shuffle(fields)
            field_pos_map = dict((field, i) for i, field in enumerate(fields))

            col_ids_attention = []

            # sample a query

            # step 1
            comp_field = np.random.choice(fields)

            comp_val = table[np.random.randint(row_num)][comp_field]

            col_ids_attention.append(field_pos_map[comp_field])

            # step 2
            select_field = np.random.choice(fields)

            if comp_field == select_field:
                continue

            col_ids_attention.append(field_pos_map[select_field])

            utterance = utterance_pattern_single_select.format(comp_field=comp_field,
                                                               comp_val=comp_val,
                                                               select_field=select_field)

            data = copy.deepcopy(table)
            for rid, row in enumerate(data):
                row['rid'] = rid

            data = where(comp_field, '=', comp_val, data)

            if len(data) != 1:
                continue

            denotation = data[0]

            # print utterance + '\t' + `denotation[0][1]`

            tokens = {comp_val, denotation[select_field]}

            examples.append({'utterance': utterance,
                             'row_id': denotation['rid'],
                             'col_id': field_pos_map[select_field],
                             'denotation': denotation[select_field],
                             'caption_pos_map': field_pos_map,
                             'tokens': tokens,
                             'table': table,
                             'col_ids_attention': col_ids_attention})

            if len(examples) == sample_size:
                break

    return examples_train, examples_test


def generate_examples_superlative_oov(world, sample_size, row_num = 10, p_oov = 0.5):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples_train = []
    examples_test = []

    for examples, sample_table, sample_size in zip((examples_train, examples_test),
                                                   (world.sample_table, lambda r: world.sample_table_with_oov(r, p_oov)),
                                                   (sample_size * 2, sample_size * 3)):

        while True:
            # sample a table
            table = sample_table(row_num)
            fields = [f for f in world.world]
            np.random.shuffle(fields)
            field_pos_map = dict((field, i) for i, field in enumerate(fields))

            col_ids_attention = []

            # samble a query
            max_min = np.random.choice(['max', 'min'])
            arg1, arg2 = np.random.choice(fields, 2)

            col_ids_attention.append(field_pos_map[arg2])
            col_ids_attention.append(field_pos_map[arg1])

            if not isint(table[0][arg2]):
                continue

            utterance = utterance_superlative_query.format(max_min=max_min,
                                                           arg1=arg1,
                                                           arg2=arg2)

            data = copy.deepcopy(table)
            for rid, row in enumerate(data):
                row['rid'] = rid

            denotation = argmax_min(arg1, arg2, max_min, data)

            if len(denotation) != 1:
                continue

            # print utterance + '\t' + `denotation[0][1]`

            tokens = {arg1, arg2 + '_2ndarg'}

            examples.append({'utterance': utterance,
                             'row_id': denotation[0][0],
                             'col_id': field_pos_map[arg1],
                             'denotation': denotation[0][1],
                             'caption_pos_map': field_pos_map,
                             'tokens': tokens,
                             'table': table,
                             'col_ids_attention': col_ids_attention})

            if len(examples) == sample_size:
                break

    return examples_train, examples_test


def generate_examples_where_superlative_indeptfield_oov(world, sample_size, row_num = 10, p_oov = 0.5):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples_train = []
    examples_test = []

    for examples, sample_table, sample_size in zip((examples_train, examples_test),
                                                   (world.sample_table, lambda r: world.sample_table_with_oov(r, p_oov)),
                                                   (sample_size * 2, sample_size * 3)):

        while True:
            # sample a table
            table = sample_table(row_num)
            fields = [f for f in world.world]
            np.random.shuffle(fields)
            field_pos_map = dict((field, i) for i, field in enumerate(fields))

            col_ids_attention = []

            # sample a query

            # step 1
            comp_field = np.random.choice(fields)

            if not isint(table[0][comp_field]):
                continue

            comp_val = table[np.random.randint(row_num)][comp_field]
            comp = np.random.choice(['<', '>'])

            col_ids_attention.append(field_pos_map[comp_field])

            # step 2 & 3
            max_min = np.random.choice(['max', 'min'])
            arg1, arg2 = np.random.choice(fields, 2)

            col_ids_attention.append(field_pos_map[arg2])
            col_ids_attention.append(field_pos_map[arg1])

            if not isint(table[0][arg2]):
                continue

            utterance = utterance_pattern_where_superlative_indeptfield.format(comp_field=comp_field,
                                                                               comp=comp,
                                                                               comp_val=comp_val,
                                                                               max_min=max_min,
                                                                               arg1=arg1,
                                                                               arg2=arg2)

            data = copy.deepcopy(table)
            for rid, row in enumerate(data):
                row['rid'] = rid

            data = where(comp_field, comp, comp_val, data)

            if len(data) < 2:
                continue

            denotation = argmax_min(arg1, arg2, max_min, data)

            if len(denotation) != 1:
                continue

            # print utterance + '\t' + `denotation[0][1]`

            tokens = {comp_field, comp_val, arg1, arg2 + '_2ndarg'}

            examples.append({'utterance': utterance,
                             'row_id': denotation[0][0],
                             'col_id': field_pos_map[arg1],
                             'denotation': denotation[0][1],
                             'caption_pos_map': field_pos_map,
                             'tokens': tokens,
                             'table': table,
                             'col_ids_attention': col_ids_attention})

            if len(examples) == sample_size:
                break

    return examples_train, examples_test


def generate_examples_nested_query_4field_oov(world, sample_size, row_num = 10, p_oov = 0.5):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples_train = []
    examples_test = []

    for examples, sample_table, sample_size in zip((examples_train, examples_test),
                                                   (world.sample_table, lambda r: world.sample_table_with_oov(r, p_oov)),
                                                   (sample_size * 2, sample_size * 3)):

        while True:
            # sample a table
            table = sample_table(row_num)

            # shuffle fields
            fields = [f for f in world.world]
            np.random.shuffle(fields)
            field_pos_map = dict((field, i) for i, field in enumerate(fields))

            col_ids_attention = []

            # sample a query!

            # query1_comp_field, query1_project_field, arg1, arg2 = np.random.choice(fields, size=4, replace=False)

            # step 1
            query1_comp_field = np.random.choice(fields)
            query1_comp = '='
            query1_comp_val = table[np.random.randint(row_num)][query1_comp_field]

            col_ids_attention.append(field_pos_map[query1_comp_field])

            # step 2
            query1_project_field = np.random.choice(fields)

            if not isint(table[0][query1_project_field]):
                continue

            col_ids_attention.append(field_pos_map[query1_project_field])

            # step 3
            query2_comp = np.random.choice(['<', '>'])

            col_ids_attention.append(field_pos_map[query1_project_field])

            # step 4 5
            max_min = np.random.choice(['max', 'min'])
            arg1, arg2 = np.random.choice(fields, 2)

            if not isint(table[0][arg2]):
                continue

            col_ids_attention.append(field_pos_map[arg2])
            col_ids_attention.append(field_pos_map[arg1])

            # remove signs of the same direction
            # if query2_comp == '<' and max_min == 'min' or query2_comp == '>' and max_min == 'max':
            #     continue

            utterance = utterance_nested_query_4field.format(query1_comp_field=query1_comp_field,
                                                             query1_comp=query1_comp,
                                                             query1_comp_val=query1_comp_val,
                                                             query1_project_field=query1_project_field,
                                                             query2_comp=query2_comp,
                                                             max_min=max_min,
                                                             arg1=arg1,
                                                             arg2=arg2)

            data = copy.deepcopy(table)
            for rid, row in enumerate(data):
                row['rid'] = rid

            data1 = where(query1_comp_field, query1_comp, query1_comp_val, data)

            if len(data1) != 1:
                continue

            query1_project_field_val = data1[0][query1_project_field]
            data1 = where(query1_project_field, query2_comp, query1_project_field_val, data)

            if len(data1) < 2:
                continue

            denotation = argmax_min(arg1, arg2, max_min, data1)

            denotation_easy = argmax_min(arg1, arg2, max_min, data)

            if len(denotation) != 1:
                continue

            # no easy example
            # if denotation[0][1] == denotation_easy[0][1]:
            #     continue

            # print utterance + '\t' + `denotation[0][1]`

            tokens = {query1_comp_field, arg1, arg2, denotation[0][1], query1_comp_val}

            examples.append({'utterance': utterance,
                             'row_id': denotation[0][0],
                             'col_id': field_pos_map[arg1],
                             'denotation': denotation[0][1],
                             'caption_pos_map': field_pos_map,
                             'tokens': tokens,
                             'table': table,
                             'col_ids_attention': col_ids_attention})

            if len(examples) == sample_size:
                break

            # print utterance

    return examples_train, examples_test


def generate_examples():
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    for fid1 in range(len(captions)):
        for fid2 in range(fid1 + 1, len(captions)):
            field1 = captions[fid1]
            field2 = captions[fid2]

            if field1 == field2:
                continue

            for comp_fields in [(field1, field1), (field2, field2), (field1, field2)]:
                comp_field1, comp_field2 = comp_fields

                if (not (type(table[0][comp_field1]) is int) or
                    not (type(table[0][comp_field2]) is int)):
                    continue

                for comps in [('<', '<'), ('<', '>'), ('>', '>'), ('>', '<')]:
                    if comp_field1 == comp_field2 and (not comps == ('>', '<')):
                        continue

                    comp1, comp2 = comps

                    for row_id1 in range(len(table)):
                        for row_id2 in range(len(table)):
                            comp_val1 = table[row_id1][comp_field1]
                            comp_val2 = table[row_id2][comp_field2]

                            assert comp_val2 != 'Paris'

                            for max_min in ['max', 'min']:
                                for arg1, arg2 in ((a1, a2) for a1 in captions for a2 in captions if not a1 == a2):
                                    if not type(table[0][arg2]) is int:
                                        continue

                                    for i in range(config['dataset.tbl_sample_size']):
                                        # permute the table, and captions
                                        _captions = copy.deepcopy(captions)
                                        _table = sample_table(rows)
                                        np.random.shuffle(_captions)

                                        _pos_caption_map = dict((i, field) for i, field in enumerate(_captions))
                                        _caption_pos_map = dict((field, i) for i, field in enumerate(_captions))

                                        # data = select(field1, field2, _table)
                                        data = copy.deepcopy(_table)
                                        for rid, row in enumerate(data):
                                            row['rid'] = rid

                                        data = where(comp_field1, comp1, comp_val1, data)
                                        data = where(comp_field2, comp2, comp_val2, data)

                                        if len(data) < 2:
                                            break

                                        denotation = argmax_min(arg1, arg2, max_min, data)

                                        if len(denotation) == 0:
                                            break
                                        elif len(denotation) > 1:
                                            if i == 0: multi_val_inst_num += 1
                                            break

                                        if i == 0:
                                            unique_query_size += 1

                                        total_inst_num += 1

                                        utterance = utterence_pattern_multi_field_with_select.format(field1=field1,
                                                                             field2=field2,
                                                                             comp_filed1=comp_field1,
                                                                             comp1=comp1,
                                                                             comp_val1=comp_val1,
                                                                             comp_filed2=comp_field2,
                                                                             comp2=comp2,
                                                                             comp_val2=comp_val2,
                                                                             max_min=max_min,
                                                                             arg1=arg1,
                                                                             arg2=arg2)
                                        # print utterance

                                        tokens = {field1, field2, comp_val1, comp_val2, arg1, arg2, denotation[0][1]}

                                        examples.append({'utterance': utterance,
                                                         'row_id': denotation[0][0],
                                                         'col_id': _caption_pos_map[arg1],
                                                         'denotation': denotation[0][1],
                                                         'caption_pos_map': _caption_pos_map,
                                                         'tokens': tokens,
                                                         'table': _table})

    print 'unique_query_size:', unique_query_size
    print 'total_inst_num:', total_inst_num
    print 'multi_val_inst_num:', multi_val_inst_num

    return examples


def generate_examples_2fields(world):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = 10

    while True:
        # sample a table
        table = world.sample_table(row_num)
        fields = [f for f in world.world]
        np.random.shuffle(fields)

        # samble a query
        field1, field2 = np.random.choice(fields, 2)
        if field1 == field2:
            continue


def generate_examples_new():
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    for comp_field in captions:
        if not type(table[0][comp_field]) is int:
            continue

        for comp in ['<', '>']:
            for row_id in range(len(table)):
                comp_val = table[row_id][comp_field]
                if comp == '<' and comp_val == table_val_range[comp_field][0]:
                    continue

                if comp == '>' and comp_val == table_val_range[comp_field][1]:
                    continue

                for max_min in ['max', 'min']:
                    for arg1, arg2 in ((a1, a2) for a1 in captions for a2 in captions if not a1 == a2):
                        if not type(table[0][arg2]) is int:
                            continue

                        utterance = utterance_pattern_multi_field.format(comp_field=comp_field,
                                                             comp=comp,
                                                             comp_val=comp_val,
                                                             max_min=max_min,
                                                             arg1=arg1,
                                                             arg2=arg2)

                        for i in range(config['dataset.tbl_sample_size']):
                            # permute the table, and captions
                            _captions = copy.deepcopy(captions)
                            # _table = copy.deepcopy(table)
                            _table = sample_table(rows)
                            np.random.shuffle(_captions)
                            # np.random.shuffle(_table)

                            _pos_caption_map = dict((i, field) for i, field in enumerate(_captions))
                            _caption_pos_map = dict((field, i) for i, field in enumerate(_captions))

                            # data = select(field1, field2, _table)
                            data = copy.deepcopy(_table)
                            for rid, row in enumerate(data):
                                row['rid'] = rid

                            data = where(comp_field, comp, comp_val, data)

                            if len(data) < 2:
                                break

                            denotation = argmax_min(arg1, arg2, max_min, data)

                            if len(denotation) == 0:
                                break
                            elif len(denotation) > 1:
                                if i == 0: multi_val_inst_num += 1
                                break

                            if i == 0:
                                unique_query_size += 1

                            total_inst_num += 1
                            print utterance

                            tokens = {comp_field, comp_val, arg1, arg2, denotation[0][1]}

                            examples.append({'utterance': utterance,
                                             'row_id': denotation[0][0],
                                             'col_id': _caption_pos_map[arg1],
                                             'denotation': denotation[0][1],
                                             'caption_pos_map': _caption_pos_map,
                                             'tokens': tokens,
                                             'table': _table})

    print 'unique_query_size:', unique_query_size
    print 'total_inst_num:', total_inst_num
    print 'multi_val_inst_num:', multi_val_inst_num

    return examples


def generate_examples_single_select_OOV_old(world, sample_size):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = 10

    while True:
        # sample a table
        table = world.sample_table(row_num)
        fields = [f for f in world.world]
        np.random.shuffle(fields)
        field_pos_map = dict((field, i) for i, field in enumerate(fields))

        col_ids_attention = []

        # samble a query

        # step 1
        comp_field = np.random.choice(fields)

        # if not isint(table[0][comp_field]):
        #     continue

        comp_val = table[np.random.randint(row_num)][comp_field]

        col_ids_attention.append(field_pos_map[comp_field])

        # step 2
        select_field = np.random.choice(fields)

        if comp_field == select_field:
            continue

        col_ids_attention.append(field_pos_map[select_field])

        utterance = utterance_pattern_single_select.format(comp_field=comp_field,
                                                           comp_val=comp_val,
                                                           select_field=select_field)


        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        data = where(comp_field, '=', comp_val, data)

        if len(data) != 1:
            continue

        denotation = data[0]

        queries.add(utterance)

        total_inst_num += 1

        # print utterance + '\t' + `denotation[0][1]`

        tokens = {comp_val, denotation[select_field]}

        examples.append({'utterance': utterance,
                         'row_id': denotation['rid'],
                         'col_id': field_pos_map[select_field],
                         'denotation': denotation[select_field],
                         'caption_pos_map': field_pos_map,
                         'tokens': tokens,
                         'table': table,
                         'col_ids_attention': col_ids_attention})

        if len(examples) == sample_size:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


def generate_examples_single_select(world, sample_size):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = 10

    while True:
        # sample a table
        table = world.sample_table(row_num)
        fields = [f for f in world.world]
        np.random.shuffle(fields)
        field_pos_map = dict((field, i) for i, field in enumerate(fields))

        col_ids_attention = []

        # samble a query

        # step 1
        comp_field = np.random.choice(fields)

        # if not isint(table[0][comp_field]):
        #     continue

        comp_val = table[np.random.randint(row_num)][comp_field]

        col_ids_attention.append(field_pos_map[comp_field])

        # step 2
        select_field = np.random.choice(fields)

        if comp_field == select_field:
            continue

        col_ids_attention.append(field_pos_map[select_field])

        utterance = utterance_pattern_single_select.format(comp_field=comp_field,
                                                           comp_val=comp_val,
                                                           select_field=select_field)

        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        data = where(comp_field, '=', comp_val, data)

        if len(data) != 1:
            continue

        denotation = data[0]

        queries.add(utterance)

        total_inst_num += 1

        # print utterance + '\t' + `denotation[0][1]`

        tokens = {comp_val, denotation[select_field]}

        examples.append({'utterance': utterance,
                         'row_id': denotation['rid'],
                         'col_id': field_pos_map[select_field],
                         'denotation': denotation[select_field],
                         'caption_pos_map': field_pos_map,
                         'tokens': tokens,
                         'table': table,
                         'col_ids_attention': col_ids_attention})

        if len(examples) == sample_size:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


def generate_examples_where_superlative_indeptfield(world, sample_size):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = config.get('KB.row_num')

    while True:
        # sample a table
        table = world.sample_table(row_num)
        fields = [f for f in world.world]
        np.random.shuffle(fields)
        field_pos_map = dict((field, i) for i, field in enumerate(fields))

        col_ids_attention = []

        # sample a query

        # step 1
        comp_field = np.random.choice(fields)

        if not isint(table[0][comp_field]):
            continue

        comp_val = table[np.random.randint(row_num)][comp_field]
        comp = np.random.choice(['<', '>'])

        col_ids_attention.append(field_pos_map[comp_field])

        # step 2 & 3
        max_min = np.random.choice(['max', 'min'])
        arg1, arg2 = np.random.choice(fields, 2)

        col_ids_attention.append(field_pos_map[arg2])
        col_ids_attention.append(field_pos_map[arg1])

        if not isint(table[0][arg2]):
            continue

        utterance = utterance_pattern_where_superlative_indeptfield.format(comp_field=comp_field,
                                                                           comp=comp,
                                                                           comp_val=comp_val,
                                                                           max_min=max_min,
                                                                           arg1=arg1,
                                                                           arg2=arg2)

        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        data = where(comp_field, comp, comp_val, data)

        if len(data) < 2:
            continue

        denotation = argmax_min(arg1, arg2, max_min, data)

        if len(denotation) != 1:
            continue

        queries.add(utterance)

        total_inst_num += 1

        # print utterance + '\t' + `denotation[0][1]`

        tokens = {comp_field, comp_val, arg1, arg2 + '_2ndarg'}

        examples.append({'utterance': utterance,
                         'row_id': denotation[0][0],
                         'col_id': field_pos_map[arg1],
                         'denotation': denotation[0][1],
                         'caption_pos_map': field_pos_map,
                         'tokens': tokens,
                         'table': table,
                         'col_ids_attention': col_ids_attention})

        if len(examples) == sample_size:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


def generate_examples_where_superlative_deptfield(world):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = 10

    while True:
        # sample a table
        table = world.sample_table(row_num)
        fields = [f for f in world.world]
        np.random.shuffle(fields)

        # samble a query
        arg1, arg2 = np.random.choice(fields, 2)

        comp_field = np.random.choice([arg1, arg2])

        if not isint(table[0][comp_field]):
            continue

        comp_val = table[np.random.randint(row_num)][comp_field]
        comp = np.random.choice(['<', '>'])

        max_min = np.random.choice(['max', 'min'])

        if not isint(table[0][arg2]):
            continue

        utterance = utterance_pattern_where_superlative_deptfield.format(comp_field=comp_field,
                                                                         comp=comp,
                                                                         comp_val=comp_val,
                                                                         max_min=max_min,
                                                                         arg1=arg1,
                                                                         arg2=arg2)

        field_pos_map = dict((field, i) for i, field in enumerate(fields))
        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        data1 = where(comp_field, comp, comp_val, data)

        if len(data1) < 2:
            continue

        denotation = argmax_min(arg1, arg2, max_min, data1)

        if len(denotation) != 1:
            continue

        # no superlative short cut!
        # denotation_shortcut = argmax_min(arg1, arg2, max_min, data)
        # if denotation_shortcut[0][1] == denotation[0][1]:
        #     continue

        queries.add(utterance)

        total_inst_num += 1

        # print utterance + '\t' + `denotation[0][1]`

        tokens = {comp_field, comp_val, arg1, arg2, denotation[0][1]}

        examples.append({'utterance': utterance,
                         'row_id': denotation[0][0],
                         'col_id': field_pos_map[arg1],
                         'denotation': denotation[0][1],
                         'caption_pos_map': field_pos_map,
                         'tokens': tokens,
                         'table': table})

        if len(examples) == 50000:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


def generate_examples_multi_query():
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []

    world = ToyWorld()

    for query1_comp_field in captions:
        for query1_comp in ['=']:
            for row_id in range(len(table)):
                query1_comp_val = table[row_id][query1_comp_field]

                for query1_project_field in captions:
                    if not type(table[0][query1_project_field]) is int:
                        continue
                    # query1_project_field_val = table[row_id]

                    for query2_comp in ['<', '>']:
                        for max_min in ['max', 'min']:
                            for arg1, arg2 in ((a1, a2) for a1 in captions for a2 in captions if not a1 == a2):
                                if not type(table[0][arg2]) is int:
                                    continue

                                utterance = utterance_multi_query.format(query1_comp_field=query1_comp_field,
                                                                         query1_comp=query1_comp,
                                                                         query1_comp_val=query1_comp_val,
                                                                         query1_project_field=query1_project_field,
                                                                         query2_comp=query2_comp,
                                                                         max_min=max_min,
                                                                         arg1=arg1,
                                                                         arg2=arg2)

                                for i in range(config['dataset.tbl_sample_size']):
                                    # permute the table, and captions
                                    _captions = copy.deepcopy([f for f in world.world])
                                    # _table = copy.deepcopy(table)
                                    # _table = sample_table(rows)
                                    _table = world.sample_table(10)
                                    np.random.shuffle(_captions)
                                    # np.random.shuffle(_table)

                                    _pos_caption_map = dict((i, field) for i, field in enumerate(_captions))
                                    _caption_pos_map = dict((field, i) for i, field in enumerate(_captions))

                                    # data = select(field1, field2, _table)
                                    data = copy.deepcopy(_table)
                                    for rid, row in enumerate(data):
                                        row['rid'] = rid

                                    data1 = where(query1_comp_field, query1_comp, query1_comp_val, data)

                                    if len(data1) != 1:
                                        continue

                                    query1_project_field_val = data1[0][query1_project_field]
                                    data = where(query1_project_field, query2_comp, query1_project_field_val, data)

                                    if len(data) < 2:
                                        continue

                                    denotation = argmax_min(arg1, arg2, max_min, data)

                                    if len(denotation) == 0:
                                        continue
                                    elif len(denotation) > 1:
                                        if i == 0: multi_val_inst_num += 1
                                        continue

                                    if i == 0:
                                        unique_query_size += 1

                                    total_inst_num += 1
                                    # print utterance

                                    tokens = {query1_comp_field, arg1, arg2, denotation[0][1]}

                                    examples.append({'utterance': utterance,
                                                     'row_id': denotation[0][0],
                                                     'col_id': _caption_pos_map[arg1],
                                                     'denotation': denotation[0][1],
                                                     'caption_pos_map': _caption_pos_map,
                                                     'tokens': tokens,
                                                     'table': _table})

    print 'unique_query_size:', unique_query_size
    print 'total_inst_num:', total_inst_num
    print 'multi_val_inst_num:', multi_val_inst_num

    return examples


def isint(x):
    t = issubclass(type(x), np.integer)
    if not t:
        t = type(x) is int

    return t


def generate_examples_multi_query(world):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = 10

    while True:
        # sample a table
        table = world.sample_table(row_num)
        fields = [f for f in world.world]
        np.random.shuffle(fields)
        # samble a query
        query1_comp = '='
        query1_comp_field = np.random.choice(fields)
        query1_comp_val = table[np.random.randint(row_num)][query1_comp_field]

        query1_project_field = np.random.choice(fields)

        if not isint(table[0][query1_project_field]):
            continue

        query2_comp = np.random.choice(['<', '>'])
        max_min = np.random.choice(['max', 'min'])
        arg1, arg2 = np.random.choice(fields, 2)

        if not isint(table[0][arg2]):
            continue

        utterance = utterance_multi_query.format(query1_comp_field=query1_comp_field,
                                                 query1_comp=query1_comp,
                                                 query1_comp_val=query1_comp_val,
                                                 query1_project_field=query1_project_field,
                                                 query2_comp=query2_comp,
                                                 max_min=max_min,
                                                 arg1=arg1,
                                                 arg2=arg2)

        field_pos_map = dict((field, i) for i, field in enumerate(fields))
        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        data1 = where(query1_comp_field, query1_comp, query1_comp_val, data)

        if len(data1) != 1:
            continue

        query1_project_field_val = data1[0][query1_project_field]
        data = where(query1_project_field, query2_comp, query1_project_field_val, data)

        if len(data) < 2:
            continue

        denotation = argmax_min(arg1, arg2, max_min, data)

        if len(denotation) != 1:
            continue

        queries.add(utterance)

        total_inst_num += 1

        # print utterance + '\t' + `denotation[0][1]`

        tokens = {query1_comp_field, arg1, arg2, denotation[0][1]}

        examples.append({'utterance': utterance,
                         'row_id': denotation[0][0],
                         'col_id': field_pos_map[arg1],
                         'denotation': denotation[0][1],
                         'caption_pos_map': field_pos_map,
                         'tokens': tokens,
                         'table': table})

        if len(examples) == 50000:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


def generate_examples_nested_query_4field_sql_like(world, sample_size):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = config.get('KB.row_num')

    while True:
        # sample a table
        table = world.sample_table(row_num)

        # shuffle fields
        fields = [f for f in world.world]
        np.random.shuffle(fields)
        field_pos_map = dict((field, i) for i, field in enumerate(fields))

        col_ids_attention = []

        # sample a query!

        # query1_comp_field, query1_project_field, arg1, arg2 = np.random.choice(fields, size=4, replace=False)

        # step 1
        query1_comp_field = np.random.choice(fields)
        query1_comp = '='
        query1_comp_val = table[np.random.randint(row_num)][query1_comp_field]

        col_ids_attention.append(field_pos_map[query1_comp_field])

        # step 2
        query1_project_field = np.random.choice(fields)

        if not isint(table[0][query1_project_field]):
            continue

        col_ids_attention.append(field_pos_map[query1_project_field])

        # step 3
        query2_comp = np.random.choice(['<', '>'])

        col_ids_attention.append(field_pos_map[query1_project_field])

        # step 4 5
        max_min = np.random.choice(['max', 'min'])
        arg1, arg2 = np.random.choice(fields, 2)

        if not isint(table[0][arg2]):
            continue

        col_ids_attention.append(field_pos_map[arg2])
        col_ids_attention.append(field_pos_map[arg1])

        # remove signs of the same direction
        # if query2_comp == '<' and max_min == 'min' or query2_comp == '>' and max_min == 'max':
        #     continue

        utterance = utterance_nested_query_4field.format(query1_comp_field=query1_comp_field,
                                                         query1_comp=query1_comp,
                                                         query1_comp_val=query1_comp_val,
                                                         query1_project_field=query1_project_field,
                                                         query2_comp=query2_comp,
                                                         max_min=max_min,
                                                         arg1=arg1,
                                                         arg2=arg2)

        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        data1 = where(query1_comp_field, query1_comp, query1_comp_val, data)

        if len(data1) != 1:
            continue

        query1_project_field_val = data1[0][query1_project_field]
        data1 = where(query1_project_field, query2_comp, query1_project_field_val, data)

        if len(data1) < 2:
            continue

        denotation = argmax_min(arg1, arg2, max_min, data1)

        denotation_easy = argmax_min(arg1, arg2, max_min, data)

        if len(denotation) != 1:
            continue

        # no easy example
        # if denotation[0][1] == denotation_easy[0][1]:
        #     continue

        queries.add(utterance)

        total_inst_num += 1

        # print utterance + '\t' + `denotation[0][1]`

        tokens = {query1_comp_field, arg1, arg2, denotation[0][1], query1_comp_val}

        examples.append({'utterance': utterance,
                         'row_id': denotation[0][0],
                         'col_id': field_pos_map[arg1],
                         'denotation': denotation[0][1],
                         'caption_pos_map': field_pos_map,
                         'tokens': tokens,
                         'table': table,
                         'col_ids_attention': col_ids_attention})

        # if len(examples) == 222223:
        if len(examples) == sample_size:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


def generate_examples_superlative(world, sample_size):
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []
    queries = set()

    row_num = 10

    while True:
        # sample a table
        table = world.sample_table(row_num)
        fields = [f for f in world.world]
        np.random.shuffle(fields)
        field_pos_map = dict((field, i) for i, field in enumerate(fields))

        col_ids_attention = []

        # samble a query
        max_min = np.random.choice(['max', 'min'])
        arg1, arg2 = np.random.choice(fields, 2)

        col_ids_attention.append(field_pos_map[arg2])
        col_ids_attention.append(field_pos_map[arg1])

        if not isint(table[0][arg2]):
            continue

        utterance = utterance_superlative_query.format(max_min=max_min,
                                                       arg1=arg1,
                                                       arg2=arg2)

        data = copy.deepcopy(table)
        for rid, row in enumerate(data):
            row['rid'] = rid

        denotation = argmax_min(arg1, arg2, max_min, data)

        if len(denotation) != 1:
            continue

        queries.add(utterance)

        total_inst_num += 1

        # print utterance + '\t' + `denotation[0][1]`

        tokens = {arg1, arg2 + '_2ndarg'}

        examples.append({'utterance': utterance,
                         'row_id': denotation[0][0],
                         'col_id': field_pos_map[arg1],
                         'denotation': denotation[0][1],
                         'caption_pos_map': field_pos_map,
                         'tokens': tokens,
                         'table': table,
                         'col_ids_attention': col_ids_attention})

        if len(examples) == sample_size:
            break

        # print utterance

    print 'unique_query_size:', len(queries)
    print 'total_inst_num:', total_inst_num

    return examples


def generate_examples_next_query():
    unique_query_size = 0
    multi_val_inst_num = 0
    total_inst_num = 0
    examples = []

    for query1_comp_field in captions:
        for query1_comp in ['=']:
            for row_id in range(len(table)):
                query1_comp_val = table[row_id][query1_comp_field]

                for query2_project_field in captions:

                    utterance = utterance_next_query.format(query1_comp_field=query1_comp_field,
                                                            query1_comp=query1_comp,
                                                            query1_comp_val=query1_comp_val,
                                                            query2_project_field=query2_project_field)

                    for index in range(config['dataset.tbl_sample_size']):
                        # permute the table, and captions
                        _captions = copy.deepcopy(captions)
                        _table = sample_table(rows)
                        np.random.shuffle(_captions)

                        _pos_caption_map = dict((i, field) for i, field in enumerate(_captions))
                        _caption_pos_map = dict((field, i) for i, field in enumerate(_captions))

                        # data = select(field1, field2, _table)
                        data = copy.deepcopy(_table)
                        for rid, row in enumerate(data):
                            row['rid'] = rid

                        data1 = where(query1_comp_field, query1_comp, query1_comp_val, data)

                        if len(data1) != 1:
                            continue

                        denotation = next_row(data1, data)

                        if len(denotation) != 1:
                            continue

                        if index == 0:
                            unique_query_size += 1

                        total_inst_num += 1
                        # print utterance

                        tokens = {query1_comp_field, query2_project_field, denotation[0][query2_project_field]}

                        examples.append({'utterance': utterance,
                                         'row_id': denotation[0]['rid'],
                                         'col_id': _caption_pos_map[query2_project_field],
                                         'denotation': denotation[0][query2_project_field],
                                         'caption_pos_map': _caption_pos_map,
                                         'tokens': tokens,
                                         'table': _table})

    print 'unique_query_size:', unique_query_size
    print 'total_inst_num:', total_inst_num
    print 'multi_val_inst_num:', multi_val_inst_num

    return examples


def get_token_idx_oov(token, dict, is_fixed_embed=False, override=False):
    if not override and token in dict:
        return dict[token]

    split_point = 400

    if is_fixed_embed:
        fixed_len = len([k for k in dict if dict[k] >= split_point])
        idx = split_point + fixed_len
    else:
        idx = len([k for k in dict if dict[k] < split_point])
        assert idx < split_point

    dict[token] = idx
    return idx


def get_token_idx(token, dict):
    if token in dict:
        return dict[token]

    idx = len(dict)

    dict[token] = idx
    return idx


def get_vocab_by_examples(examples):
    vocab = dict()
    world = ToyWorld()

    # zero-index token is reserved for padding!
    pad = get_token_idx('<pad>', vocab)

    if world:
        for field in world.world:
            for v in world.world[field]:
                get_token_idx(str(v), vocab)

    begin_idx = get_token_idx('<bos>', vocab)
    end_idx = get_token_idx('<eos>', vocab)

    for example in examples:
        for x in example['utterance'].split(): get_token_idx(x, vocab)

    return vocab


def to_sempre_dataset(file_path):
    dataset = deserialize_from_file(file_path)
    (meta_data_train, utterances_train, table_train, labels_train, col_ids_attention_train), (meta_data_test, utterances_test, table_test, labels_test, col_ids_attention_test), vocab_dict = dataset

    vocab_dict_inverse = dict((vocab_dict[tok], tok) for tok in vocab_dict)

    entry_tmp ='(example (id {eid}) (utterance "{utterance}") (context (graph tables.TableKnowledgeGraph {eid})) (targetValue (list (description "{answer}"))))'
    f_tb_store = open(file_path + '.sempre.table', 'w')
    f_query_type = open(file_path + '.query_type', 'w')

    for meta_data, utterances, tables, labels, col_ids_attention, suffix in [(meta_data_train, utterances_train, table_train, labels_train, col_ids_attention_train, 'train'),
                                                                             (meta_data_test, utterances_test, table_test, labels_test, col_ids_attention_test, 'test')]:
        exp_num = utterances.shape[0]

        f_dataset = open(file_path + '.sempre.dataset.' + suffix, 'w')

        for idx in range(exp_num):
            query_str, table, ref_ans, fields_order = interpret_example(utterances[idx], vocab_dict_inverse,
                                                                            tables[idx], labels[idx], ret_table_obj_fmt=True)

            query_str = query_str.replace('<bos> ', '').replace(' <eos>', '').strip()
            query_str += '?'

            eid = suffix + '-' + str(idx)
            entry_str = entry_tmp.format(eid=eid, utterance=query_str, answer=ref_ans)

            table_str = eid + '\t' + '|||'.join([','.join(fields_order)] + [','.join([r[f] for f in fields_order]) for r in table])

            f_dataset.write(entry_str + '\n')
            f_tb_store.write(table_str + '\n')

            f_query_type.write(eid + '\t' + meta_data[idx]['type'] + '\n')

        f_dataset.close()

    f_tb_store.close()



def generate_oov_dataset(dataset_path, p_oov=1.0):
    # the training part is kept unchanged!

    dataset = deserialize_from_file(dataset_path)
    (train_meta_data, utterances_train_indexed, table_train, labels_train, col_ids_attention_train), \
        (test_meta_data, utterances_test_indexed, table_test, labels_test, col_ids_attention_test), \
        vocab_dict = dataset

    world = ToyWorld()
    vocab_dict_inverse_old = dict((vocab_dict[tok], tok) for tok in vocab_dict)

    # augment the existing vocab with new oov entity
    for field in world.world:
        for v in world.world[field]:
            if field in ToyWorld.oov_fields:
                get_token_idx_oov(str(v), vocab_dict, is_fixed_embed=True, override=True)
                get_token_idx_oov(str(v) + '_new', vocab_dict, is_fixed_embed=True, override=True)

    print vocab_dict

    vocab_dict_inverse_new = dict((vocab_dict[tok], tok) for tok in vocab_dict)

    for tables, utterances, is_sample in [(table_train, utterances_train_indexed, False), (table_test, utterances_test_indexed, True)]:

        row_num = tables.shape[1]
        col_num = tables.shape[2]

        for id in range(tables.shape[0]):
            change_dict = dict()
            entity_words = set()
            for row_id in range(row_num):
                for col_id in range(col_num):
                    field = vocab_dict_inverse_old[tables[id][row_id][col_id][0]]
                    entity = vocab_dict_inverse_old[tables[id][row_id][col_id][1]]
                    if field in ToyWorld.oov_fields:
                        entity_words.add(entity)

            # sample!
            if is_sample:
                for entity in entity_words:
                    replace_new = np.random.choice([True, False], p=[p_oov, 1 - p_oov])
                    if replace_new: change_dict[entity] = entity + '_new'
                    else: change_dict[entity] = entity

            # replace the old vocabulary with new one
            for tid in range(utterances.shape[1]):
                word = vocab_dict_inverse_old[utterances[id][tid]]
                word_new_idx = vocab_dict[change_dict[word]] if word in change_dict else vocab_dict[word]
                utterances[id][tid] = word_new_idx

            for row_id in range(row_num):
                for col_id in range(col_num):
                    field = vocab_dict_inverse_old[tables[id][row_id][col_id][0]]
                    entity = vocab_dict_inverse_old[tables[id][row_id][col_id][1]]
                    word_new_idx = vocab_dict[change_dict[entity]] if entity in change_dict else vocab_dict[entity]
                    tables[id][row_id][col_id][0] = vocab_dict[field]
                    tables[id][row_id][col_id][1] = word_new_idx

    dataset_oov = (train_meta_data, utterances_train_indexed, table_train, labels_train, col_ids_attention_train), \
        (test_meta_data, utterances_test_indexed, table_test, labels_test, col_ids_attention_test), \
        vocab_dict

    oov_dataset_path = dataset_path + '.oov' + str(p_oov)
    save_dataset(dataset_oov, oov_dataset_path)
    interpret_dataset(oov_dataset_path)
    print 'done!'


def generate_dataset_fixed_size(train_examples, test_examples):
    np.random.shuffle(train_examples)
    np.random.shuffle(test_examples)

    word_dict = get_vocab_by_examples(train_examples + test_examples)
    begin_idx = word_dict['<bos>']
    end_idx = word_dict['<eos>']

    train_tokens = {x for e in train_examples for x in e['tokens']}
    test_tokens = {x for e in test_examples for x in e['tokens']}

    for x in test_tokens:
        if x not in train_tokens:
            raise RuntimeError('[%s] in test tokens is not in train tokens!' % x)

    for name, examples in (('train', train_examples), ('test', test_examples)):
        # index table!
        col_num = len(examples[0]['table'][0])
        table = []
        for example in examples:
            cur_table_idx = [[[get_token_idx(str(col_name), word_dict), get_token_idx(str(row[col_name]), word_dict)]
                               for col_name in sorted(row, key=lambda x:example['caption_pos_map'][x])]
                               for row in example['table']]
            table.append(cur_table_idx)

        labels_t = [col_num * example['row_id'] + example['col_id'] for example in examples]
        labels = np.zeros((len(examples)), dtype='int32')
        for i in range(len(labels)):
            labels[i] = labels_t[i]

        col_ids_attentions = [e['col_ids_attention'] for e in examples]

        utterances = [[word_dict[x] for x in example['utterance'].split()] + [end_idx] for example in examples]

        if name == 'train':
            utterances_train_indexed = pad_sequences(utterances)
            labels_train = labels
            table_train = table
            table_train = np.asarray(table_train, dtype='int32')
            col_ids_attention_train = col_ids_attentions
            train_meta_data = [e['meta'] for e in examples]

        if name == 'test':
            utterances_test_indexed = pad_sequences(utterances)
            labels_test = labels
            table_test = table
            table_test = np.asarray(table_test, dtype='int32')
            col_ids_attention_test = col_ids_attentions
            test_meta_data = [e['meta'] for e in examples]

    print '# training example:', len(utterances_train_indexed)
    print '# testing example:', len(utterances_test_indexed)

    # assert all(not any([(u == u1).all() for u1 in utterances_test_indexed]) for u in utterances_train_indexed)
    return (train_meta_data, utterances_train_indexed, table_train, labels_train, col_ids_attention_train), \
           (test_meta_data, utterances_test_indexed, table_test, labels_test, col_ids_attention_test), \
           word_dict


def generate_dataset(examples, test_split=0.1, mode='pos', world=None, vocab=None):
    np.random.shuffle(examples)
    # examples = random.sample(examples, 800)
    word_dict = dict() if vocab is None else vocab

    col_num = len(world.world)
    row_num = config.get('KB.row_num')

    # zero-index token is reserved for padding!
    pad = get_token_idx('<pad>', word_dict)

    if world:
        for field in world.world:
            for v in world.world[field]:
                # if field == 'host_city':
                #     get_token_idx(str(v), word_dict, True)
                #     get_token_idx(str(v) + '_new', word_dict, True)
                # else:
                #     get_token_idx(str(v), word_dict, False)
                get_token_idx(str(v), word_dict)

    end_idx = get_token_idx('<eos>', word_dict)

    unique_utterances = list(set(e['utterance'] for e in examples))
    np.random.shuffle(unique_utterances)

    # unique_utterances_ordinary_case = [u for u in unique_utterances if u.split(' ')[1] != 'host_city']
    # unique_utterances_special_case_train = [u for u in unique_utterances if u.split(' ')[1] == 'host_city' and '_new' not in u]
    # unique_utterances_special_case_test = [u for u in unique_utterances if u.split(' ')[1] == 'host_city' and '_new' in u]

    test_utterances = unique_utterances[int(len(unique_utterances) * (1 - test_split)):]
    # test_utterances = unique_utterances_ordinary_case[int(len(unique_utterances_ordinary_case) / 2):] + unique_utterances_special_case_test

    print '# unique_utterances:', len(unique_utterances)
    print '# test_utterances:', len(test_utterances)

    # for u in test_utterances:
    #    print u

    print '# train_utterances:', len(unique_utterances) - len(test_utterances)

    train_examples = [e for e in examples if e['utterance'] not in test_utterances]
    test_examples = [e for e in examples if e['utterance'] in test_utterances]
    test_examples_num = len(test_examples)

    train_denot_tokens = set(e['denotation'] for e in train_examples)
    print len(train_denot_tokens)

    set1 = set(d['utterance'] for d in train_examples)
    set2 = set(d['utterance'] for d in test_examples)

    assert len(set1.intersection(set2)) == 0

    # train_examples_num = int(len(examples) * (1- test_split))
    # train_examples = examples[:train_examples_num]
    # test_examples = examples[train_examples_num:]
    # test_examples_num = len(test_examples)
    # test_utterances = set(e['utterance'] for e in test_examples)
    # train_utterances = set(e['utterance'] for e in train_examples)
    # train_examples = [e for e in train_examples if e['utterance'] not in test_utterances]

    examples = train_examples + test_examples

    utterances = [[get_token_idx(x, word_dict) for x in example['utterance'].split()] + [end_idx] for example in examples]
    # utterances = [[get_token_idx(x, word_dict) for x in example['utterance'].split()] for example in examples]
    # example[1][0] - rowId
    # example[1][1] - colId
    # labels = [get_token_idx(str((example[1][0], example[1][1])), word_dict) for example in examples]

    train_tokens = {x for e in examples[:int(len(examples)*(1-test_split))] for x in e['tokens']}
    test_tokens = {x for e in examples[int(len(examples)*(1-test_split)):] for x in e['tokens']}

    # TODO: open it!
    for x in test_tokens:
        if x not in train_tokens:
            raise RuntimeError('[%s] not in train tokens!' % x)
    # assert all([(x in train_tokens) for x in test_tokens])

    # index table!
    # table_idx = [[get_token_idx(str(row[key]), word_dict) for key in row] for row in table]
    table_idx = []
    for example in examples:
        cur_table_idx = [[[get_token_idx(str(col_name), word_dict), get_token_idx(str(row[col_name]), word_dict)]
                           for col_name in sorted(row, key=lambda x:example['caption_pos_map'][x])]
                           for row in example['table']]
        table_idx.append(cur_table_idx)

    print 'using mode', mode

    if mode == 'pos':
        labels = [col_num * example['row_id'] + example['col_id'] for example in examples]
        labels_binary = np.zeros((len(examples), col_num * row_num), dtype='float32')
        for i in range(len(labels)):
            labels_binary[i, labels[i]] = 1
            assert labels_binary[i, :].sum() == 1
    elif mode == 'pos_val':
        labels = [col_num * example['row_id'] + example['col_id'] for example in examples]
        labels_binary = np.zeros((len(examples)), dtype='int32')
        for i in range(len(labels_binary)):
            labels_binary[i] = labels[i]
    elif mode == 'vocab':
        labels_binary = np.zeros((len(examples), len(word_dict)), dtype='float32')
        for i in range(len(labels_binary)):
            e = examples[i]
            labels_binary[i, get_token_idx(str(e['denotation']), word_dict)] = 1
            assert labels_binary[i, :].sum() == 1
    elif mode == 'ans_vocab':
        print 'world_tok_count', world.world_tok_count
        labels_binary = np.zeros((len(examples), world.world_tok_count), dtype='float32')
        for i in range(len(labels_binary)):
            e = examples[i]
            # minus 1 because of padding!
            labels_binary[i, get_token_idx(str(e['denotation']), word_dict) - 1] = 1
            assert labels_binary[i, :].sum() == 1
    else:
        raise Exception('unknown mode')

    # duplicate the table
    # table_idx = [table_idx] * len(examples)

    # utterances_train_indexed = pad_sequences(utterances[:int(len(examples)*(1-test_split))])
    # labels_train = labels_binary[:int(len(examples)*(1-test_split))]
    # table_train = table_idx[:int(len(examples)*(1-test_split))]
    # table_train = np.asarray(table_train)
    #
    # utterances_test_indexed = pad_sequences(utterances[int(len(examples)*(1-test_split)):])
    # labels_test = labels_binary[int(len(examples)*(1-test_split)):]
    # table_test = table_idx[int(len(examples)*(1-test_split)):]
    # table_test = np.asarray(table_test)

    col_ids_attentions = [e['col_ids_attention'] for e in examples]

    utterances_train_indexed = pad_sequences(utterances[:len(examples) - test_examples_num])
    labels_train = labels_binary[:len(examples) - test_examples_num]
    table_train = table_idx[:len(examples) - test_examples_num]
    table_train = np.asarray(table_train, dtype='int32')
    col_ids_attention_train = col_ids_attentions[:len(examples) - test_examples_num]

    utterances_test_indexed = pad_sequences(utterances[len(examples) - test_examples_num:])
    labels_test = labels_binary[len(examples) - test_examples_num:]
    table_test = table_idx[len(examples) - test_examples_num:]
    table_test = np.asarray(table_test, dtype='int32')
    col_ids_attention_test = col_ids_attentions[len(examples) - test_examples_num:]

    print '# training example:', len(utterances_train_indexed)
    print '# testing example:', len(utterances_test_indexed)

    print word_dict

    # assert all(not any([(u == u1).all() for u1 in utterances_test_indexed]) for u in utterances_train_indexed)
    return (utterances_train_indexed, table_train, labels_train, col_ids_attention_train), \
           (utterances_test_indexed, table_test, labels_test, col_ids_attention_test), \
           word_dict


def generate_datasets_fixed_test_size(examples, test_size, train_sizes):
    np.random.shuffle(examples)
    max_train_size = train_sizes[-1]

    unique_logical_forms = list(set(e['logical_form'] for e in examples))

    unique_logical_forms_test_num = int(len(unique_logical_forms) / float(test_size + max_train_size) * test_size)
    print '# unique logical_forms: ', len(unique_logical_forms)

    unique_logical_forms_test = set(list(np.random.choice(unique_logical_forms, size=unique_logical_forms_test_num, replace=False)))
    unique_logical_forms_train = set(e['logical_form'] for e in examples) - unique_logical_forms_test

    # test_examples = examples[-test_size:]
    # unique_logical_forms_test = set(e['utterance'] for e in test_examples)
    # unique_logical_forms_train = set(e['utterance'] for e in examples) - test_utterances

    train_examples = [e for e in examples if e['logical_form'] in unique_logical_forms_train]
    print '# train_logical_forms:', len(unique_logical_forms_train)
    print '# test_logical_forms:', len(unique_logical_forms_test)
    print '# train examples: ', len(train_examples)

    test_examples = np.random.choice([e for e in examples if e['logical_form'] in unique_logical_forms_test],
                                     size=test_size,
                                     replace=False).tolist()

    print '# test examples: ', len(test_examples)

    set1 = set(d['logical_form'] for d in train_examples)
    set2 = set(d['logical_form'] for d in test_examples)

    assert len(set1.intersection(set2)) == 0, 'train logical form and test logical form overlap!'

    for train_size in train_sizes:
        print '======== train_size = %d ========' % train_size
        train_examples_sliced = list(np.random.choice(train_examples, size=train_size, replace=False))

        dataset = generate_dataset_fixed_size(train_examples_sliced, test_examples)
        yield {'train_examples': train_examples_sliced, 'test_examples': test_examples, 'dataset': dataset, 'train_size': train_size}


def generate_dataset_oov(examples, sample_size, mode='pos', world=None, vocab=None):
    examples_train, examples_test = examples
    np.random.shuffle(examples_train)
    np.random.shuffle(examples_test)

    get_token_idx = get_token_idx_oov

    word_dict = dict() if vocab is None else vocab

    col_num = len(examples_train[0]['table'][0])
    row_num = len(examples_train[0]['table'])

    # zero-index token is reserved for padding!
    pad = get_token_idx('<pad>', word_dict)

    if world:
        for field in world.world:
            for v in world.world[field]:
                if field in ToyWorld.oov_fields:
                     get_token_idx(str(v), word_dict, True)
                     get_token_idx(str(v) + '_new', word_dict, True)
                else:
                     get_token_idx(str(v), word_dict, False)

    end_idx = get_token_idx('<eos>', word_dict)

    unique_utterances_train = set(e['utterance'] for e in examples_train)
    unique_utterances_test = set(e['utterance'] for e in examples_test)
    unique_utterances_intersect = unique_utterances_train.intersection(unique_utterances_test)

    unique_utterances_train_num = len(unique_utterances_train)
    unique_utterances_test_num = len(unique_utterances_test)
    unique_utterances_intersect_num = len(unique_utterances_intersect)

    print '# unique utterances in train set: {}'.format(unique_utterances_train_num)
    print '# unique utterances in test set: {}'.format(unique_utterances_test_num)
    print '# unique utterances in intersect sets: {}'.format(unique_utterances_intersect_num)

    assert unique_utterances_test_num >= unique_utterances_train_num
    if unique_utterances_test_num - unique_utterances_intersect_num >= unique_utterances_train_num:
        unique_utterances_train = unique_utterances_train
        unique_utterances_test = unique_utterances_test - unique_utterances_intersect
    else:
        t1 = int((unique_utterances_test_num - unique_utterances_train_num + unique_utterances_intersect_num) / 2)
        t2 = unique_utterances_intersect_num - t1

        unique_utterances_train = (unique_utterances_train - unique_utterances_intersect).union(set(list(unique_utterances_intersect)[:t1]))
        unique_utterances_test = (unique_utterances_test - unique_utterances_intersect).union(set(list(unique_utterances_intersect)[t1:]))

    print '# balanced unique_utterances_train:', len(unique_utterances_train)
    print '# balanced unique_utterances_test:', len(unique_utterances_test)

    train_examples = [e for e in examples_train if e['utterance'] in unique_utterances_train][0:sample_size]
    test_examples = [e for e in examples_test if e['utterance'] in unique_utterances_test][0:sample_size]
    test_examples_num = len(test_examples)

    set1 = set(d['utterance'] for d in train_examples)
    set2 = set(d['utterance'] for d in test_examples)

    assert len(set1.intersection(set2)) == 0

    # train_examples_num = int(len(examples) * (1- test_split))
    # train_examples = examples[:train_examples_num]
    # test_examples = examples[train_examples_num:]
    # test_examples_num = len(test_examples)
    # test_utterances = set(e['utterance'] for e in test_examples)
    # train_utterances = set(e['utterance'] for e in train_examples)
    # train_examples = [e for e in train_examples if e['utterance'] not in test_utterances]

    examples = train_examples + test_examples

    utterances = [[get_token_idx(x, word_dict) for x in example['utterance'].split()] + [end_idx] for example in examples]
    # utterances = [[get_token_idx(x, word_dict) for x in example['utterance'].split()] for example in examples]
    # example[1][0] - rowId
    # example[1][1] - colId
    # labels = [get_token_idx(str((example[1][0], example[1][1])), word_dict) for example in examples]

    # train_tokens = {x for e in examples[:int(len(examples)*(1-test_split))] for x in e['tokens']}
    # test_tokens = {x for e in examples[int(len(examples)*(1-test_split)):] for x in e['tokens']}

    # TODO: open it!
    # for x in test_tokens:
    #     if x not in train_tokens:
    #         raise RuntimeError('[%s] not in train tokens!' % x)
    # assert all([(x in train_tokens) for x in test_tokens])

    # index table!
    # table_idx = [[get_token_idx(str(row[key]), word_dict) for key in row] for row in table]
    table_idx = []
    for example in examples:
        cur_table_idx = [[[get_token_idx(str(col_name), word_dict), get_token_idx(str(row[col_name]), word_dict)]
                           for col_name in sorted(row, key=lambda x:example['caption_pos_map'][x])]
                           for row in example['table']]
        table_idx.append(cur_table_idx)

    print 'using mode', mode

    if mode == 'pos':
        labels = [col_num * example['row_id'] + example['col_id'] for example in examples]
        labels_binary = np.zeros((len(examples), col_num * row_num), dtype='float32')
        for i in range(len(labels)):
            labels_binary[i, labels[i]] = 1
            assert labels_binary[i, :].sum() == 1
    elif mode == 'pos_val':
        labels = [col_num * example['row_id'] + example['col_id'] for example in examples]
        labels_binary = np.zeros((len(examples)), dtype='int32')
        for i in range(len(labels_binary)):
            labels_binary[i] = labels[i]
    elif mode == 'vocab':
        labels_binary = np.zeros((len(examples), len(word_dict)), dtype='float32')
        for i in range(len(labels_binary)):
            e = examples[i]
            labels_binary[i, get_token_idx(str(e['denotation']), word_dict)] = 1
            assert labels_binary[i, :].sum() == 1
    elif mode == 'ans_vocab':
        print 'world_tok_count', world.world_tok_count
        labels_binary = np.zeros((len(examples), world.world_tok_count), dtype='float32')
        for i in range(len(labels_binary)):
            e = examples[i]
            # minus 1 because of padding!
            labels_binary[i, get_token_idx(str(e['denotation']), word_dict) - 1] = 1
            assert labels_binary[i, :].sum() == 1
    else:
        raise Exception('unknown mode')

    # duplicate the table
    # table_idx = [table_idx] * len(examples)

    # utterances_train_indexed = pad_sequences(utterances[:int(len(examples)*(1-test_split))])
    # labels_train = labels_binary[:int(len(examples)*(1-test_split))]
    # table_train = table_idx[:int(len(examples)*(1-test_split))]
    # table_train = np.asarray(table_train)
    #
    # utterances_test_indexed = pad_sequences(utterances[int(len(examples)*(1-test_split)):])
    # labels_test = labels_binary[int(len(examples)*(1-test_split)):]
    # table_test = table_idx[int(len(examples)*(1-test_split)):]
    # table_test = np.asarray(table_test)

    col_ids_attentions = [e['col_ids_attention'] for e in examples]

    utterances_train_indexed = pad_sequences(utterances[:len(examples) - test_examples_num])
    labels_train = labels_binary[:len(examples) - test_examples_num]
    table_train = table_idx[:len(examples) - test_examples_num]
    table_train = np.asarray(table_train, dtype='int32')
    col_ids_attention_train = col_ids_attentions[:len(examples) - test_examples_num]

    utterances_test_indexed = pad_sequences(utterances[len(examples) - test_examples_num:])
    labels_test = labels_binary[len(examples) - test_examples_num:]
    table_test = table_idx[len(examples) - test_examples_num:]
    table_test = np.asarray(table_test, dtype='int32')
    col_ids_attention_test = col_ids_attentions[len(examples) - test_examples_num:]

    print '# training example:', len(utterances_train_indexed)
    print '# testing example:', len(utterances_test_indexed)

    print word_dict

    # assert all(not any([(u == u1).all() for u1 in utterances_test_indexed]) for u in utterances_train_indexed)
    return (utterances_train_indexed, table_train, labels_train, col_ids_attention_train), \
           (utterances_test_indexed, table_test, labels_test, col_ids_attention_test), \
           word_dict


def mix_examples(examples1, examples2, examples3, examples4, ratio1, ratio2, ratio3, ratio4):
    return np.random.choice(examples1, int(len(examples1) * ratio1)).tolist() + \
           np.random.choice(examples2, int(len(examples2) * ratio2)).tolist() + \
           np.random.choice(examples3, int(len(examples3) * ratio3)).tolist() + \
           np.random.choice(examples4, int(len(examples4) * ratio4)).tolist()


def save_dataset(data, file):
    world = ToyWorld()
    # examples = generate_examples_multi_query(world)
    # examples = generate_examples_superlative(world)
    # data = generate_dataset(examples, mode='pos', world=world)
    # data = generate_dataset_fixed_size(examples, train_num=50000, test_num=5000, mode='vocab')
    import utils
    utils.serialize_to_file(data, file)


def interpret_dataset(file):
    dataset = deserialize_from_file(file)
    (meta_data_train, utterances_train, table_train, labels_train, col_ids_attention_train), (meta_data_test, utterances_test, table_test, labels_test, col_ids_attention_test), vocab_dict = dataset

    vocab_dict_inverse = dict((vocab_dict[tok], tok) for tok in vocab_dict)

    for meta_data, utterances, tables, labels, col_ids_attention, suffix in [(meta_data_train, utterances_train, table_train, labels_train, col_ids_attention_train, 'train'),
                                                                             (meta_data_test, utterances_test, table_test, labels_test, col_ids_attention_test, 'test')]:
        exp_num = utterances.shape[0]

        f = open(file + '.interpret.' + suffix, 'w') # sys.stdout

        unique_queries = set()

        for idx in range(exp_num):
            query_str, table_str, ref_ans, fields_order = interpret_example(utterances[idx], vocab_dict_inverse,
                                                                            tables[idx], labels[idx])

            print >>f, 'query: ' + query_str
            print >>f, 'table:'
            print >>f, table_str

            print >>f, 'query type: ' + meta_data[idx]['type']
            print >>f, 'col_ids_attention: ' + str(col_ids_attention[idx])
            print >>f, 'target answer: ' + ref_ans
            print >>f, '==========================='

            unique_queries.add(query_str)

        print >>f, '\n\nunique queries:'
        for q in unique_queries:
            print >>f, q

        f.close()


def ans_idx2word(pos_id, table, vocab_dict_inverse, col_num=5):
    row_id = int(pos_id / col_num)
    col_id = int(pos_id % col_num)

    pred_ans_id = table[row_id][col_id][1]
    pred_ans = vocab_dict_inverse[pred_ans_id]

    return pred_ans


def get_query_utterance(query_idx, vocab_dict, is_inversed_vocab=False):
    if not is_inversed_vocab:
        vocab_dict = dict((vocab_dict[tok], tok) for tok in vocab_dict)

    query = [vocab_dict[i] for i in query_idx]
    from itertools import dropwhile

    query = dropwhile(lambda x: x == '<pad>', query)

    query_str = ' '.join(query)

    return query_str


def interpret_example(utterance, vocab_dict, table=None, label=None, remove_padding=True, is_inversed_vocab=True,
                      ret_table_obj_fmt=False):
    if not is_inversed_vocab:
        vocab_dict = dict((vocab_dict[tok], tok) for tok in vocab_dict)

    query = [vocab_dict[i] for i in utterance]
    from itertools import dropwhile
    if remove_padding:
        query = dropwhile(lambda x: x == '<pad>', query)

    query_str = ' '.join(query)

    table_str = ref_ans = fields_order = None
    table_obj = None

    if table is not None:
        first_row = True
        fields_order = []
        table_obj = list()

        table_str = ''
        for row in table:
            cur_row = dict()
            for col in row:
                if first_row:
                    fields_order.append(vocab_dict[col[0]])
                table_str += '(' + vocab_dict[col[0]] + ', ' + vocab_dict[col[1]] + ') '
                cur_row[vocab_dict[col[0]]] = vocab_dict[col[1]]
            table_str += '\n'
            table_obj.append(cur_row)
            first_row = False

    if label is not None and table is not None:
        ref_ans = ans_idx2word(label, table, vocab_dict, len(table[0]))

    return query_str, table_obj if ret_table_obj_fmt else table_str, ref_ans, fields_order


def modify_dataset_by_vocab(examples, vocab_dict, target_vocab_dict):
    vocab_dict_inverse = dict((vocab_dict[tok], tok) for tok in vocab_dict)
    row_num = config.get('KB.row_num')
    col_num = config.get('KB.col_num')

    utterances_test, table_test = examples

    exp_num = utterances_test.shape[0]
    for idx in range(exp_num):
        for i in range(len(utterances_test[idx])):
            utterances_test[idx][i] = target_vocab_dict[vocab_dict_inverse[utterances_test[idx][i]]]

        for row_id in range(row_num):
            for col_id in range(col_num):
                tok = table_test[idx][row_id][col_id][0]
                table_test[idx][row_id][col_id][0] = target_vocab_dict[vocab_dict_inverse[tok]]
                tok = table_test[idx][row_id][col_id][1]
                table_test[idx][row_id][col_id][1] = target_vocab_dict[vocab_dict_inverse[tok]]

    return (utterances_test, table_test)


def reverse_dataset(file):
    dataset = deserialize_from_file(file)
    (utterances_train, table_train, labels_train), (utterances_test, table_test, labels_test), vocab_dict = dataset
    import utils

    num = len(utterances_train)
    for idx in range(num):
        non_zero_i = 0
        while utterances_train[idx][non_zero_i] == 0:
            non_zero_i += 1

        sent_len = len(utterances_train[idx])
        utterances_train[idx][non_zero_i:-1] = utterances_train[idx][non_zero_i:-1][::-1]

    num = len(utterances_test)
    for idx in range(num):
        non_zero_i = 0
        while utterances_test[idx][non_zero_i] == 0:
            non_zero_i += 1

        sent_len = len(utterances_test[idx])
        utterances_test[idx][non_zero_i:-1] = utterances_test[idx][non_zero_i:-1][::-1]

    dataset = (utterances_train, table_train, labels_train), (utterances_test, table_test, labels_test), vocab_dict
    utils.serialize_to_file(dataset, file + '.reverse')


def double_input_dataset(file):
    dataset = deserialize_from_file(file)
    (utterances_train, table_train, labels_train), (utterances_test, table_test, labels_test), vocab_dict = dataset
    import utils

    num = len(utterances_train)
    sent_len = len(utterances_train[0])

    utterances_train_doubled = np.zeros((num, sent_len * 2), dtype='int32')

    for idx in range(num):
        non_zero_i = 0
        while utterances_train[idx][non_zero_i] == 0:
            non_zero_i += 1

        utterances_train_doubled[idx][:non_zero_i] = utterances_train[idx][:non_zero_i]
        utterances_train_doubled[idx][non_zero_i:non_zero_i * 2] = utterances_train[idx][:non_zero_i]
        utterances_train_doubled[idx][non_zero_i * 2:non_zero_i + sent_len] = utterances_train[idx][non_zero_i:]
        utterances_train_doubled[idx][non_zero_i + sent_len:] = utterances_train[idx][non_zero_i:]

    num = len(utterances_test)
    sent_len = len(utterances_test[0])
    utterances_test_doubled = np.zeros((num, sent_len * 2), dtype='int32')

    for idx in range(num):
        non_zero_i = 0
        while utterances_test[idx][non_zero_i] == 0:
            non_zero_i += 1

        utterances_test_doubled[idx][:non_zero_i] = utterances_test[idx][:non_zero_i]
        utterances_test_doubled[idx][non_zero_i:non_zero_i * 2] = utterances_test[idx][:non_zero_i]
        utterances_test_doubled[idx][non_zero_i * 2:non_zero_i + sent_len] = utterances_test[idx][non_zero_i:]
        utterances_test_doubled[idx][non_zero_i + sent_len:] = utterances_test[idx][non_zero_i:]

    dataset = (utterances_train_doubled, table_train, labels_train), (utterances_test_doubled, table_test, labels_test), vocab_dict
    utils.serialize_to_file(dataset, file + '.double')


def generate_dataset_based_on_vocab():
    world = ToyWorld()
    _, _, vocab = deserialize_from_file('datasets/examples.sample.nested_query_4field.20w.world_size60.pos_val.padding.no_superlative_short_cut.distinct_field')

    examples = generate_examples_nested_query_4field(world)
    # examples = generate_examples_single_select(world)
    data = generate_dataset(examples, mode='pos_val', world=world, vocab=vocab)
    # save_dataset(data, 'datasets/examples.sample.single_select.50k.world_size60.pos_val.padding')
    save_dataset(data, 'datasets/examples.sample.nested_query_4field.50k.world_size60.pos_val.padding.20w_vocab')

    print 'generate_dataset_based_on_vocab() done!'


def generate_dataset_by_func(f, sample_size, dataset_name, test_split=0.5):
    print f.__name__, dataset_name
    world = ToyWorld()
    examples = f(world, sample_size)
    data = generate_dataset(examples, mode='pos_val', world=world, test_split=test_split)
    save_dataset(data, dataset_name)
    interpret_dataset(dataset_name)
    print 'generate_dataset_by_func done!'


def generate_dataset_by_func_fixed_test_size(f, dataset_prefix):
    print f.__name__, dataset_prefix
    world = ToyWorld()
    examples = f(world, 33000)
    # examples = f(world, 27000)
    datasets = generate_datasets_fixed_test_size(examples, 10000, max_train_size=20000)
    for d in datasets:
        dataset_name = dataset_prefix + '.' + str(d['train_size'])
        save_dataset(d['dataset'], dataset_name)
        interpret_dataset(dataset_name)
    print 'generate_dataset_by_func_fixed_test_size done!'


def generate_dataset_by_func_oov(f, sample_size, dataset_name, p_oov=0.5):
    print f.__name__, dataset_name
    world = ToyWorld()
    examples = f(world, sample_size, p_oov=p_oov)
    data = generate_dataset_oov(examples, sample_size, mode='pos_val', world=world)
    save_dataset(data, dataset_name)
    interpret_dataset(dataset_name)
    print 'generate_dataset_by_func_oov done!'


def generate_mixed_dataset():
    world = ToyWorld()

    examples1 = generate_examples_single_select(world, 40000)  # 4w
    examples2 = generate_examples_superlative(world, 40000)    # 4w
    examples3 = generate_examples_where_superlative_indeptfield(world, 40000)    # 4w
    examples4 = generate_examples_nested_query_4field(world, 100000)    # 10w

    examples = examples1 + examples2 + examples3 + examples4

    data = generate_dataset(examples, mode='pos_val', world=world, test_split=0.5)
    save_dataset(data, 'datasets/examples.sample.combined.11w.world_size60.pos_val.padding.with_col_ids.col_num10.gen2')
    interpret_dataset('datasets/examples.sample.combined.11w.world_size60.pos_val.padding.with_col_ids.col_num10.gen2')

    print 'generate_mixed_dataset() done!'


def generate_mixed_dataset_nl_query():
    world = ToyWorld()

    suffix = '.refine2.gdp.sup_same_dir'

    funcs = [(generate_examples_single_select_nl_query, 'datasets/examples.single_select.world_size60.pos_val.col_num10.nl' + suffix, [5000, 10000, 15000, 20000]),
             (generate_examples_superlative_nl_query, 'datasets/examples.superlative.world_size60.pos_val.col_num10.nl' + suffix, [5000, 10000, 15000, 20000]),
             (generate_examples_where_superlative_indeptfield_nl_query, 'datasets/examples.where_superlative_indeptfield.world_size60.pos_val.col_num10.nl' + suffix, [5000, 10000, 15000, 20000]),
             (generate_examples_nested_query_4field_nl_query, 'datasets/examples.nested_query_4field.world_size60.pos_val.col_num10.nl' + suffix, [10000, 20000, 30000, 40000])]

    datasets = []

    for id, (f, dataset_prefix, train_sizes) in enumerate(funcs):
        print 'generate %s' % f.__name__
        # examples = f(world, max_train_size + 10500)
        max_train_size = train_sizes[-1]
        examples = f(world, max_train_size + 5500)

        cur_datasets = []

        for d in generate_datasets_fixed_test_size(examples, 5000, train_sizes=train_sizes):
            cur_datasets.append(d)

        # save last dataset!
        train_size = cur_datasets[-1]['train_size']
        dataset_name = dataset_prefix + '.' + str(train_size)
        save_dataset(cur_datasets[-1]['dataset'], dataset_name)
        interpret_dataset(dataset_name)

        datasets.append(cur_datasets)

    print '============generate mixed dataset============'

    for ds_id in range(4):
        train_examples = []
        test_examples = []
        size = 0
        for type_id in range(4):
            cur_dataset = datasets[type_id][ds_id]
            size += cur_dataset['train_size']
            train_examples.extend(cur_dataset['train_examples'])
            test_examples.extend(cur_dataset['test_examples'])

        mixed_dataset = generate_dataset_fixed_size(train_examples, test_examples)
        mixed_dataset_name = ('datasets/examples.sample.combined.{size}.nl' + suffix).format(size=size)
        save_dataset(mixed_dataset, mixed_dataset_name)
        interpret_dataset(mixed_dataset_name)

    print 'generate mixed dataset done'
    print 'generate_mixed_dataset_nl_query done!'


def generate_mixed_dataset_oov(p_oov=1.0):
    world = ToyWorld()

    examples1 = generate_examples_single_select_oov(world, 20000, p_oov=p_oov)  # 2w
    examples2 = generate_examples_superlative_oov(world, 20000, p_oov=p_oov)    # 2w
    examples3 = generate_examples_where_superlative_indeptfield_oov(world, 20000, p_oov=p_oov)    # 2w
    examples4 = generate_examples_nested_query_4field_oov(world, 20000, p_oov=p_oov)    # 5w

    examples_train = examples1[0] + examples2[0] + examples3[0] + examples4[0]
    examples_test = examples1[1] + examples2[1] + examples3[1] + examples4[1]

    data = generate_dataset_oov((examples_train, examples_test), 80000, mode='pos_val', world=world)
    save_dataset(data, 'datasets/examples.sample.combined.8w.world_size60.pos_val.padding.with_col_ids.col_num10.oov1.0')
    interpret_dataset('datasets/examples.sample.combined.8w.world_size60.pos_val.padding.with_col_ids.col_num10.oov1.0')

    print 'generate_mixed_dataset_oov() done!'


def generate_cross_table_dataset():
    world = ToyWorld()

    suffix = '.7.5w.5tables.5fields'
    dataset_name = 'datasets/cross_table.nl' + suffix

    funcs = [(generate_examples_single_select_nl_query, 5000),
             (generate_examples_superlative_nl_query, 5000),
             (generate_examples_where_superlative_indeptfield_nl_query, 5000),
             ]

    # field_set_1 = {'host_city', 'year', '#_participants', 'country_gdp', '#_duration'}
    # field_set_2 = {'country', '#_audience', 'country_size', '#_medals', 'country_population'}
    field_sets_num = 5
    field_num_per_table = 5
    field_sets = []
    all_fields = [f for f in world.world]

    for i in range(field_sets_num):
        field_sets.append(set(np.random.choice(all_fields, size=field_num_per_table, replace=False)))
        print field_sets[i]

    for i, s in enumerate(field_sets):
        u_set = set()
        for j in range(field_sets_num):
            if j != i: u_set |= field_sets[j]

        diff_set = s - u_set
        print diff_set

    train_examples = []
    test_examples = []

    for id, (f, sample_size) in enumerate(funcs):
        print 'generate %s' % f.__name__
        for field_set in field_sets:
            train_examples_field_set = f(world, sample_size, field_set)
            train_examples.extend(train_examples_field_set)

        test_examples_combo_field_sets = f(world, 10000, all_fields, query_cross_field_sets=field_sets)

        test_examples.extend(test_examples_combo_field_sets)

    print '============generate mixed dataset============'

    dataset = generate_dataset_fixed_size(train_examples, test_examples)
    save_dataset(dataset, dataset_name)
    interpret_dataset(dataset_name)


def extract_field_names(table):
    # table: (example_num, row_num, col_num, 2)
    example_num = table.shape[0]
    row_num = table.shape[1]
    col_num = table.shape[2]

    field_names = np.zeros((example_num, col_num), dtype='int32')

    for idx in range(example_num):
        for col_id in range(col_num):
            field_names[idx][col_id] = table[idx][0][col_id][0]

    return field_names


class Dataset(object):
    def __int__(self):
        # [string, Dict[string, numpy_array]]
        self.data = dict()


if __name__ == '__main__':
    config.init_config()
    world = ToyWorld()

    # generate_mixed_dataset_oov()

    # generate_mixed_dataset_nl_query()

    # generate_cross_table_dataset()

    to_sempre_dataset('datasets/examples.sample.combined.25000.nl.refine2.gdp.sup_same_dir')


    # world = ToyWorld()
    # tbl = world.sample_table(10)

    # interpret_dataset('datasets/examples.sample.nested_query_4field.20w.world_size60.pos_val.padding')

    # path = 'datasets/examples.sample.nested_query_4field.50k.world_size60.pos_val.padding.20w_vocab'
    # dataset = deserialize_from_file(path)
    # (utterances_train, table_train, labels_train), (utterances_test, table_test, labels_test), vocab_dict = dataset
    # att_weights = add_attention_supervision_signal(path)
    # dataset = (utterances_train, table_train, labels_train, att_weights[0]), (utterances_test, table_test, labels_test, att_weights[1]), vocab_dict
    # #
    # save_dataset(dataset, path + '.att_weights')


    # reverse_dataset('datasets/examples.sample.nested_query_4field.50k.world_size60.pos_val.padding.20w_vocab')
    # double_input_dataset('datasets/examples.sample.nested_query_4field.50k.world_size60.pos_val.padding.20w_vocab')
    # interpret_dataset('datasets/examples.sample.nested_query_4field.50k.world_size60.pos_val.padding.20w_vocab.double')

    print 'done!'

