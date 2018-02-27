# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import glove
from collections import defaultdict

embedding_dim = glove.Glove.embedding_dim
wiki_path='/home/wzw0022/all_contrib_dev_test/data/DATA/wiki'
overnight_path='/home/wzw0022/all_contrib_dev_test/data/DATA/overnight_source/all'
save_path = '/home/wzw0022/all_contrib_dev_test/data'
'''
0: pad
1: bos
2: eos
'''
_PAD = 0
_GO = 1
_END = 2
_UNK = 3
def build_vocab_all( load=True, files=[os.path.join(wiki_path,'train.lon'),os.path.join(wiki_path,'train.qu'),os.path.join(overnight_path,'train.lon'),os.path.join(overnight_path,'train.qu')] ):
    if load==False:
        vocabs = set()
        for fname in files:
            with gfile.GFile(fname, mode='r') as DATA:
                lines = DATA.readlines()
                for line in lines:
                    for word in line.split():
                        if word not in vocabs:
                            vocabs.add(word)
        vocab_tokens = ['pad','bos','eos','unk']
        vocab_tokens.extend(list(vocabs))
        np.save(os.path.join(save_path,'vocab_tokens_all.npy'),vocab_tokens)
    else:
        vocab_tokens=np.load(os.path.join(save_path,'data/vocab_tokens_all.npy'))

    return vocab_tokens

def load_vocab_all( load=True ):

    if load==False:
        vocab_dict = {}
        reverse_vocab_dict = {}
        embedding = glove.Glove()
        vocabs = build_vocab_all()
        vocab_tokens = []
        for i,word in enumerate(vocabs):
            vocab_dict[word]=i
            reverse_vocab_dict[i]=word.decode('utf-8')
            vocab_tokens.append([word.decode('utf-8')])
        np.save(os.path.join(save_path,'vocab_dict_all.npy'),vocab_dict)
        np.save(os.path.join(save_path,'reverse_vocab_dict_all.npy'),reverse_vocab_dict)
        vocab_emb = embedding.embedding(vocab_tokens, maxlen=1)
        vocab_emb = vocab_emb[:,0] #retrieve embedding
        print(np.max(vocab_emb))
        print(np.min(vocab_emb))
        
        i = 0
        emb_dict = {}
        emb_dict['f'] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
        emb_dict['v'] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
        for token,emb in zip(vocab_tokens,vocab_emb):
            token = token[0]
            if (token[0]=='f' or token[0]=='v') and token[1:].isdigit():
                #print(token)
                #print(emb.shape)
                #print(i)
                if token[1:] in emb_dict:
                    right = emb_dict[token[1:]]
                else:
                    emb_dict[token[1:]] = (np.random.rand(embedding_dim/2)-.5)*2*np.sqrt(3)
                    right = emb_dict[token[1:]]
                re = np.concatenate((emb_dict[token[0]],right))
                assert re.shape==(300,)
                vocab_emb[i]=re
            i += 1

        np.save(os.path.join(save_path,'vocab_emb_all.npy'),vocab_emb)
        print('Vocab shape:')
        print(vocab_emb.shape)
    else:
        vocab_emb=np.load(os.path.join(save_path,'vocab_emb_all.npy'))
        vocab_dict=np.load(os.path.join(save_path,'vocab_dict_all.npy')).item()
        reverse_vocab_dict=np.load(os.path.join(save_path,'reverse_vocab_dict_all.npy')).item()
        print('Vocab shape:')
        print(vocab_emb.shape)
    return vocab_dict,reverse_vocab_dict,vocab_emb


def load_data_wiki(maxlen=20,load=True,s='train'):
    if load:
        emb = np.load(os.path.join(save_path,'vocab_emb_all.npy'))
        print('========embedding shape========')
        print(emb.shape)
        all_q_tokens = np.load(os.path.join(save_path,'wiki/%s_qu_idx.npy'%(s)))
        all_logic_ids = np.load(os.path.join(save_path,'wiki/%s_lon_idx.npy'%(s)))
        print(all_q_tokens.shape)
        print(all_logic_ids.shape)
    else:
        all_q_tokens = []
        all_logic_ids = []
        vocab_dict,_,_=load_vocab_all()
        vocab_dict = defaultdict(lambda:_UNK, vocab_dict)
        questionFile=os.path.join(wiki_path,'%s.qu'%(s))
        logicFile=os.path.join(wiki_path,'%s.lon'%(s))
        with gfile.GFile(questionFile, mode='r') as questions, gfile.GFile(logicFile, mode='r') as logics:
			q_sentences = questions.readlines()
			logics = logics.readlines()
			assert len(q_sentences)==len(logics)
			i = 0
			length = len(logics)
			for q_sentence,logic in zip(q_sentences,logics):
				i+=1
				print('counting: %d / %d'%(i,length),end='\r')
				sys.stdout.flush()
				token_ids = [_GO]
				token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
				token_ids.append(_END)
				logic_ids = [_GO]
				logic_ids.extend([vocab_dict[x] for x in logic.split()])
				logic_ids.append(_END)
				if maxlen>len(logic_ids):
					logic_ids.extend([ _PAD for i in range(len(logic_ids),maxlen)])
				else:
					logic_ids = logic_ids[:maxlen]
				if maxlen>len(token_ids):
					token_ids.extend([ _PAD for i in range(len(token_ids),maxlen)])
				else:
					token_ids = token_ids[:maxlen]
				all_q_tokens.append(token_ids)
				all_logic_ids.append(logic_ids)
        all_logic_ids=np.asarray(all_logic_ids)
        print(all_logic_ids.shape)
        all_q_tokens=np.asarray(all_q_tokens)
        np.save(os.path.join(save_path,'wiki/%s_lon_idx.npy'%s),all_logic_ids)
        np.save(os.path.join(save_path,'wiki/%s_qu_idx.npy'%s),all_q_tokens)
	
    return all_q_tokens,all_logic_ids

def load_data_overnight(maxlen=20,subset='all',load=False,s='train'):
    all_q_tokens = []
    all_logic_ids = []
    vocab_dict,_,_=load_vocab_all()
    vocab_dict = defaultdict(lambda:_UNK,vocab_dict)
    questionFile=os.path.join(overnight_path,'%s.qu'%(s))
    logicFile=os.path.join(overnight_path,'%s.lon'%(s))
    with gfile.GFile(questionFile, mode='r') as questions, gfile.GFile(logicFile, mode='r') as logics:
        q_sentences = questions.readlines()
        logics = logics.readlines()
        assert len(q_sentences)==len(logics)
        for q_sentence,logic in zip(q_sentences,logics):
            token_ids = [_GO]
            token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
            token_ids.append(_END)
            logic_ids = [_GO]
            logic_ids.extend([vocab_dict[x] for x in logic.split()])
            logic_ids.append(_END)
            if maxlen>len(logic_ids):
                logic_ids.extend([ _PAD for i in range(len(logic_ids),maxlen)])
            else:
                logic_ids = logic_ids[:maxlen]
            if maxlen>len(token_ids):
                token_ids.extend([ _PAD for i in range(len(token_ids),maxlen)])
            else:
                token_ids = token_ids[:maxlen]
            all_q_tokens.append(token_ids)
            all_logic_ids.append(logic_ids)
    all_logic_ids=np.asarray(all_logic_ids)
    print('overnight shape:')
    print(all_logic_ids.shape)
    all_q_tokens=np.asarray(all_q_tokens)
    return all_q_tokens,all_logic_ids


def load_data(maxlen=20, load=False, s='train'):
    X1, y1 = load_data_wiki(maxlen=maxlen,load=load,s=s)
    X2, y2 = load_data_overnight(maxlen=maxlen,load=load,s=s)
    X = np.concatenate([X1,X2],axis=0)
    y = np.concatenate([y1,y2],axis=0)
    print(X.shape)
    print(y.shape)
    return X,y

#vocab_dict,_,_=load_vocab_all()
#for word in vocab_dict.keys():
#    if '(' in word or ')' in word:
#        print(word)

def main():
    #load_data(load=False,s='train')
    #load_data(load=False,s='test')
    load_data(load=True,s='dev')

main()
