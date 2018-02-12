import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from utils import glove
embedding_dim = glove.Glove.embedding_dim
path='/home/wzw0022/ww_text/data/overnight_source'
all_path = '/home/wzw0022/ww_text/data/overnight_source/all'
def build_vocab(load=True,data_file=os.path.join(all_path,'all_train.lon')):
	if load==False:
		vocabs = set()
		with gfile.GFile(data_file,mode='r') as DATA:
                        lines = DATA.readlines()
			for line in lines:
				for word in line.split():
					if word not in vocabs:
						vocabs.add(word)
		vocab_tokens = ['pad','bos','eos']
                vocab_tokens.extend(list(vocabs))
		np.save('vocab_tokens.npy',vocab_tokens)
	else:
		vocab_tokens=np.load('vocab_tokens.npy')
	return vocab_tokens

def build_vocab_all(load=True,data_file1=os.path.join(all_path,'all_train.lon'),data_file2=os.path.join(all_path,'all.qu')):
	if load==False:
		vocabs = set()
		with gfile.GFile(data_file1,mode='r') as DATA:
                        lines = DATA.readlines()
			for line in lines:
				for word in line.split():
					if word not in vocabs:
						vocabs.add(word)
		with gfile.GFile(data_file2,mode='r') as DATA:
                        lines = DATA.readlines()
                        for line in lines:
                                for word in line.split():
                                        if word not in vocabs:
                                                vocabs.add(word)
		vocab_tokens = ['pad','bos','eos']
                vocab_tokens.extend(list(vocabs))
		np.save('vocab_tokens_all.npy',vocab_tokens)
		
	else:
		vocab_tokens=np.load('vocab_tokens_all.npy')
	return vocab_tokens

def load_vocab(load=True):
	maxlen=2
	if load==False:
		vocab_dict = {}
		reverse_vocab_dict = {}
		embedding = glove.Glove()
		vocabs = build_vocab()
		vocab_tokens = []
		for i,word in enumerate(vocabs):
			vocab_dict[word]=i
			reverse_vocab_dict[i]=word
			vocab_tokens.append([word])
		np.save('vocab_dict.npy',vocab_dict)
		np.save('reverse_vocab_dict.npy',reverse_vocab_dict)
		vocab_emb = embedding.embedding(vocab_tokens, maxlen=maxlen-1)
                print(vocab_emb.shape)
		vocab_emb = vocab_emb[:,0] #discard begin word
		np.save('vocab_emb.npy',vocab_emb)
		print('vocab shape:')
		print(vocab_emb.shape)
		
	else:
		vocab_dict=np.load('vocab_dict.npy').item()
		reverse_vocab_dict=np.load('reverse_vocab_dict.npy').item()
	return vocab_dict,reverse_vocab_dict


def load_vocab_all(load=True):
	maxlen=2
	if load==False:
		vocab_dict = {}
		reverse_vocab_dict = {}
		embedding = glove.Glove()
		vocabs = build_vocab_all()
		vocab_tokens = []
		for i,word in enumerate(vocabs):
			vocab_dict[word]=i
			reverse_vocab_dict[i]=word
			vocab_tokens.append([word])
		np.save('vocab_dict_all.npy',vocab_dict)
		np.save('reverse_vocab_dict_all.npy',reverse_vocab_dict)
	        vocab_emb = embedding.embedding(vocab_tokens, maxlen=maxlen-1)
                print(vocab_emb.shape)
                vocab_emb = vocab_emb[:,0] #discard begin word
                np.save('vocab_emb_all.npy',vocab_emb)
                print('vocab shape:')
                print(vocab_emb.shape)	
	else:
		vocab_dict=np.load('vocab_dict_all.npy').item()
		reverse_vocab_dict=np.load('reverse_vocab_dict_all.npy').item()
	return vocab_dict,reverse_vocab_dict



def load_data(maxlen=20,subset='basketball',load=True,s='train'):
	all_q_tokens = []
	all_logic_ids = []
	vocab_dict,_=load_vocab()
	questionFile=os.path.join(path,'%s/%s_%s.qu'%(subset,subset,s))
	logicFile=os.path.join(path,'%s/%s_%s.lon'%(subset,subset,s))
	with gfile.GFile(questionFile, mode='r') as questions, gfile.GFile(logicFile, mode='r') as logics:
		q_sentences = questions.readlines()
		logics = logics.readlines()
		assert len(q_sentences)==len(logics)
		i = 0
		for q_sentence,logic in zip(q_sentences,logics):
			i+=1
			tokens = [x for x in q_sentence.split()]
			for x in logic.split():
				if x not in vocab_dict.keys():
					print("ERROR:not found in dict:%s"%x)
			logic_ids = [vocab_dict[x] for x in logic.split()]
			if maxlen>len(logic_ids):
				logic_ids.extend([ 0 for i in range(len(logic_ids),maxlen)])
			else:
				logic_ids = logic_ids[:maxlen]
			all_q_tokens.append(tokens)
			all_logic_ids.append(logic_ids)
	if load==False:
		embedding = glove.Glove()
		emb = embedding.embedding(all_q_tokens, maxlen=maxlen-1)
		np.save('%s_%s_%d_emb.npy'%(subset,s,maxlen),emb)	
	else:
		emb = np.load('%s_%s_%d_emb.npy'%(subset,s,maxlen))
	print(all_q_tokens[0])
	print(emb.shape)
	all_logic_ids=np.asarray(all_logic_ids)
	print(all_logic_ids.shape)
	return emb,all_logic_ids


def load_data_idx(maxlen=20,subset='basketball',load=True,s='train'):
	all_q_tokens = []
	all_logic_ids = []
	vocab_dict,_=load_vocab_all()
	questionFile=os.path.join(path,'%s/%s_%s.qu'%(subset,subset,s))
	logicFile=os.path.join(path,'%s/%s_%s.lon'%(subset,subset,s))
	with gfile.GFile(questionFile, mode='r') as questions, gfile.GFile(logicFile, mode='r') as logics:
		q_sentences = questions.readlines()
		logics = logics.readlines()
		assert len(q_sentences)==len(logics)
		i = 0
		for q_sentence,logic in zip(q_sentences,logics):
			i+=1
			for x in q_sentence.split():
				if x not in vocab_dict.keys():
					print('QUESTION ERROR:not in dict:%s'%x)
                        token_ids = [1]
			token_ids.extend([vocab_dict[x] for x in q_sentence.split()])
                        token_ids.append(2)
			for x in logic.split():
				if x not in vocab_dict.keys():
					print("ERROR:not found in dict:%s"%x)
			logic_ids = [1]
			logic_ids.extend([vocab_dict[x] for x in logic.split()])
                        logic_ids.append(2)
			if maxlen>len(logic_ids):
				logic_ids.extend([ 0 for i in range(len(logic_ids),maxlen)])
			else:
				logic_ids = logic_ids[:maxlen]
			if maxlen>len(token_ids):
                                token_ids.extend([ 0 for i in range(len(token_ids),maxlen)])
                        else:
                                token_ids = token_ids[:maxlen]
			all_q_tokens.append(token_ids)
			all_logic_ids.append(logic_ids)
	all_logic_ids=np.asarray(all_logic_ids)
	print(all_logic_ids.shape)
        all_q_tokens=np.asarray(all_q_tokens)
	return all_q_tokens,all_logic_ids

load_data_idx(subset='restaurants',load=False,s='train')
