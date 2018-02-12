from __future__ import print_function
from __future__ import division
import numpy as np
import glove
import math
from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer

def load(filepath,train=True,test=True):
    print('Load text ...')
    data=np.load(filepath)
    X_train_p = []
    X_train_q = []
    X_test_p = []
    X_test_q = []
    if train:
        X_train_p=data['X_train_p']
        X_train_q=data['X_train_q']
        print(X_train_p.shape)
        print(X_train_q.shape)
    if test:
        X_test_p=data['X_test_p']
        X_test_q=data['X_test_q']
        print(X_test_p.shape)
        print(X_test_q.shape)
    print('Load text done.')
    return X_train_p, X_train_q, X_test_p, X_test_q

def load_ans(filepath):
    print('Load ans index ...')
    data=np.load(filepath)
    X_train_ans=data['X_train_ans']
    X_test_ans=data['X_test_ans']
    print(X_train_ans.shape)
    print(X_test_ans.shape)
    print('Load ans index done.')
    return X_train_ans, X_test_ans

def print_one(x_p,x_q,ans,embedding,D):
    cur_process_num=1
    cur_batch_size=int(math.ceil(len(x_p)/cur_process_num))
    tokens_all_p,_=embedding.reverse_embedding(x_p,k=1,embedding=True,batch_size=cur_batch_size,process_num=cur_process_num,maxlen=300)
    cur_batch_size=int(math.ceil(len(x_q)/cur_process_num))
    tokens_all_q,_=embedding.reverse_embedding(x_q,k=1,embedding=True,batch_size=cur_batch_size,process_num=cur_process_num,maxlen=60)

    for idx in range(len(x_p)):
        tokens_p=tokens_all_p[idx][:,0]
        print(D.tokenize(tokens_p, return_str=True))
        tokens_q=tokens_all_q[idx][:,0]
        print(D.tokenize(tokens_q, return_str=True))
	#for ans
        for i in range(ans[idx][0],ans[idx][1]+1):
            print(tokens_p[i])
class Squad:

    def load_data(self, train=True, test=True):
        filepath="/home/wzw0022/data/QA/squad.npz"
        X_train_p, X_train_q, X_test_p, X_test_q = load(filepath, train=train, test=test)
        filepath="/home/wzw0022/data/QA/ans_idx.npz"
        X_train_ans, X_test_ans = load_ans(filepath)
        return X_train_p, X_train_q, X_train_ans, X_test_p, X_test_q, X_test_ans

if __name__ == "__main__":
    filepath="~/data/QA/squad.npz"
    embedding = glove.Glove()
    D = MosesDetokenizer()
    X_train_p, X_train_q, X_test_p, X_test_q = load(filepath)
    filepath="~/data/QA/ans_idx.npz"
    X_train_ans, X_test_ans = load_ans(filepath)
    leng=10
    print_one(X_test_p[200:200+leng],X_test_q[200:200+leng],X_test_ans[200:200+leng],embedding,D)
