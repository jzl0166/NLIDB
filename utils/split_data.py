import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

path='/home/wzw0022/all_contrib_dev_test/data/DATA/overnight_source/all'
def split(path, s='new_all_train', interval=8):
    qu=os.path.join(path,'%s.qu'%(s))
    lo=os.path.join(path,'%s.lon'%(s))
    qu_train=os.path.join(path,'train.qu')
    lo_train=os.path.join(path,'train.lon')
    qu_dev=os.path.join(path,'dev.qu')
    lo_dev=os.path.join(path,'dev.lon')
    with gfile.GFile(qu, mode='r') as qu_r, gfile.GFile(lo, mode='r') as lo_r, gfile.GFile(qu_train, mode='w') as qu_train_w, gfile.GFile(qu_dev, mode='w') as qu_dev_w, gfile.GFile(lo_train, mode='w') as lo_train_w, gfile.GFile(lo_dev, mode='w') as lo_dev_w:
        qus = qu_r.readlines()
        los = lo_r.readlines()
        assert len(qus)==len(los)
        i = 0 
        for q_sentence,logic in zip(qus,los):
            
            if i%interval==0:
                qu_dev_w.write(q_sentence)
                lo_dev_w.write(logic)
            else:
                qu_train_w.write(q_sentence)
                lo_train_w.write(logic)
            i += 1
    return
            
split(path)
