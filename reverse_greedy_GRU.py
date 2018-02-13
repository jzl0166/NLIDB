from __future__ import print_function
import sys
import os
import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils.overnight import load_data,load_data_idx,build_vocab_all
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ----------------------------------------------------------------------------
_PAD = 0
_GO = 1
_END = 2
epochs = 100
lr = 0.001
BS = 64
maxlen = 20
embedding_dim = 300
D = embedding_dim
n_states = int(D/2)
T = maxlen
in_drop=.0
out_drop=.0
vocabulary_size=928
embedding_size=300
subset='all'
load_model=True
input_vocab_size = vocabulary_size
output_vocab_size = vocabulary_size
S = 'bos meeting where ( ( important equal true ) and ( end_time equal 3pm ) )'
dim = n_states
# ----------------------------------------------------------------------------
def train(sess, env, X_data, y_data, epochs=5, load=False, shuffle=True, batch_size=BS,
          name='model'):
    if load:
        print('\nLoading saved model')
        env.saver.restore(sess, 'reverse_model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch+1, epochs))
        sys.stdout.flush()
        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]
       
        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
            start = batch * batch_size
            end = min(n_sample, start+batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                              env.y: y_data[start:end],
                                              env.training: True})
        evaluate(sess, env, X_data, y_data, batch_size=batch_size)
       
	    if (epoch+1)%20==0:
            print('\n Saving model')
            env.saver.save(sess, 'reverse_model/{}'.format(name), global_step=(epoch+1))

def evaluate(sess, env, X_data, y_data, batch_size=BS):
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc, perp = 0, 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
        sys.stdout.flush()
        start = batch * batch_size
        end = min(n_sample, start+batch_size)
        cnt = end - start
        batch_loss, batch_acc, batch_perp = sess.run(
            [env.loss,env.acc,env.perp],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
	perp += batch_perp * cnt
    loss /= n_sample
    acc /= n_sample
    perp /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f} perp: {2:.4f}'.format(loss, acc, perp))
    return acc
#------------------------------------------------------------------------------

class Dummy:
    pass
env = Dummy()



def _decoder( encoder_outputs , encoder_state , mode , beam_width , batch_size):
    
    num_units = 2*dim
    # [batch_size, max_time,...]
    memory = encoder_outputs
    
    seq_len = tf.tile(tf.constant([maxlen], dtype=tf.int32), [ batch_size ] )
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention( num_units = num_units, memory=memory, 
                                                               normalize=True,
                                                               memory_sequence_length=seq_len)

    cell0 = tf.contrib.rnn.GRUCell( 2*dim )
    cell = tf.contrib.rnn.DropoutWrapper(cell0, input_keep_prob=1-in_drop,output_keep_prob=1-out_drop)
    cell = tf.contrib.seq2seq.AttentionWrapper( cell,
                                                attention_mechanism,
                                                attention_layer_size=num_units,
                                                name="attention")

    decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone( cell_state=encoder_state )

    return cell, decoder_initial_state


def Decoder( mode , enc_rnn_out , enc_rnn_state , emb_Y , emb_out):
    
    with tf.variable_scope("Decoder") as decoder_scope:

        mem_units = 2*dim
        out_layer = Dense( output_vocab_size ) #projection W*X+b
        beam_width = 5
        batch_size = tf.shape(enc_rnn_out)[0]

        cell , initial_state = _decoder( enc_rnn_out ,enc_rnn_state  , mode , beam_width ,batch_size)
        

        if mode == "train":

            seq_len = tf.tile(tf.constant([maxlen], dtype=tf.int32), [ batch_size ] )
            #[None]/[batch_size]
            helper = tf.contrib.seq2seq.TrainingHelper( inputs = emb_Y , sequence_length = seq_len )
            decoder = tf.contrib.seq2seq.BasicDecoder( cell = cell, helper = helper, initial_state = initial_state, output_layer=out_layer) 
            outputs, final_state, _= tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=maxlen, scope=decoder_scope)
            logits = outputs.rnn_output
            sample_ids = outputs.sample_id
        else:

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(emb_out, tf.fill([batch_size], _GO), _END)
            my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state, output_layer=out_layer)

                      
            outputs, t1 , t2 = tf.contrib.seq2seq.dynamic_decode(  my_decoder,
                                                                   maximum_iterations=maxlen,scope=decoder_scope )
            logits = outputs.rnn_output
            sample_ids = outputs.sample_id
        
    return logits , sample_ids
#--------------------------------------------------------------------------------------------------------------
def construct_graph(mode,env=env):

    vocab_emb = np.load('vocab_emb_all.npy')
    emb_out = tf.get_variable( "emb_out" , initializer=vocab_emb)
    emb_X = tf.nn.embedding_lookup( emb_out , env.x ) 
    emb_Y = tf.nn.embedding_lookup( emb_out , env.y )
    #[None, 20, 300]

    with tf.name_scope("Encoder"):
        cell_fw0 = tf.contrib.rnn.GRUCell(dim)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw0, input_keep_prob=1-in_drop,output_keep_prob=1-out_drop)
        cell_bw0 = tf.contrib.rnn.GRUCell(dim)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw0, input_keep_prob=1-in_drop,output_keep_prob=1-out_drop)

        enc_rnn_out , enc_rnn_state = tf.nn.bidirectional_dynamic_rnn( cell_fw , cell_bw , emb_X , dtype=tf.float32)
        #state: (output_state_fw, output_state_bw) 
        #([None, 20, 150],[None, 20, 150])
        enc_rnn_out = tf.concat(enc_rnn_out, 2)
        #[None,20,300]
        enc_rnn_state = tf.concat([enc_rnn_state[0],enc_rnn_state[1]],axis=1)

    logits , sample_ids = Decoder(mode, enc_rnn_out , enc_rnn_state , emb_Y, emb_out)
    #shift env.y by 1 to remove _GO , and pad with _PAD
    env.pred = tf.concat( (env.y[:,1:],tf.zeros((tf.shape(env.y)[0],1), dtype=tf.int32)),axis=1)
    env.loss = tf.losses.softmax_cross_entropy(  tf.one_hot( env.pred, output_vocab_size ) , logits )
    optimizer = tf.train.AdamOptimizer(lr)
    optimizer.minimize(env.loss)
    gvs = optimizer.compute_gradients(env.loss)
    capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
    env.train_op = optimizer.apply_gradients(capped_gvs)
    env.perp = tf.exp(env.loss)
    a = tf.equal( sample_ids , env.pred )
    b = tf.reduce_all(a, axis=1)
    env.acc = tf.reduce_mean( tf.cast( b , dtype=tf.float32 ) ) 
   
    return env.train_op , env.loss , env.acc , sample_ids , logits, env.perp



def decode_data():
    ybar = sess.run(
            pred_ids,
            feed_dict={env.x: X_test})
    ybar=np.asarray(ybar)
    _,reverse_vocab_dict=util.load_vocab_all()
    print(ybar.shape)
    count=0
    for true_seq,seq in zip(y_test,ybar):
        true_seq=true_seq[1:]
        for i in range(len(true_seq)):
            if true_seq[i]==2:
                seq=seq[:i]
                true_seq=true_seq[:i]
                break
        logic=" ".join([reverse_vocab_dict[idx] for idx in seq ])
        true_logic=" ".join([reverse_vocab_dict[idx] for idx in true_seq ])
        count+=(logic==true_logic)
  
    print('count acc')
    print(count*1./len(ybar))


def decode_one(sent):
    vocab_dict,reverse_vocab_dict=util.load_vocab_all()
    x_data = [vocab_dict[x] for x in sent.split()]
    x_data.append(_END)
    x_data.extend([_PAD for x in range(maxlen-len(x_data))])
    x_data = np.asarray(x_data).reshape(1,maxlen)
    ybar = sess.run(
            pred_ids,
            feed_dict={env.x: x_data})
    ybar=np.asarray(ybar)
    _,reverse_vocab_dict=util.load_vocab_all()
    for seq in ybar:
        logic=" ".join([reverse_vocab_dict[idx] for idx in seq ])
        print(logic)
#------------------------------------------------------------------
y_train,X_train=load_data_idx(subset=subset,maxlen=maxlen,load=False)
y_test,X_test=load_data_idx(subset=subset,maxlen=maxlen,load=False,s='test')

tf.reset_default_graph()
train_graph = tf.Graph()
infer_graph = tf.Graph()

with train_graph.as_default():   
    env.x = tf.placeholder( tf.int32 , shape=[None,maxlen], name='x' )
    env.y = tf.placeholder(tf.int32, (None, maxlen), name='y')
    env.training = tf.placeholder_with_default(False, (), name='train_mode')
    env.train_op, env.loss , env.acc, sample_ids,logits, env.perp = construct_graph("train")
    env.saver = tf.train.Saver()  
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) 
    train(sess, env, X_train, y_train, epochs = epochs,load=load_model,name=subset,batch_size=BS)

with infer_graph.as_default():
    env.x = tf.placeholder( tf.int32 , shape=[None,maxlen], name='x' )
    env.y = tf.placeholder(tf.int32, (None, maxlen), name='y')
    env.training = tf.placeholder_with_default(False, (), name='train_mode')   
    _ , env.loss , env.acc , pred_ids, _ , env.perp = construct_graph("infer")
    env.infer_saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    env.infer_saver.restore(sess, "reverse_model/{}".format(subset) )
    decode_one(S)
    


