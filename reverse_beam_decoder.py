from __future__ import print_function
import sys
import os
import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils.overnight import load_data,load_data_idx,load_vocab_all
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ----------------------------------------------------------------------------
_GO = 1
_PAD = 0
_END = 2
epochs = 30
lr = 0.01
BS = 128
maxlen = 20
embedding_dim = 300
D = embedding_dim
n_states = int(D/2)
classes = 2
T = maxlen
in_drop=.0
out_drop=.0
vocabulary_size=928
embedding_size=300
subset='all'
load_model=True
input_vocab_size = vocabulary_size
output_vocab_size = vocabulary_size
S = 'bos player where ( ( number_of_assists equal 3 ) and ( season equal 2004 ) )'
dim = n_states
# ----------------------------------------------------------------------------
def evaluate(sess, env, X_data, y_data, batch_size=BS):
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
        sys.stdout.flush()
        start = batch * batch_size
        end = min(n_sample, start+batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [env.loss,env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return acc
#-----------------------------------------------------------------------------

class Dummy:
    pass
env = Dummy()



def _decoder( encoder_outputs , encoder_state , mode , beam_width , batch_size):
    
    num_units = 2*dim
    # [batch_size, max_time,...]
    memory = encoder_outputs
    
    if mode == "infer":
        memory = tf.contrib.seq2seq.tile_batch( memory, multiplier=beam_width )
        encoder_state = tf.contrib.seq2seq.tile_batch( encoder_state, multiplier=beam_width )
        batch_size = batch_size * beam_width
    else:
        batch_size = batch_size

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

            start_tokens = tf.tile(tf.constant([_GO], dtype=tf.int32), [ batch_size ] )
            end_token = _END

            my_decoder = tf.contrib.seq2seq.BeamSearchDecoder( cell = cell,
                                                               embedding = emb_out,
                                                               start_tokens = start_tokens,
                                                               end_token = end_token,
                                                               initial_state = initial_state,
                                                               beam_width = beam_width,
                                                               output_layer = out_layer )
                      
            outputs, t1 , t2 = tf.contrib.seq2seq.dynamic_decode(  my_decoder,
                                                                   maximum_iterations=maxlen,scope=decoder_scope )
            logits = tf.no_op()
            sample_ids = outputs.predicted_ids
        
    return logits , sample_ids
#-------------------------------------------------------------------------------------------
def construct_graph(mode,env=env):

    vocab_emb = np.load('vocab_emb_all.npy')
    emb_out = tf.get_variable( "emb_out" , initializer=vocab_emb)
    emb_X = tf.nn.embedding_lookup( emb_out , env.x ) 
    emb_Y = tf.nn.embedding_lookup( emb_out , env.y )

    with tf.name_scope("Encoder"):
        cell_fw0 = tf.contrib.rnn.GRUCell(dim)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw0, input_keep_prob=1-in_drop,output_keep_prob=1-out_drop)
        cell_bw0 = tf.contrib.rnn.GRUCell(dim)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw0, input_keep_prob=1-in_drop,output_keep_prob=1-out_drop)

        enc_rnn_out , enc_rnn_state = tf.nn.bidirectional_dynamic_rnn( cell_fw , cell_bw , emb_X , dtype=tf.float32)
        enc_rnn_out = tf.concat(enc_rnn_out, 2)
        enc_rnn_state = tf.concat([enc_rnn_state[0],enc_rnn_state[1]],axis=1)

    logits , sample_ids = Decoder(mode, enc_rnn_out , enc_rnn_state , emb_Y, emb_out)
    if mode == 'train':
        #shift env.y by 1 to remove _GO , and pad with _PAD
        env.pred = tf.concat( (env.y[:,1:],tf.zeros((tf.shape(env.y)[0],1), dtype=tf.int32)),axis=1)
        env.loss = tf.losses.softmax_cross_entropy(  tf.one_hot( env.pred, output_vocab_size ) , logits )
        optimizer = tf.train.AdamOptimizer(lr)
        optimizer.minimize(env.loss)
        gvs = optimizer.compute_gradients(env.loss)
        capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
        env.train_op = optimizer.apply_gradients(capped_gvs)

        a = tf.equal( sample_ids , env.pred )
        b = tf.reduce_all(a, axis=1)
        env.acc = tf.reduce_mean( tf.cast( b , dtype=tf.float32 ) ) 
    else:
        sample_ids = tf.transpose( sample_ids , [0,2,1] )#[batch_size,beam_width,sentence length]
        env.acc = None
        env.loss = None
        env.train_op = None 
        
    return env.train_op , env.loss , env.acc , sample_ids , logits



def decode_data():
    ybar = sess.run(
            pred_ids,
            feed_dict={env.x: X_test})
    ybar=np.asarray(ybar)
    _,reverse_vocab_dict=load_vocab_all()
    print(ybar.shape)
    count=0
    for true_seq,seq in zip(y_test,ybar):
        true_seq=true_seq[1:]
        for i in range(len(true_seq)):
            if true_seq[i]==2 or seq[i]==2:
                seq=seq[:i]
                true_seq=true_seq[:i]
                break
        logic=" ".join([reverse_vocab_dict[idx] for idx in seq ])
        true_logic=" ".join([reverse_vocab_dict[idx] for idx in true_seq ])
        count+=(logic==true_logic)
  
    print('count acc')
    print(count*1./len(ybar))

def decode_one(sents):
    vocab_dict,reverse_vocab_dict=load_vocab_all()
    reverse_vocab_dict[-1]='pad'
    X_data = []
    for sent in sents: 
        x_data = [vocab_dict[x] for x in sent.split()]
        x_data.append(2)
        x_data.extend([0  for x in range(maxlen-len(x_data))])
        X_data.append(x_data)

    X_data = np.asarray(X_data)
    ybar = sess.run(
            pred_ids,
            feed_dict={env.x: X_data})
    ybar=np.asarray(ybar)
    print(ybar.shape)
    for i,seq_per_beam in enumerate(ybar):
        print('=========SQL==========')
        print(X_data[i])
        print('--------beam decoding--------')
        for seq in seq_per_beam:
            logic=" ".join([reverse_vocab_dict[idx] for idx in seq ])
            print(logic)


#-----------------------------------------------------------------------
y_test,X_test=load_data_idx(subset=subset,maxlen=maxlen,load=False,s='test')

tf.reset_default_graph()
train_graph = tf.Graph()
infer_graph = tf.Graph()

with infer_graph.as_default():
    env.x = tf.placeholder( tf.int32 , shape=[None,maxlen], name='x' )
    env.y = tf.placeholder(tf.int32, (None, maxlen), name='y')
    env.training = tf.placeholder_with_default(False, (), name='train_mode')   
    _ , env.loss , env.acc , pred_ids, _ = construct_graph("infer")
    env.infer_saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    env.infer_saver.restore(sess, "reverse_model/{}".format(subset) )
    decode_one([S])
    


