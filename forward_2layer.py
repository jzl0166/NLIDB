from __future__ import print_function
import sys
import os
import keras
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils.both import load_data,load_vocab_all
from utils.bleu import moses_multi_bleu
from collections import defaultdict
import sys
reload(sys)
sys.setdefaultencoding('utf8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ----------------------------------------------------------------------------
'''
TODO:
l2_scale regularizer
'''
_PAD = 0
_GO = 1
_END = 2
epochs = 2
lr = 0.0001
BS = 128
maxlen = 60
embedding_dim = 300
D = embedding_dim
#dim = int(D/2)
dim = 200
T = maxlen
in_drop=.0
out_drop=.0
vocabulary_size=20637
embedding_size=300
subset='all'
load_model=False
input_vocab_size = vocabulary_size
output_vocab_size = vocabulary_size
# ----------------------------------------------------------------------------
def train(sess, env, X_data, y_data, epochs=10, load=False, shuffle=True, batch_size=BS,
          name='model',base=0):
    if load:
        print('\nLoading saved model')
        env.saver.restore(sess, model2Bload )

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

        if (epoch+1)==epochs:
            print('\n Saving model')
            env.saver.save(sess, 'model/{0}-{1}'.format(name,base))
    return 'model/{0}-{1}'.format(name,base) 

def evaluate(sess, env, X_data, y_data, batch_size=BS):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
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

#---------------------------------------------------------------------------
class Dummy:
    pass
env = Dummy()

from tf_utils.attention_wrapper import AttentionWrapper,BahdanauAttention
from tf_utils.beam_search_decoder import BeamSearchDecoder
from tf_utils.decoder import dynamic_decode
from tf_utils.basic_decoder import BasicDecoder
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
    attention_mechanism = BahdanauAttention( num_units = num_units, memory=memory, 
                                                               normalize=True,
                                                               memory_sequence_length=seq_len)

    cell0 = tf.contrib.rnn.GRUCell( 2*dim )
    cell = tf.contrib.rnn.DropoutWrapper(cell0, input_keep_prob=1-in_drop,output_keep_prob=1-out_drop)
    cell = AttentionWrapper( cell,
                                                attention_mechanism,
                                                attention_layer_size=num_units,
                                                name="attention")

    decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone( cell_state=encoder_state )

    return cell, decoder_initial_state


def Decoder( mode , enc_rnn_out , enc_rnn_state , X,  emb_Y , emb_out):
    
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
            decoder = BasicDecoder( cell = cell, helper = helper, initial_state = initial_state,X=X, output_layer=out_layer) 
            outputs, final_state, _= tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=maxlen, scope=decoder_scope)
            logits = outputs.rnn_output
            sample_ids = outputs.sample_id
        else:

            start_tokens = tf.tile(tf.constant([_GO], dtype=tf.int32), [ batch_size ] )
            end_token = _END

            my_decoder = BeamSearchDecoder( cell = cell,
                                                               embedding = emb_out,
                                                               start_tokens = start_tokens,
                                                               end_token = end_token,
                                                               initial_state = initial_state,
                                                               beam_width = beam_width,
                                                               X = X,
                                                               output_layer = out_layer ,
                                                               length_penalty_weight=0.0 )
                      
            outputs, t1 , t2 = tf.contrib.seq2seq.dynamic_decode(  my_decoder, maximum_iterations=maxlen,scope=decoder_scope )
            logits = tf.no_op()
            sample_ids = outputs.predicted_ids
        
    return logits , sample_ids

#----------------------------------------------------------------------------------------------
def construct_graph(mode,env=env):

    _, _, vocab_emb, train_idx = load_vocab_all()
    print(train_idx)
    print('Vocab size:')
    print(vocab_emb.shape)
    emb_out = tf.get_variable( "emb_out" , initializer=vocab_emb)
    emb_X = tf.nn.embedding_lookup(emb_out, env.x) 
    emb_Y = tf.nn.embedding_lookup(emb_out, env.y)

    with tf.name_scope("Encoder"):
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            cell_fw = tf.contrib.rnn.GRUCell(dim)
            cell_bw = tf.contrib.rnn.GRUCell(dim)
            input_layer = Dense(dim, dtype=tf.float32, name='input_projection') 
            emb_X = input_layer(emb_X)
            enc_rnn_out, enc_rnn_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, emb_X, dtype=tf.float32)
            enc_rnn_out = tf.concat(enc_rnn_out, 2)
        with tf.variable_scope("bidirectional-rnn1"):
            cell_fw1 = tf.contrib.rnn.GRUCell(dim)
            cell_bw1 = tf.contrib.rnn.GRUCell(dim)
            input_layer1 = Dense(dim, dtype=tf.float32, name='input_projection1')
            enc_rnn_out = input_layer1(enc_rnn_out)
            enc_rnn_out, enc_rnn_state = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, enc_rnn_out, dtype=tf.float32)
            enc_rnn_out = tf.concat(enc_rnn_out, 2)
        enc_rnn_state = tf.concat([enc_rnn_state[0],enc_rnn_state[1]],axis=1)

    logits , sample_ids = Decoder(mode, enc_rnn_out , enc_rnn_state , env.x , emb_Y, emb_out)
    if mode == 'train':
        env.pred = tf.concat( (env.y[:,1:],tf.zeros((tf.shape(env.y)[0],1), dtype=tf.int32)),axis=1)
        env.loss = tf.losses.softmax_cross_entropy(  tf.one_hot( env.pred, output_vocab_size ) , logits )
        optimizer = tf.train.AdamOptimizer(lr)
        optimizer.minimize(env.loss)
        gvs = optimizer.compute_gradients(env.loss)
        
        #train_idx = np.arange(17)
        train_idx_tensor = tf.constant(train_idx, dtype = tf.int32)
        #m = tf.ones( shape=[output_vocab_size,embedding_dim] )
        n = tf.sparse_to_dense(train_idx_tensor, tf.stack([output_vocab_size]) , sparse_values=1.0, default_value=0.0, validate_indices=False )
        #mask = tf.transpose( tf.transpose(m) * n )
        mask = tf.reshape(n, [output_vocab_size, 1])
        #capped_gvs = [(tf.clip_by_norm(grad, 5.), var) if var.name != 'emb_out:0' else (tf.clip_by_norm(tf.multiply(grad,mask), 5.), var)  for grad, var in gvs]
        capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
        env.train_op = optimizer.apply_gradients(capped_gvs)
        a = tf.equal( sample_ids , env.pred )
        b = tf.reduce_all(a, axis=1)
        env.acc = tf.reduce_mean( tf.cast( b , dtype=tf.float32 ) ) 
    else:
	    #[None,sentence length,beam_width]
	    sample_ids = tf.transpose( sample_ids , [0,2,1] )
	    #[None,beam_width,sentence length]
	    env.acc = None
	    env.loss = None
	    env.train_op = None 
        
    return env.train_op , env.loss , env.acc , sample_ids , logits


def decode_data_recover(sess, X_data, y_data, s, batch_size = BS):
    print('\nDecoding and Evaluate recovered EM acc')
    n_sample = X_data.shape[0]
    sample_ids = np.random.choice(n_sample, 100)
    n_batch = int((n_sample+batch_size-1) / batch_size)
    acc = 0
    more_acc = 0
    full_anno_acc = 0
    no_anno_acc = 0
    true_values , values = [], []
    _,reverse_vocab_dict,_,_=load_vocab_all()
    inf_logics = []
    i = 0
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
        sys.stdout.flush()
        start = batch * batch_size
        end = min(n_sample, start+batch_size)
        cnt = end - start
        ybar = sess.run(
            pred_ids,
            feed_dict={env.x: X_data[start:end]})
        ybar = np.asarray(ybar)
        ybar = np.squeeze(ybar[:,0,:])
        for seq in ybar:
            try:
                seq=seq[:list(seq).index(2)]
            except ValueError:
                pass  
            logic=" ".join([reverse_vocab_dict[idx] for idx in seq ])
            inf_logics.append(logic) 
     
    xtru = X_data
    ytru = y_data
    with gfile.GFile('%s_infer.txt'%s, mode='w') as output, gfile.GFile('%s_ground_truth.txt'%s, mode='r') as S_ori_file, gfile.GFile('%s_sym_pairs.txt'%s, mode='r') as sym_pair_file:

        sym_pairs = sym_pair_file.readlines()
        S_oris = S_ori_file.readlines()
        for true_seq, logic, x, sym_pair, S_ori in zip(ytru, inf_logics, xtru, sym_pairs, S_oris):
            sym_pair = sym_pair.replace('<>\n','')
            S_ori = S_ori.replace('\n','')
            Qpairs = []
            for pair in sym_pair.split('<>'):
                Qpairs.append(pair.split('=>'))
            true_seq = true_seq[1:]
            x = x[1:]
            try:
                true_seq=true_seq[:list(true_seq).index(2)]
            except ValueError:
                pass

            try:
                x=x[:list(x).index(2)]
            except ValueError:
                pass
            

            xseq = " ".join([reverse_vocab_dict[idx] for idx in x ])
            true_logic=" ".join([reverse_vocab_dict[idx] for idx in true_seq ])

            logic = logic.replace(' (','').replace(' )','')
            true_logic = true_logic.replace(' (','').replace(' )','') 

            full_annotate = True
            for s in true_logic.split():
                if s[:2]!='<f' and s[:2]!='<v' and s!='<eof>' and s not in ['(',')','where','less','greater','equal','max','min','count','sum','avg','and','true']:
                    full_annotate = False
                    break
            logic_tokens = logic.split()
            WRITE = True
            if len(logic_tokens) > 8 and logic_tokens[5] == 'and':
                newlogic = [x for x in logic_tokens]
                newlogic[2] = logic_tokens[6]
                newlogic[6] = logic_tokens[2]
                newlogic[4] = logic_tokens[8]
                newlogic[8] = logic_tokens[4]
                newline = ' '.join(newlogic)
                if newline == true_logic:
                    logic = newline
                    more_acc += 1
                    WRITE = False 
            elif len(logic_tokens) > 9 and logic_tokens[6] == 'and':
                newlogic = [x for x in logic_tokens]
                newlogic[3] = logic_tokens[7]
                newlogic[7] = logic_tokens[3]
                newlogic[5] = logic_tokens[9]
                newlogic[9] = logic_tokens[5]
                newline = ' '.join(newlogic)
                if newline == true_logic:
                    logic = newline
                    more_acc += 1
                    WRITE = False 

            recover_S = logic
            for sym,word in Qpairs:
                recover_S = recover_S.replace(sym,word) 
            if recover_S!=S_ori and logic==true_logic:
                print(xseq)
                print(Qpairs)
                print(recover_S)
                print(S_ori)
                print('infer logic:')
                print(logic)
                print('true logic:')
                print(true_logic)

            acc+=(recover_S==S_ori)
            output.write(recover_S + '\n')
            
            full_anno_acc += (full_annotate and ( logic==true_logic or not WRITE ) )
            no_anno_acc += (not full_annotate and ( logic==true_logic or not WRITE ) )
            '''
            if logic != true_logic and WRITE and full_annotate:
                output.write('----------'+str(i)+': SQL/True/Inference\n')
                output.write(xseq+'\n')
                output.write(true_logic+'\n')
                output.write(logic+'\n')
            '''
            i += 1
            true_values.append(true_logic)
            values.append(logic)        
    
    print('EM: %.4f'%(acc*1./len(y_data)))  
    print('fully annotation EM acc: %.4f'%(full_anno_acc*1./len(y_data)))
    print('not fully annotation EM acc: %.4f'%(no_anno_acc*1./len(y_data)))
    print('number of correct ones:%d'%acc)
    
    true_values, values= np.asarray(true_values), np.asarray(values)
    bleu_score = moses_multi_bleu(true_values,values)
    print('BLEU score:%.4f'%bleu_score)
    return acc*1./len(y_data)

def decode_data(sess, X_data, y_data , batch_size = BS):
    print('\nDecoding')
    n_sample = X_data.shape[0]
    sample_ids = np.random.choice(n_sample, 100)
    n_batch = int((n_sample+batch_size-1) / batch_size)
    acc = 0
    more_acc = 0
    full_anno_acc = 0
    no_anno_acc = 0
    true_values , values = [], []
    _,reverse_vocab_dict,_,_=load_vocab_all()
    with gfile.GFile('output.txt', mode='w') as output:
        i = 0
        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch+1, n_batch),end='\r')
            sys.stdout.flush()
            start = batch * batch_size
            end = min(n_sample, start+batch_size)
            cnt = end - start
            ybar = sess.run(
                pred_ids,
                feed_dict={env.x: X_data[start:end]})
            xtru = X_data[start:end]
            ytru = y_data[start:end]
            ybar = np.asarray(ybar)
            ybar = np.squeeze(ybar[:,0,:])
            #print(ybar.shape)
            for true_seq,seq,x in zip(ytru, ybar, xtru):
                true_seq = true_seq[1:]
                x = x[1:]
                try:
                    true_seq=true_seq[:list(true_seq).index(2)]
                except ValueError:
                    pass

                try:
                    seq=seq[:list(seq).index(2)]
                except ValueError:
                    pass
                
                try:
                    x=x[:list(x).index(2)]
                except ValueError:
                    pass
                
                xseq = " ".join([reverse_vocab_dict[idx] for idx in x ])
                logic=" ".join([reverse_vocab_dict[idx] for idx in seq ])
                true_logic=" ".join([reverse_vocab_dict[idx] for idx in true_seq ])

                logic = logic.replace(' (','').replace(' )','')
                true_logic = true_logic.replace(' (','').replace(' )','') 
                full_annotate = True
                for s in true_logic.split():
                    if s[:2]!='<f' and s[:2]!='<v' and s!='<eof>' and s not in ['(',')','where','less','greater','equal','max','min','count','sum','avg','and','true']:
                        full_annotate = False
                        break
                logic_tokens = logic.split()
                WRITE = True
                if len(logic_tokens) > 8 and logic_tokens[5] == 'and':
                    newlogic = [x for x in logic_tokens]
                    newlogic[2] = logic_tokens[6]
                    newlogic[6] = logic_tokens[2]
                    newlogic[4] = logic_tokens[8]
                    newlogic[8] = logic_tokens[4]
                    newline = ' '.join(newlogic)
                    if newline == true_logic:
                        more_acc += 1
                        WRITE = False 
                elif len(logic_tokens) > 9 and logic_tokens[6] == 'and':
                    newlogic = [x for x in logic_tokens]
                    newlogic[3] = logic_tokens[7]
                    newlogic[7] = logic_tokens[3]
                    newlogic[5] = logic_tokens[9]
                    newlogic[9] = logic_tokens[5]
                    newline = ' '.join(newlogic)
                    if newline == true_logic:
                        more_acc += 1
                        WRITE = False 
                acc+=(logic==true_logic)
                full_anno_acc += (full_annotate and ( logic==true_logic or not WRITE ) )
                no_anno_acc += (not full_annotate and ( logic==true_logic or not WRITE ) )
                if logic != true_logic and WRITE and full_annotate:
                    output.write('----------'+str(i)+': SQL/True/Inference\n')
                    output.write(xseq+'\n')
                    output.write(true_logic+'\n')
                    output.write(logic+'\n')
                i += 1
                true_values.append(true_logic)
                values.append(logic)        
    acc += more_acc
    print('EM: %.4f'%(acc*1./len(y_data)))  
    print('fully annotation EM acc: %.4f'%(full_anno_acc*1./len(y_data)))
    print('not fully annotation EM acc: %.4f'%(no_anno_acc*1./len(y_data)))
    print('number of correct ones:%d'%acc)
    
    true_values, values= np.asarray(true_values), np.asarray(values)
    bleu_score = moses_multi_bleu(true_values,values)
    print('BLEU score:%.4f'%bleu_score)
    return acc*1./len(y_data) 
#----------------------------------------------------------------------
X_train, y_train = load_data(maxlen=maxlen,load=True,s='train')
X_test, y_test = load_data(maxlen=maxlen,load=True,s='test')
X_dev, y_dev = load_data(maxlen=maxlen,load=True,s='dev')
X_tran, y_tran = load_data(maxlen=maxlen,load=True,s='overnight')
model2Bload = 'model/{}'.format(subset)
max_em, global_test_em, global_tran_em, best_base = -1, -1, -1, -1
for base in range(20):
        print('~~~~~~~~~~~~~~~~~%d~~~~~~~~~~~~~~~~~~~~~'%base)
        tf.reset_default_graph()
        train_graph = tf.Graph()
        infer_graph = tf.Graph()
        with train_graph.as_default():
            [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
            env.x = tf.placeholder( tf.int32 , shape=[None,maxlen], name='x' )
            env.y = tf.placeholder(tf.int32, (None, maxlen), name='y')
            env.training = tf.placeholder_with_default(False, (), name='train_mode')
            env.train_op, env.loss , env.acc, sample_ids,logits = construct_graph("train")
            env.saver = tf.train.Saver()

            sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            model2Bload = train(sess, env, X_train, y_train, epochs = epochs, load=load_model, name=subset, batch_size=BS, base=base)
            load_model = True
        with infer_graph.as_default():
            env.x = tf.placeholder( tf.int32 , shape=[None,maxlen], name='x' )
            env.y = tf.placeholder(tf.int32, (None, maxlen), name='y')
            env.training = tf.placeholder_with_default(False, (), name='train_mode')   
            _ , env.loss , env.acc , pred_ids, _ = construct_graph("infer")
            env.infer_saver = tf.train.Saver()

            sess = tf.InteractiveSession()
            env.infer_saver.restore(sess, model2Bload )
            #decode_data(sess, X_train, y_train)
            print('===========dev set============')
            decode_data(sess, X_dev, y_dev)
            em = decode_data_recover(sess, X_dev, y_dev,'dev')
            print('==========test set===========')
            decode_data(sess, X_test, y_test)
            test_em = decode_data_recover(sess, X_test, y_test,'test')
             
            print('=========transfer set=========')
            tran_em = decode_data(sess, X_tran, y_tran)
            if em > max_em:
                max_em = em
                global_test_em = test_em
                global_tran_em = tran_em
                best_base = base
            print('Max EM acc: %.4f during %d iteration.'%(max_em, best_base)) 
            print('test EM acc: %.4f '%global_test_em)
            print('transfer EM acc %.4f '%global_tran_em)
            #decode_data_recover(sess, X_tran, y_tran,'tran')
            
                      


