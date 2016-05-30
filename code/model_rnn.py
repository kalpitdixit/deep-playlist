import tensorflow as tf
from get_data import get_data
from model import LanguageModel
import math
from utils import Vocab
import numpy as np
import time
import collections
import sys
import itertools
 
class Config:
    n_class     = 2 # ?
    batch_size  = 1
    embed_size  = 50
    window_size = -1 # ?
    hidden_size = 10
    reg         = 0
    max_epochs  = 100
    lr          = 1 * (1.0)
    max_steps   = -1
    model       = 'lstm'
    debug       = True

class Model_RNN(LanguageModel):
    def load_data(self):
        pair_fname  = '../lastfm_train_mappings.txt'
        lyrics_path = '../data/lyrics/train/'
    
        # X_train is a list of all examples. each examples is a 2-len list. each element is a list of words in lyrics.
        # word_counts is a dictionary that maps
        if self.config.debug:
            X_train, l_train, self.word_counts, seq_len1, seq_len2, self.config.max_steps = get_data(pair_fname, lyrics_path, '../glove.6B.50d.txt', threshold_down=0, threshold_up=float('inf'), npos=100, nneg=100)
        else:
            X_train, l_train, self.word_counts, seq_len1, seq_len2, self.config.max_steps = get_data(pair_fname, lyrics_path, threshold_down=100, threshold_up=4000, npos=10000, nneg=10000)

        self.labels_train = np.zeros((len(X_train),self.config.n_class))
        self.labels_train[range(len(X_train)),l_train] = 1
        
        x = collections.Counter(l_train)
        for k in x.keys():
            print 'class:', k, x[k]
        print ''

        self.vocab = Vocab()
        self.vocab.construct(self.word_counts.keys())
        self.wv = self.vocab.get_wv('../glove.6B.50d.txt')

        with open('word_hist.csv', 'w') as f:
            for w in self.word_counts.keys():
                f.write(w+','+str(self.word_counts[w])+'\n')
            
        self.encoded_train_1 = np.zeros((len(X_train), self.config.max_steps)) # need to handle this better. 
        self.encoded_train_2 = np.zeros((len(X_train), self.config.max_steps))
        for i in range(len(X_train)):
            self.encoded_train_1[i,:len(X_train[i][0])] = [self.vocab.encode(word) for word in X_train[i][0]]       
            self.encoded_train_2[i,:len(X_train[i][1])] = [self.vocab.encode(word) for word in X_train[i][1]]       
        self.sequence_len1 = np.array(seq_len1)
        self.sequence_len2 = np.array(seq_len2)

    def add_placeholders(self):
        self.X1            = tf.placeholder(tf.int32,   shape=(None, self.config.max_steps), name='X1')
        self.X2            = tf.placeholder(tf.int32,   shape=(None, self.config.max_steps), name='X2')
        self.labels        = tf.placeholder(tf.float32, shape=(None, self.config.n_class), name='labels')
        #self.initial_state = tf.placeholder(tf.float32, shape=(None, self.config.hidden_size), name='initial_state')
        self.seq_len1      = tf.placeholder(tf.int32,   shape=(None),                        name='seq_len1') # for variable length sequences
        self.seq_len2      = tf.placeholder(tf.int32,   shape=(None),                        name='seq_len2') # for variable length sequences

    def add_embedding(self):
        #L = tf.get_variable('L', shape=(len(self.vocab), self.config.embed_size), dtype=tf.float32) 
        L = tf.Variable(tf.convert_to_tensor(self.wv, dtype=tf.float32), name='L')
        #L = tf.constant(tf.convert_to_tensor(self.wvi), dtype=tf.float32, name='L')
        inputs1 = tf.nn.embedding_lookup(L, self.X1) # self.X1 is batch_size x self.config.max_steps 
        inputs2 = tf.nn.embedding_lookup(L, self.X2) # input2 is batch_size x self.config.max_steps x self.config.embed_size
        inputs1 = tf.split(1, self.config.max_steps, inputs1) # list of len self.config.max_steps where each element is batch_size x self.config.embed_size
        inputs1 = [tf.squeeze(x, squeeze_dims=[1]) for x in inputs1]
        inputs2 = tf.split(1, self.config.max_steps, inputs2) # list of len self.config.max_steps where each element is batch_size x self.config.embed_size
        inputs2 = [tf.squeeze(x, squeeze_dims=[1]) for x in inputs2]
        return inputs1, inputs2

    def add_model_rnn(self, inputs1, inputs2, seq_len1, seq_len2):
        #self.initial_state = tf.constant(np.zeros(()), dtype=tf.float32)
        self.initial_state = tf.constant(np.zeros((self.config.batch_size,self.config.hidden_size)), dtype=tf.float32)
        rnn_outputs  = []
        rnn_outputs1 = []
        rnn_outputs2 = []
        h_curr1 = self.initial_state
        h_curr2 = self.initial_state

        with tf.variable_scope('rnn'):
            Whh = tf.get_variable('Whh', shape=(self.config.hidden_size,self.config.hidden_size), dtype=tf.float32)
            Wxh = tf.get_variable('Wxh', shape=(self.config.embed_size,self.config.hidden_size),  dtype=tf.float32)
            b1  = tf.get_variable('bhx', shape=(4*self.config.hidden_size,),                        dtype=tf.float32)

            for i in range(self.config.max_steps):
                if self.config.batch_size==1:
                    if i==seq_len1[0]:
                        breaka
                tmp = tf.matmul(h_curr1,Whh) + tf.matmul(inputs1[i],Wxh) + b1
                
                rnn_outputs1.append(h_curr1)

            for i in range(self.config.max_steps):
                if self.config.batch_size==1:
                    if i==seq_len2[0]:
                        breaka
                h_curr2 = tf.sigmoid(tf.matmul(h_curr2,Whh) + tf.matmul(inputs2[i],Wxh) + b1)
                rnn_outputs2.append(h_curr2)

        #lstm_states = [tf.concat(1, [rnn_outputs1[i], rnn_outputs2[i]]) for i in range(self.config.max_steps)]
        rnn_final_states = tf.concat(1, [rnn_outputs1[-1], rnn_outputs2[-1]])
        return rnn_final_states

    def add_model_lstm(self, inputs1, inputs2, seq_len1, seq_len2):
        #self.initial_state = tf.constant(np.zeros(()), dtype=tf.float32)
        self.initial_state = tf.constant(np.zeros((self.config.batch_size,self.config.hidden_size)), dtype=tf.float32)
        lstm_outputs1 = []
        lstm_outputs2 = []
        h_curr1 = self.initial_state
        h_curr2 = self.initial_state
        cell1   = self.initial_state
        cell2   = self.initial_state

        with tf.variable_scope('lstm'):
            Whc = tf.get_variable('Whh', shape=(self.config.hidden_size,4*self.config.hidden_size), dtype=tf.float32, initializer=tf.random_normal_initializer())
            Wxc = tf.get_variable('Wxh', shape=(self.config.embed_size,4*self.config.hidden_size),  dtype=tf.float32, initializer=tf.random_normal_initializer())
            b1  = tf.get_variable('bhx', shape=(self.config.hidden_size,),                          dtype=tf.float32, initializer=tf.random_normal_initializer())

            for i in range(self.config.max_steps):
                if self.config.batch_size==1:
                    if i==seq_len1[0]:
                        break
                ifog1 = tf.matmul(h_curr1,Whc) + tf.matmul(inputs1[i],Wxc)
                i1, f1, o1, g1 = tf.split(1, 4, ifog1)
                i1 = tf.sigmoid(i1)
                f1 = tf.sigmoid(f1)
                o1 = tf.sigmoid(o1)
                g1 = tf.tanh(g1)

                cell1   = f1*cell1 + i1*g1
                h_curr1 = o1*tf.tanh(cell1)
                lstm_outputs1.append(h_curr1)

            for i in range(self.config.max_steps):
                if self.config.batch_size==1:
                    if i==seq_len2[0]:
                        break
                ifog2 = tf.matmul(h_curr2,Whc) + tf.matmul(inputs2[i],Wxc)
                i2, f2, o2, g2 = tf.split(1, 4, ifog2)
                i2 = tf.sigmoid(i2)
                f2 = tf.sigmoid(f2)
                o2 = tf.sigmoid(o2)
                g2 = tf.tanh(g2)

                cell2   = f2*cell2 + i2*g2
                h_curr2 = o2*tf.tanh(cell2)
                lstm_outputs2.append(h_curr2)

        lstm_final_states = tf.concat(1, [lstm_outputs1[-1], lstm_outputs2[-1]])
        return lstm_final_states

    def add_final_projections(self, rnn_final_states):
        # rnn_outputs is a list of length batch_size of lengths = seq_len. Where each list element is ??. I think.
        Whu = tf.get_variable('Whu', shape=(2*self.config.hidden_size,self.config.n_class), initializer=tf.random_normal_initializer())
        bhu = tf.get_variable('bhu', shape=(self.config.n_class,), initializer=tf.random_normal_initializer())
        final_projections = tf.matmul(rnn_final_states,Whu) + bhu # in case we stop short sequences, the rnn_state in further time_steps should be unch
        return final_projections

    def add_loss_op(self, y):
        loss = tf.nn.softmax_cross_entropy_with_logits(y, self.labels)
        loss = tf.reduce_mean(loss)
        return loss
      
    def add_training_op(self, loss):
        #train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        train_op = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()

        self.inputs1, self.inputs2 = self.add_embedding()
        if self.config.model=='rnn':
            self.final_hidden_states = self.add_model_rnn(self.inputs1, self.inputs2, self.seq_len1, self.seq_len2)
        elif self.config.model=='lstm':
            self.final_hidden_states = self.add_model_lstm(self.inputs1, self.inputs2, self.seq_len1, self.seq_len2)
        self.final_projections     = self.add_final_projections(self.final_hidden_states)
        self.loss                  = self.add_loss_op(self.final_projections)
        self.train_step            = self.add_training_op(self.loss)
        self.predictions           = tf.argmax(tf.nn.softmax(self.final_projections),1)
        self.correct_predictions   = tf.equal(self.predictions,tf.argmax(self.labels,1))
        self.correct_predictions   = tf.reduce_sum(tf.cast(self.correct_predictions, 'int32'))

    def run_epoch(self, session, X1, X2, labels, sequence_len1, sequence_len2, train_op, verbose=10): # X and y are 2D np arrays
        config = self.config
        #state = tf.zeros([self.config.batch_size, self.config.hidden_size])
        state = self.initial_state.eval()
        data_len = np.shape(X1)[0]
        index = np.arange(data_len)
        np.random.shuffle(index)
        n_batches  = data_len // self.config.batch_size
        
        loss = 0.0
        total_loss = []
        total_correct = 0
        all_preds = -np.ones((data_len,))
        for batch_num in range(n_batches):
            x1_batch = X1[index[batch_num * self.config.batch_size : (batch_num+1) * self.config.batch_size], :]
            x2_batch = X2[index[batch_num * self.config.batch_size : (batch_num+1) * self.config.batch_size], :]
            labels_batch = labels[index[batch_num * self.config.batch_size : (batch_num+1) * self.config.batch_size], :]
            seq_len_batch1 = sequence_len1[index[batch_num * self.config.batch_size : (batch_num+1) * self.config.batch_size]]
            seq_len_batch2 = sequence_len2[index[batch_num * self.config.batch_size : (batch_num+1) * self.config.batch_size]]
            feed_dict = {self.X1: x1_batch,
                         self.X2: x2_batch,
                         self.labels: labels_batch,
                         self.seq_len1: seq_len_batch1, 
                         self.seq_len2: seq_len_batch2} 
                         #self.initial_state: state}
            
            loss, preds, correct, final_projections, _ = session.run([self.loss, self.predictions, self.correct_predictions, self.final_projections, train_op], feed_dict=feed_dict)
            #print str(batch_num)+'/'+str(n_batches)+' : '+str(final_projections[0][0])+'  '+str(final_projections[0][1])
            total_loss.append(loss)
            total_correct += correct
            all_preds[index[batch_num * self.config.batch_size : (batch_num+1) * self.config.batch_size]] = preds

            if verbose and (batch_num+1)%verbose==0:
                sys.stdout.write('\r{} / {} : loss = {:.4f} : train_acc = {:.2f}%'.format(batch_num+1, n_batches, np.mean(total_loss), 100.0*total_correct/((batch_num+1)*self.config.batch_size)))
                sys.stdout.flush()
            if verbose:
                sys.stdout.write('\r')
            
        return np.mean(total_loss), all_preds

def make_conf(y,yhat):
    confmat = np.zeros([2,2])
    for i in range(len(y)):
        confmat[y[i],yhat[i]] += 1
    return confmat

def run():
    config = Config()

    # We create the training model and generative model
    with tf.variable_scope('Model_RNN') as scope:
        model = Model_RNN(config)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0
  
        session.run(init)
    
        prev_loss = float('inf')
        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()
            ###
            train_loss, all_preds = model.run_epoch(
                session, model.encoded_train_1, model.encoded_train_2, model.labels_train, model.sequence_len1, model.sequence_len2,
                train_op=model.train_step)
            #valid_pp = model.run_epoch(session, model.encoded_valid)
            print 'Training Loss: {:.4f}'.format(train_loss)
            if train_loss > prev_loss:
                model.config.lr /= 2
                print 'Loss diverging, decreasing learning rate to', model.config.lr
            prev_loss = train_loss
            #print 'Validation perplexity: {}'.format(valid_pp)
            #if valid_pp < best_val_pp:
                #best_val_pp = valid_pp
                #best_val_epoch = epoch
                #saver.save(session, './ptb_rnnlm.weights')
            #if epoch - best_val_epoch > config.early_stopping:
                #break
            print make_conf(np.argmax(model.labels_train,axis=1), all_preds)
            print 'Total time: {:.2f}'.format(time.time() - start)
            print ''

if __name__=='__main__':
    run()
