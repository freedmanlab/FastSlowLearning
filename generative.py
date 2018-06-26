import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import stimulus
import AdamOpt
from parameters_RL import *

par['forward_shape'] = [900,200,100]
par['n_output'] = 2
par['n_inter'] = 100
par['n_latent'] = 50

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Model:

    def __init__(self, input_data, target_data):

        self.input_data = input_data
        self.target_data = target_data

        self.declare_variables()

        self.run_model()

        self.optimize()


    def declare_variables(self):

        self.var_dict = {}

        with tf.variable_scope('feedforward'):
            for h in range(len(par['forward_shape'])-1):
                self.var_dict['W_in{}'.format(h)] = tf.get_variable('W_in{}'.format(h), shape=[par['forward_shape'][h],par['forward_shape'][h+1]])
                self.var_dict['b_hid{}'.format(h)] = tf.get_variable('b_hid{}'.format(h), shape=[par['forward_shape'][h+1]])

            self.var_dict['W_out'] = tf.get_variable('W_out', shape=[par['forward_shape'][-1],par['n_output']])
            self.var_dict['b_out'] = tf.get_variable('b_out', shape=par['n_output'])

        with tf.variable_scope('latent_interface'):

            # Pre-latent
            self.var_dict['W_pre'] = tf.get_variable('W_pre', shape=[par['forward_shape'][-1],par['n_inter']])
            self.var_dict['W_mu_in'] = tf.get_variable('W_mu_in', shape=[par['n_inter'],par['n_latent']])
            self.var_dict['W_si_in'] = tf.get_variable('W_si_in', shape=[par['n_inter'],par['n_latent']])

            self.var_dict['b_pre'] = tf.get_variable('b_pre', shape=[1,par['n_inter']])
            self.var_dict['b_mu'] = tf.get_variable('b_mu', shape=[1,par['n_latent']])
            self.var_dict['b_si'] = tf.get_variable('b_si', shape=[1,par['n_latent']])

        with tf.variable_scope('post_latent'):
            # Latent to post-latent layer
            self.var_dict['W_lat'] = tf.get_variable('W_lat', shape=[par['n_latent'],par['n_inter']])
            self.var_dict['W_post'] = tf.get_variable('W_post', shape=[par['n_inter'],par['forward_shape'][-1]])
            self.var_dict['b_post'] = tf.get_variable('b_post', shape=[1,par['n_inter']])

            # From post-latent layer to input
            for h in range(0,len(par['forward_shape']))[::-1]:
                if h is not 0:
                    self.var_dict['W_rec{}'.format(h)] = tf.get_variable('W_rec{}'.format(h), shape=[par['forward_shape'][h],par['forward_shape'][h-1]])
                self.var_dict['b_rec{}'.format(h)] = tf.get_variable('b_rec{}'.format(h), shape=[1,par['forward_shape'][h]])


    def run_model(self):

        h_in = []
        for h in range(len(par['forward_shape'])-1):
            if len(h_in) == 0:
                inp = self.input_data
            else:
                inp = h_in[-1]

            act = inp @ self.var_dict['W_in{}'.format(h)] + self.var_dict['b_hid{}'.format(h)]
            act = tf.nn.relu(act + 0.*tf.random_normal(act.shape))
            act = tf.nn.dropout(act, 0.8)
            h_in.append(act)

        self.y = h_in[-1] @ self.var_dict['W_out'] + self.var_dict['b_out']
        self.pre = tf.nn.relu(h_in[-1] @ self.var_dict['W_pre'] + self.var_dict['b_pre'])

        self.mu = self.pre @ self.var_dict['W_mu_in'] + self.var_dict['b_mu']
        self.si = self.pre @ self.var_dict['W_si_in'] + self.var_dict['b_si']

        ### Copy from here down to include generative setup in full network
        self.latent_sample = self.mu + tf.exp(self.si)*tf.random_normal(self.si.shape)

        self.post = tf.nn.relu(self.latent_sample @ self.var_dict['W_lat'] + self.var_dict['b_post'])
        self.post = tf.nn.relu(self.mu @ self.var_dict['W_lat'] + self.var_dict['b_post'])

        h_out = []
        for h in range(len(par['forward_shape']))[::-1]:
            if len(h_out) == 0:
                inp = self.post
                W = self.var_dict['W_post']
            else:
                inp = h_out[-1]
                W = self.var_dict['W_rec{}'.format(h+1)]

            act = inp @ W + self.var_dict['b_rec{}'.format(h)]
            if h is not 0:
                act = tf.nn.relu(act + 0.*tf.random_normal(act.shape))
                act = tf.nn.dropout(act, 0.8)
                h_out.append(act)
            else:
                h_out.append(act)

        self.x_hat = h_out[-1]


    def optimize(self):

        #opt = tf.train.GradientDescentOptimizer(par['learning_rate'])
        opt = AdamOpt.AdamOpt(tf.trainable_variables(), par['learning_rate'])

        #self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y, labels=self.target_data, dim=1))

        self.task_loss = 0.*tf.reduce_mean(tf.square(self.y - self.target_data))
        self.recon_loss = 5000*tf.reduce_mean(tf.square(self.x_hat - self.input_data))
        #self.recon_loss = 1e-3*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.x_hat, labels=self.input_data))

        self.latent_loss = 1e-6 * -0.5*tf.reduce_mean(tf.reduce_sum(1+self.si-tf.square(self.mu)-tf.exp(self.si),axis=-1))

        self.total_loss = self.task_loss + self.recon_loss + self.latent_loss

        self.train_op = opt.compute_gradients(self.total_loss)



def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [par['batch_size'], par['forward_shape'][0]], 'stim')
    y = tf.placeholder(tf.float32, [par['batch_size'], par['n_output']], 'out')

    with tf.device('/gpu:0'):
        model = Model(x, y)

    stim = stimulus.MultiStimulus()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(par['n_train_batches']):

            name, inputs, neural_inputs, outputs = stim.generate_trial(0, False, False)

            feed_dict = {x:neural_inputs, y:outputs}
            _, loss, recon_loss, latent_loss, y_hat, x_hat = sess.run([model.train_op, model.task_loss, \
                model.recon_loss, model.latent_loss, model.y, model.x_hat], feed_dict=feed_dict)

            #accuracy = np.mean(np.equal(lab, np.argmax(y_hat, axis=1)))

            if i%50 == 0:
                acc = get_perf(inputs,y_hat)
                print('{} | Acc: {:.3f} | Loss: {:.3f} | Recon. Loss: {:.3f} | Latent Loss: {:.3f}'.format( \
                    i, acc, loss, recon_loss, latent_loss))
            """
            if i%1000 == 0:

                inp = np.sum(np.reshape(neural_inputs[0], [9,10,10]), axis=0)
                hat = np.sum(np.reshape(x_hat[0], [9,10,10]), axis=0)

                fig, ax = plt.subplots(1,2)
                ax[0].set_title('Actual')
                ax[0].imshow(inp, clim=[0,1])
                ax[1].set_title('Reconstructed')
                ax[1].imshow(hat, clim=[0,1])
                plt.show()
            """

    print('Complete.')


def get_perf(target, output):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when target[:,:,-1] is not 0
    """
    return np.sum(np.float32((np.absolute(target[:,0] - output[:,0]) < 0.05) * (np.absolute(target[:,1] - output[:,1]) < 0.05)))/par['batch_size']



main()
