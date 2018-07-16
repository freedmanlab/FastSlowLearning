import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import stimulus
import AdamOpt
from parameters_RL import *
from scipy.stats import pearsonr
import pickle
from analysis import *

par['n_latent'] = 6

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Model:

    def __init__(self, input_data, target_data, alpha):

        self.input_data = input_data
        self.target_data = target_data
        self.alpha = alpha

        self.declare_variables()

        self.run_model()

        self.optimize()


    def declare_variables(self):

        self.var_dict = {}

        # with tf.variable_scope('feedforward'):
        #     for h in range(len(par['forward_shape'])-1):
        #         self.var_dict['W_in{}'.format(h)] = tf.get_variable('W_in{}'.format(h), shape=[par['forward_shape'][h],par['forward_shape'][h+1]])
        #         self.var_dict['b_hid{}'.format(h)] = tf.get_variable('b_hid{}'.format(h), shape=[par['forward_shape'][h+1]])

            # self.var_dict['W_out'] = tf.get_variable('W_out', shape=[par['forward_shape'][-1],par['n_output']])
            # self.var_dict['b_out'] = tf.get_variable('b_out', shape=par['n_output'])

        with tf.variable_scope('latent_interface'):

            # Pre-latent
            # self.var_dict['W_pre'] = tf.get_variable('W_pre', shape=[par['forward_shape'][-1],par['n_inter']])
            self.var_dict['W_mu_in'] = tf.get_variable('W_mu_in', shape=[par['n_input'],par['n_latent']])
            self.var_dict['W_si_in'] = tf.get_variable('W_si_in', shape=[par['n_input'],par['n_latent']])

            # self.var_dict['b_pre'] = tf.get_variable('b_pre', shape=[1,par['n_inter']])
            self.var_dict['b_mu'] = tf.get_variable('b_mu', shape=[1,par['n_latent']])
            self.var_dict['b_si'] = tf.get_variable('b_si', initializer=-10*tf.ones(shape=[1,par['n_latent']]))

        # with tf.variable_scope('post_latent'):
        #     # Latent to post-latent layer
        #     self.var_dict['W_lat'] = tf.get_variable('W_lat', shape=[par['n_latent'],par['n_inter']])
        #     self.var_dict['W_post'] = tf.get_variable('W_post', shape=[par['n_inter'],par['forward_shape'][-1]])
        #     self.var_dict['b_post'] = tf.get_variable('b_post', shape=[1,par['n_inter']])

        #     # From post-latent layer to input
        #     for h in range(0,len(par['forward_shape']))[::-1]:
        #         if h is not 0:
        #             self.var_dict['W_rec{}'.format(h)] = tf.get_variable('W_rec{}'.format(h), shape=[par['forward_shape'][h],par['forward_shape'][h-1]])
        #         self.var_dict['b_rec{}'.format(h)] = tf.get_variable('b_rec{}'.format(h), shape=[1,par['forward_shape'][h]])

        with tf.variable_scope('out'):
            self.var_dict['W_out'] = tf.get_variable('W_out', shape=[par['n_latent'],par['n_input']])
            self.var_dict['b_out'] = tf.get_variable('b_out', shape=[1,par['n_input']])

        with tf.variable_scope('task'):
            # self.var_dict['W_layer_in'] = tf.get_variable('W_layer_in', shape=[par['n_latent'], par['n_layer']])
            # self.var_dict['b_layer'] = tf.get_variable('b_layer', shape=[1,par['n_layer']])
            self.var_dict['W_layer_out'] = tf.get_variable('W_layer_out', shape=[par['n_latent'],par['n_output']])
            self.var_dict['b_layer_out'] = tf.get_variable('b_layer_out', shape=[1,par['n_output']])

    def run_model(self):

        # h_in = []
        # for h in range(len(par['forward_shape'])-1):
        #     if len(h_in) == 0:
        #         inp = self.input_data
        #     else:
        #         inp = h_in[-1]

        #     act = inp @ self.var_dict['W_in{}'.format(h)] + self.var_dict['b_hid{}'.format(h)]
        #     act = tf.nn.dropout(tf.nn.relu(act + 0.*tf.random_normal(act.shape)), 1)
        #     #act = tf.nn.dropout(act, 0.8)
        #     h_in.append(act)

        # self.y = h_in[-1] @ self.var_dict['W_out'] + self.var_dict['b_out']
        self.pre = self.input_data #tf.nn.dropout(tf.nn.relu(h_in[-1] @ self.var_dict['W_pre'] + self.var_dict['b_pre']), 1)

        self.mu = self.pre @ self.var_dict['W_mu_in'] + self.var_dict['b_mu']
        self.si = self.pre @ self.var_dict['W_si_in'] + self.var_dict['b_si']

        ### Copy from here down to include generative setup in full network
        self.latent_sample = self.mu + tf.exp(0.5*self.si)*tf.random_normal(self.si.shape)
        # self.layer = self.latent_sample @ self.var_dict['W_layer_in'] + self.var_dict['b_layer']
        self.x_hat = self.latent_sample @ self.var_dict['W_out'] + self.var_dict['b_out']
        
        self.y = self.latent_sample @ self.var_dict['W_layer_out'] + self.var_dict['b_layer_out']

        # self.post = tf.nn.dropout(tf.nn.relu(self.latent_sample @ self.var_dict['W_lat'] + self.var_dict['b_post']), 1)
        # #self.post = tf.nn.relu(self.mu @ self.var_dict['W_lat'] + self.var_dict['b_post'])

        # h_out = []
        # for h in range(len(par['forward_shape']))[::-1]:
        #     if len(h_out) == 0:
        #         inp = self.post
        #         W = self.var_dict['W_post']
        #     else:
        #         inp = h_out[-1]
        #         W = self.var_dict['W_rec{}'.format(h+1)]

        #     act = inp @ W + self.var_dict['b_rec{}'.format(h)]
        #     if h is not 0:
        #         act = tf.nn.dropout(tf.nn.relu(act + 0.*tf.random_normal(act.shape)), 1)
        #         #act = tf.nn.dropout(act, 0.8)
        #         h_out.append(act)
        #     else:
        #         h_out.append(act)

        # self.x_hat = h_out[-1]


    def optimize(self):

        #opt = tf.train.GradientDescentOptimizer(par['learning_rate'])
        opt = AdamOpt.AdamOpt(tf.trainable_variables(), par['learning_rate'])

        #self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y, labels=self.target_data, dim=1))

        self.task_loss = self.alpha*tf.reduce_mean(tf.square(self.y - self.target_data))
        self.recon_loss = 1*tf.reduce_mean(tf.square(self.x_hat - self.input_data))
        #self.recon_loss = 1e-3*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.x_hat, labels=self.input_data))

        self.latent_loss = 8e-5 * -0.5*tf.reduce_mean(tf.reduce_sum(1+self.si-tf.square(self.mu)-tf.exp(self.si),axis=-1))

        self.total_loss = self.task_loss + self.recon_loss + self.latent_loss

        self.train_op = opt.compute_gradients(self.total_loss)


        self.generative_vars = {}
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='post_latent'):
            self.generative_vars[var.op.name] = var



def main():

    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [par['batch_size'], par['forward_shape'][0]], 'stim')
    y = tf.placeholder(tf.float32, [par['batch_size'], par['n_output']], 'out')
    alpha = tf.placeholder(tf.float32, [], 'alpha')

    #with tf.device('/gpu:0'):
    #    model = Model(x, y)
    model = Model(x, y, alpha)

    stim = stimulus.MultiStimulus()

    iteration = []
    accuracy = []

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(par['n_train_batches_gen']):

            if i%4 != 0:
                par['subset_loc'] = True
                alpha_val = 0.05
            else:
                par['subset_loc'] = False
                alpha_val = 0
            name, inputs, neural_inputs, outputs = stim.generate_trial(0, par['subset_dirs'], par['subset_loc'])

            feed_dict = {x:neural_inputs, y:outputs, alpha:alpha_val}
            _, loss, recon_loss, latent_loss, task_loss, y_hat, x_hat, mu, sigma, latent_sample = sess.run([model.train_op, model.task_loss, \
                model.recon_loss, model.latent_loss, model.task_loss, model.y, model.x_hat, model.mu, model.si, model.latent_sample], feed_dict=feed_dict)


            if i%120 == 0 or i%120 == 1 or i%120 == 2 or i%120 == 3:
                ind = np.intersect1d(np.argwhere(inputs[:,3]==1), np.argwhere(inputs[:,4]==0))
                acc = get_perf(outputs[ind],y_hat[ind])
                print('{} | Reconstr. Loss: {:.3f} | Latent Loss: {:.3f} | Task Loss: {:.3f} | Accuracy: {:.3f} | <Sig>: {:.3f} +/- {:.3f}'.format( \
                    i, recon_loss, latent_loss, task_loss, acc, np.mean(sigma), np.std(sigma)))
                if i%120 == 0:
                    iteration.append(i)
                    accuracy.append(acc)


            if i%480 == 0:

                correlation = np.zeros((par['n_latent'], 7))
                for l in range(par['n_latent']):
                # for latent_sample in latent_sample:
                    correlation[l,0] += pearsonr(latent_sample[:,l], inputs[:,0])[0] #x
                    correlation[l,1] += pearsonr(latent_sample[:,l], inputs[:,1])[0] #y
                    correlation[l,2] += pearsonr(latent_sample[:,l], inputs[:,2])[0] #dir_ind
                    correlation[l,3] += pearsonr(latent_sample[:,l], inputs[:,3])[0] #m
                    correlation[l,4] += pearsonr(latent_sample[:,l], inputs[:,4])[0] #fix
                    correlation[l,5] += pearsonr(latent_sample[:,l], outputs[:,0])[0] #motion_x
                    correlation[l,6] += pearsonr(latent_sample[:,l], outputs[:,1])[0] #motion_y
                print(['loc_x','loc_y','dir','m','fix','mot_x','mot_y'])
                print(np.round(correlation,3))
                print("")

                m = []
                for motion in range(par['num_motion_dirs']):
                    m.append(np.where(inputs[:,2]==motion))
                
                temp = np.zeros((par['n_latent'],8))
                for dir in range(par['num_motion_dirs']):
                    temp[:,dir] = np.round(np.mean(latent_sample[m[dir]], axis=0),3)
                plt.imshow(temp, cmap='inferno')
                plt.colorbar()
                # plt.show()


                var_dict = sess.run(model.generative_vars)
                with open('./savedir/generative_var_dict_trial.pkl', 'wb') as vf:
                    pickle.dump(var_dict, vf)

                # visualization(inputs, neural_inputs)

                for b in range(10):

                    output_string = ''
                    output_string += '\n--- {} ---\n'.format(b)
                    output_string += 'mu:  {}\n'.format(str(mu[b,:]))
                    output_string += 'sig: {}\n'.format(str(sigma[b,:]))

                    if b == 0:
                        rw = 'w'
                    else:
                        rw = 'a'

                    with open('./savedir/recon_data_iter{}.txt'.format(i,b), rw) as f:
                        f.write(output_string)


                    fig, ax = plt.subplots(2,2,figsize=[8,8])
                    for a in range(2):
                        inp = np.sum(np.reshape(neural_inputs[b], [9,10,10]), axis=a)
                        hat = np.sum(np.reshape(x_hat[b], [9,10,10]), axis=a)

                        ax[a,0].set_title('Actual (Axis {})'.format(a))
                        ax[a,0].imshow(inp, clim=[0,1])
                        ax[a,1].set_title('Reconstructed (Axis {})'.format(a))
                        ax[a,1].imshow(hat, clim=[0,1])

                    plt.savefig('./savedir/recon_iter{}_trial{}.png'.format(i,b))
                    plt.close(fig)

        print("TRAINING ON SUBSET LOC FINISHED\n")
        print("TESTING ON ALL QUADRANTS")
        test(stim, model, 0, sess, x, y, ff=False, gff=True)


        print("STARTING TO TEST TRAINING ON ALL QUADRANTS")


        for i in range(par['n_train_batches_gen'],par['n_train_batches_gen']+1500):

            name, inputs, neural_inputs, outputs = stim.generate_trial(0, False, False)
            feed_dict = {x:neural_inputs, y:outputs, alpha:0.05}
            _, loss, recon_loss, latent_loss, y_hat, x_hat, mu, sigma, latent_sample = sess.run([model.train_op, model.task_loss, \
                model.recon_loss, model.latent_loss, model.y, model.x_hat, model.mu, model.si, model.latent_sample], feed_dict=feed_dict)


            if i%50 == 0:
                ind = np.intersect1d(np.argwhere(inputs[:,3]==1), np.argwhere(inputs[:,4]==0))
                acc = get_perf(outputs,y_hat)
                print('{} | Reconstr. Loss: {:.3f} | Latent Loss: {:.3f} | Accuracy: {:.3f} | <Sig>: {:.3f} +/- {:.3f}'.format( \
                    i, recon_loss, latent_loss, acc, np.mean(sigma), np.std(sigma)))
                iteration.append(i)
                accuracy.append(acc)

        # print(iteration)
        # print(accuracy)
        plt.figure()
        plt.plot(iteration, accuracy, '-o', linestyle='-', marker='o',linewidth=2)
        plt.show()
        plt.savefig('./savedir/gen_model_learning_curve.png')
        plt.close()



            


    print('Complete.')


main()
