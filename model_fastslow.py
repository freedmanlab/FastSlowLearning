### Authors: Nicolas Y. Masse, Gregory D. Grant

import tensorflow as tf
import numpy as np
import stimulus
from AdamOpt import *
from parameters_RL import *
import os, time
import pickle
import convolutional_layers
from itertools import product
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

###################
### Model setup ###
###################
class Model:

    def __init__(self, x, y, ys):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.x_data             = x
        self.y_data             = y
        self.ys_data            = ys
        # par['var_dict']            = weights

        # Build the TensorFlow graph
        self.run_model_ff()
        self.run_model_full()

        # Train the model
        self.optimize()


    def run_model_ff(self):
        with tf.variable_scope('ff_in'):
            self.W_in = tf.get_variable('W_in',initializer=par['W_in_init'], trainable=True)

        with tf.variable_scope('ff_transfer'):
            self.W_ls = []
            self.b_ls = []
            for i in range(par['num_layers_ff']-1):
                self.W_ls.append(tf.get_variable('W_l'+str(i+1),initializer=par['W_l_inits'][i], trainable=True))
            for i in range(par['num_layers_ff']):
                self.b_ls.append(tf.get_variable('b_l'+str(i+1),initializer=par['b_l_inits'][i], trainable=True))

        with tf.variable_scope('ff_output'):
            self.W_out = tf.get_variable('W_out',initializer=par['W_out_init'], trainable=True)
            self.b_out = tf.get_variable('b_out',initializer=par['b_out_init'], trainable=True)

        ls = [tf.nn.relu(tf.matmul(self.x_data, self.W_in) + self.b_ls[0])]
        for i in range(1,par['num_layers_ff']):
            ls.append(tf.nn.relu(tf.matmul(ls[i-1], self.W_ls[i-1]) + self.b_ls[i]))

        self.ff_output = tf.matmul(ls[-1], self.W_out) + self.b_out


    def run_model_full(self):
        with tf.variable_scope('conn_in'):
            W_conn_in = tf.get_variable('W_conn_in', initializer=par['W_conn_in_init'], trainable=True)
            b_conn = tf.get_variable('b_conn', initializer=par['b_conn_init'], trainable=True)

        with tf.variable_scope('conn_out'):
            W_conn_out = tf.get_variable('W_conn_out', initializer=par['W_conn_out_init'], trainable=True)
            b_conn_out = tf.get_variable('b_conn_out', initializer=par['b_conn_out_init'], trainable=True)

        with tf.variable_scope('conn_transfer'):
            W_mu_conn_in = tf.get_variable('W_mu_conn_in', shape=[par['n_inter'],par['n_latent']], trainable=True)
            W_si_conn_in = tf.get_variable('W_si_conn_in', shape=[par['n_inter'],par['n_latent']], trainable=True)
            b_mu_conn = tf.get_variable('b_mu_conn', shape=[1,par['n_latent']], trainable=True)
            b_si_conn = tf.get_variable('b_si_conn', shape=[1,par['n_latent']], trainable=True)

        # ys -> connection -> gFF -> FF -> y_hat
        connect_layer = tf.nn.relu(tf.matmul(self.ys_data, W_conn_in) + b_conn)
        self.conn_output = tf.matmul(connect_layer, W_conn_out) + b_conn_out

        self.mu = self.conn_output @ W_mu_conn_in + b_mu_conn
        self.si = self.conn_output @ W_si_conn_in + b_si_conn

        ### Copy from here down to include generative setup in full network
        self.latent_sample = self.mu + tf.exp(self.si)*tf.random_normal(self.si.shape)

        self.post = tf.nn.relu(self.latent_sample @ par['var_dict']['post_latent/W_lat'] + par['var_dict']['post_latent/b_post'])
        # self.post = tf.nn.relu(self.mu @ par['var_dict']['W_lat'] + par['var_dict']['b_post'])

        # h_out = []
        # for h in range(len(par['forward_shape']))[::-1]:
        #     if len(h_out) == 0:
        #         inp = self.post
        #         W = par['var_dict']['post_latent/W_post']
        #     else:
        #         inp = h_out[-1]
        #         W = par['var_dict']['post_latent/W_rec{}'.format(h+1)]

        #     act = inp @ W + par['var_dict']['post_latent/b_rec{}'.format(h)]
        #     if h is not 0:
        #         act = tf.nn.relu(act + 0.*tf.random_normal(act.shape))
        #         act = tf.nn.dropout(act, 0.8)
        #         h_out.append(act)
        #     else:
        #         h_out.append(act)
        W = np.float32(np.random.normal(0,1,size=[50,par['n_input']]))
        self.x_hat = self.post @ W

        # self.x_hat = np.float32(np.random.normal(0,1,size=[par['batch_size'],par['n_input']])) #h_out[-1]

        ls = [tf.nn.relu(tf.matmul(self.x_hat, self.W_in) + self.b_ls[0])]
        for i in range(1,par['num_layers_ff']):
            ls.append(tf.nn.relu(tf.matmul(ls[i-1], self.W_ls[i-1]) + self.b_ls[i]))

        self.full_output = tf.matmul(ls[-1], self.W_out) + self.b_out


    def optimize(self):

        # Trainable variables for FF / Generative / Connection
        self.variables_ff = [var for var in tf.trainable_variables() if var.op.name.find('ff')==0]
        self.variables_full = [var for var in tf.trainable_variables() if var.op.name.find('conn')==0]

        adam_optimizer_ff = AdamOpt(self.variables_ff, learning_rate = par['learning_rate'])
        adam_optimizer_full = AdamOpt(self.variables_full, learning_rate = par['learning_rate'])

        
        self.ff_loss = tf.reduce_mean([tf.square(y - y_hat) for (y, y_hat) in zip(tf.unstack(self.y_data,axis=0), tf.unstack(self.ff_output, axis=0))])
        with tf.control_dependencies([self.ff_loss]):
            self.train_op_ff = adam_optimizer_ff.compute_gradients(self.ff_loss)

        self.full_loss = tf.reduce_mean([tf.square(ys - ys_hat) for (ys, ys_hat) in zip(tf.unstack(self.ys_data,axis=0), tf.unstack(self.full_output, axis=0))])
        with tf.control_dependencies([self.full_loss]):
            self.train_op_full = adam_optimizer_full.compute_gradients(self.full_loss)
            

        # self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op_ff = adam_optimizer_ff.reset_params()
        self.reset_adam_op_full = adam_optimizer_full.reset_params()

        self.reset_weights_ff()
        self.reset_weights_full()

        self.make_recurrent_weights_positive_ff()
        self.make_recurrent_weights_positive_full()


    def reset_weights_ff(self):

        reset_weights_ff = []

        for var in self.variables_ff:
            if 'b' in var.op.name:
                # reset biases to 0
                reset_weights_ff.append(tf.assign(var, var*0.))
            elif 'W' in var.op.name:
                # reset weights to initial-like conditions
                new_weight = initialize_weight(var.shape, par['connection_prob'])
                reset_weights_ff.append(tf.assign(var,new_weight))

        self.reset_weights_ff = tf.group(*reset_weights_ff)

    def reset_weights_full(self):

        reset_weights_full = []

        for var in self.variables_full:
            if 'b' in var.op.name:
                # reset biases to 0
                reset_weights_full.append(tf.assign(var, var*0.))
            elif 'W' in var.op.name:
                # reset weights to initial-like conditions
                new_weight = initialize_weight(var.shape, par['connection_prob'])
                reset_weights_full.append(tf.assign(var,new_weight))

        self.reset_weights_full = tf.group(*reset_weights_full)

    def make_recurrent_weights_positive_ff(self):

        reset_weights = []
        for var in self.variables_ff:
            if 'W_rnn' in var.op.name:
                # make all negative weights slightly positive
                reset_weights.append(tf.assign(var,tf.maximum(1e-9, var)))

        self.reset_rnn_weights = tf.group(*reset_weights)

    def make_recurrent_weights_positive_full(self):

        reset_weights = []
        for var in self.variables_full:
            if 'W_rnn' in var.op.name:
                # make all negative weights slightly positive
                reset_weights.append(tf.assign(var,tf.maximum(1e-9, var)))

        self.reset_rnn_weights = tf.group(*reset_weights)


    def EWC(self):

        # Kirkpatrick method
        epsilon = 1e-5
        fisher_ops = []
        opt = tf.train.GradientDescentOptimizer(1)

        # model results p(y|x, theta)
        p_theta = tf.nn.softmax(self.output, dim = 1)
        # sample label from p(y|x, theta)
        class_ind = tf.multinomial(p_theta, 1)
        class_ind_one_hot = tf.reshape(tf.one_hot(class_ind, par['layer_dims'][-1]), \
            [par['batch_size'], par['layer_dims'][-1]])
        # calculate the gradient of log p(y|x, theta)
        log_p_theta = tf.unstack(class_ind_one_hot*tf.log(p_theta + epsilon), axis = 0)
        for lp in log_p_theta:
            grads_and_vars = opt.compute_gradients(lp)
            for grad, var in grads_and_vars:
                fisher_ops.append(tf.assign_add(self.big_omega_var[var.op.name], \
                    grad*grad/par['batch_size']/par['EWC_fisher_num_batches']))

        self.update_big_omega = tf.group(*fisher_ops)


    def pathint_stabilization(self, adam_optimizer, previous_weights_mu_minus_1):
        # Zenke method

        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  1.0)
        small_omega_var = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        initialize_prev_weights_ops = []

        for var in self.variables:

            small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )
            update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
                (par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))


        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        # This is called every batch
        with tf.control_dependencies([self.train_op]):
            self.delta_grads = adam_optimizer.return_delta_grads()
            self.gradients = optimizer_task.compute_gradients(self.task_loss)
            for grad,var in self.gradients:
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ) )
            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!



def main(save_fn=None, gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()
    f = open("./generative_var_dict.pkl","rb")
    par['var_dict'] = pickle.load(f)


    # Create placeholders for the model
    x  = tf.placeholder(tf.float32, [par['batch_size'], par['n_input']], 'stim')
    target   = tf.placeholder(tf.float32, [par['batch_size'], par['n_output']], 'out')
    ys = tf.placeholder(tf.float32, [par['n_ys'], 2], 'stim_y')

    stim = stimulus.MultiStimulus()
    accuracy_full = []
    accuracy_grid = np.zeros((par['n_tasks'], par['n_tasks']))
    accuracy_grid_slow = np.zeros((par['n_tasks'], par['n_tasks']))


    key_info = ['synapse_config','spike_cost','weight_cost','entropy_cost','omega_c','omega_xi',\
        'constrain_input_weights','num_sublayers','n_hidden','noise_rnn_sd','learning_rate','gating_type', 'gate_pct']
    print('Key info')
    for k in key_info:
        print(k, ' ', par[k])

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True

    # Model run session
    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, target, ys)

        sess.run(tf.global_variables_initializer())
        t_start = time.time()

        for task in range(0,par['n_tasks']):

            #################################
            ###     Training FF model     ###
            #################################
            print('FF Model execution starting.\n')
            for i in range(par['n_train_batches']):

                # make batch of training data
                name, stim_real, stim_in, y_hat = stim.generate_trial(task, subset_dirs=par['subset_dirs_ff'], subset_loc=par['subset_loc_ff'])

                # 0 for training FF, 1 for training connection, 2 for gFF
                _, ff_loss, ff_output = sess.run([model.train_op_ff, model.ff_loss, model.ff_output], feed_dict = {x:stim_in, target:y_hat})

                if i%50 == 0:
                    ff_acc = get_perf(y_hat, ff_output, ff=True)
                    print('Iter ', i, 'Task name ', name, ' accuracy', ff_acc, ' loss ', ff_loss)
            print('FF Model execution complete.\n')

            # Test all tasks at the end of each learning session
            print("FF Testing Phase")
            test(stim, model, task, sess, x, ys, ff=True)


            ################################
            ### Training Connected Model ###
            ################################
            print('Connected Model execution starting.\n')
            for i in range(par['n_train_batches_full']):

                # make batch of training data
                name, stim_real, stim_in, y_hat = stim.generate_trial(task, subset_dirs=par['subset_dirs'], subset_loc=par['subset_loc'])
                y_sample = y_hat[np.random.choice(np.arange(par['batch_size']), size=par['n_ys'])]

                # 0 for training FF, 1 for training gFF, 2 for connection
                _, full_loss, full_output, x_hat = sess.run([model.train_op_full, model.full_loss, model.full_output, model.x_hat], feed_dict = {ys: y_sample})

                if i%50 == 0:
                    conn_acc = get_perf(y_sample, full_output, ff=False)
                    print('Iter ', i, 'Task name ', name, ' accuracy', conn_acc, ' loss ', full_loss)
            print('Connected Model execution complete.\n')

            # Test all tasks at the end of each learning session
            print("Connected Model Testing Phase")
            x_hat_perf(stim_real, stim_in, x_hat)
            test(stim, model, task, sess, x, ys, ff=False)


            # Reset the Adam Optimizer, and set the previous parater values to their current values
            sess.run(model.reset_adam_op_ff)
            sess.run(model.reset_adam_op_full)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)

            # reset weights between tasks if called upon
            if par['reset_weights']:
                sess.run(model.reset_weights_ff)
                sess.run(model.reset_weights_full)



        if par['save_analysis']:
            save_results = {'task': task, 'accuracy': accuracy, 'accuracy_full': accuracy_full, \
                            'accuracy_grid': accuracy_grid, 'big_omegas': big_omegas, 'par': par}
            pickle.dump(save_results, open(par['save_dir'] + save_fn, 'wb'))

        print('\nModel execution complete.\n')


def softmax(x):
    temp = np.exp(x/par['T'])
    s = [np.sum(temp, axis=2) for i in range(par['n_output'])]
    return temp / np.stack(s, axis=2)
    #return np.divide(temp, np.stack(np.sum(temp, axis=2)))

def heat_map(input, target, output, grid, counter,loc=True,ff=True):

    num_total = par['batch_size'] if ff else par['n_ys']
    for b in range(num_total):
        x, y, dir, m = input[b]
        if m != 0:
            if loc:
                counter[int(x), int(y)] += 1
            else:
                counter[0,int(dir)] += 1

            if ((np.absolute(target[b,0] - output[b,0]) < par['tol']) and (np.absolute(target[b,1] - output[b,1]) < par['tol'])):
                if loc:
                    grid[int(x),int(y)] += 1
                else:
                    grid[0,int(dir)] += 1

    return grid, counter


def get_perf(target, output,ff):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when target[:,:,-1] is not 0
    """
    if ff:
        num_total = par['batch_size']
    else:
        num_total = par['n_ys']
    return np.sum(np.float32((np.absolute(target[:,0] - output[:,0]) < par['tol']) * (np.absolute(target[:,1] - output[:,1]) < par['tol'])))/num_total

def x_hat_perf(stim_real, stim_in, x_hat):
    for b in range(10):#par['batch_size']):
        dir = stim_real[b,2]
        stim = np.reshape(stim_in[b], (9,10,10))
        hat = np.reshape(x_hat[b], (9,10,10))

        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(stim[int(dir),:,:], cmap='inferno')
        plt.colorbar()

        plt.subplot(1,2,2)
        plt.imshow(hat[int(dir),:,:], cmap='inferno')
        plt.colorbar()
        plt.show()

def test(stim, model, task, sess, x, ys, ff):
    print("FF: ", ff)
    num_reps = 10
    acc = 0
    if (ff and par['subset_loc_ff']) or (not ff and par['subset_loc']):
        grid_loc = np.zeros((par['n_neurons'],par['n_neurons']), dtype=np.float32)
        counter_loc = np.zeros((par['n_neurons'],par['n_neurons']), dtype=np.int32)
    if (ff and par['subset_dirs_ff']) or (not ff and par['subset_dirs']):
        grid_dirs = np.zeros((1,par['num_motion_dirs']+1), dtype=np.float32)
        counter_dirs = np.zeros((1,par['num_motion_dirs']+1), dtype=np.int32)
    
    for (task_prime, r) in product(range(task+1), range(num_reps)):
        # make batch of training data
        name, stim_real, stim_in, y_hat = stim.generate_trial(task_prime, subset_dirs=False, subset_loc=False)
        if ff:
            output = sess.run(model.ff_output, feed_dict = {x:stim_in})
        else:
            index = np.random.choice(np.arange(par['batch_size']), size=par['n_ys'])
            stim_real = stim_real[index]
            y_hat = y_hat[index]
            output = sess.run(model.full_output, feed_dict = {ys:y_hat})

        if (ff and par['subset_loc_ff']) or (not ff and par['subset_loc']):
            grid_loc, counter_loc = heat_map(stim_real, y_hat, output, grid_loc, counter_loc, loc=True, ff=ff)
        if (ff and par['subset_dirs_ff']) or (not ff and par['subset_dirs']):
            grid_dirs, counter_dirs = heat_map(stim_real, y_hat, output, grid_dirs, counter_dirs, loc=False, ff=ff)
        acc += get_perf(y_hat, output, ff)

    print("Testing accuracy: ", acc/num_reps, "\n")

    if (ff and par['subset_loc_ff']) or (not ff and par['subset_loc']):
        counter_loc[counter_loc == 0] = 1
        plt.imshow(grid_loc/counter_loc, cmap='inferno')
        plt.colorbar()
        plt.clim(0,1)
        if ff:
            plt.savefig("./FF_subset_loc_"+str(par['tol'])+"_new.png")
        else:
            plt.savefig("./Full_model_subset_loc_"+str(par['tol'])+"_new.png")
        plt.show()
    if (ff and par['subset_dirs_ff']) or (not ff and par['subset_dirs']):
        counter_dirs[counter_dirs == 0] = 1
        plt.imshow(grid_dirs/counter_dirs, cmap='inferno')
        plt.colorbar()
        plt.clim(0,1)
        if ff:
            plt.savefig(("./FF_subset_dirs_"+str(par['tol'])+"_new.png"))
        else:
            plt.savefig("./Full_model_subset_dirs_"+str(par['tol'])+"_new.png")
        plt.show()

def gFF_test(stim, model, task):
    """
    Testing function for gFF model
    """

#main('testing')
