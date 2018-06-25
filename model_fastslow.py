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

########################
### Fast Model setup ###
########################
class Fast_Model:

    def __init__(self, input_data, target_data):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data         = input_data
        #self.gating             = tf.reshape(gating, [1,-1])
        self.target_data        = target_data
        #self.mask               = tf.unstack(mask, axis=0)
        #self.W_ei               = tf.constant(par['EI_matrix'])

        # Build the TensorFlow graph
        """self.hidden_state_hist = []
        self.syn_x_hist = []
        self.syn_u_hist = []
        self.output = []"""
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        with tf.variable_scope('ff_in'):
            W_in = tf.get_variable('W_in',initializer=par['W_in_init'], trainable=True)

        with tf.variable_scope('ff_transfer'):
            W_ls = []
            b_ls = []
            for i in range(par['num_layers_ff']-1):
                W_ls.append(tf.get_variable('W_l'+str(i+1),initializer=par['W_l_inits'][i], trainable=True))
            for i in range(par['num_layers_ff']):
                b_ls.append(tf.get_variable('b_l'+str(i+1),initializer=par['b_l_inits'][i], trainable=True))

        with tf.variable_scope('ff_output'):
            W_out = tf.get_variable('W_out',initializer=par['W_out_init'], trainable=True)
            b_out = tf.get_variable('b_out',initializer=par['b_out_init'], trainable=True)

        if par['EI']:
            W_rnn = tf.matmul(self.W_ei, tf.nn.relu(W_rnn))

        ls = [tf.nn.relu(tf.matmul(self.input_data, W_in) + b_ls[0])]
        for i in range(1,par['num_layers_ff']):
            ls.append(tf.nn.relu(tf.matmul(ls[i-1], W_ls[i-1]) + b_ls[i]))

        y = tf.matmul(ls[-1], W_out) + b_out

        self.output = y

    def optimize(self):

        # Use all trainable variables, except those in the convolutional layers
        self.variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        adam_optimizer = AdamOpt(self.variables, learning_rate = par['learning_rate'])

        previous_weights_mu_minus_1 = {}
        reset_prev_vars_ops = []
        self.big_omega_var = {}
        aux_losses = []

        for var in self.variables:
            self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            aux_losses.append(par['omega_c']*tf.reduce_sum(tf.multiply(self.big_omega_var[var.op.name], \
               tf.square(previous_weights_mu_minus_1[var.op.name] - var) )))
            reset_prev_vars_ops.append( tf.assign(previous_weights_mu_minus_1[var.op.name], var ) )

        # self.aux_loss = tf.add_n(aux_losses)
        self.loss = tf.reduce_mean([tf.square(y - y_hat) for (y, y_hat) in zip(tf.unstack(self.target_data,axis=0), tf.unstack(self.output, axis=0))])


        """
        with tf.variable_scope('rnn', reuse = True):
            W_in  = tf.get_variable('W_in')
            W_rnn = tf.get_variable('W_rnn')

        active_weights_rnn = tf.matmul(tf.reshape(self.gating,[-1,1]), tf.reshape(self.gating,[1,-1]))
        active_weights_in = tf.tile(tf.reshape(self.gating,[1,-1]),[par['n_input'], 1])
        self.weight_loss = par['weight_cost']*(tf.reduce_mean(active_weights_in*W_in**2) + tf.reduce_mean(tf.nn.relu(active_weights_rnn*W_rnn)**2))
        """
        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        #with tf.control_dependencies([self.task_loss, self.aux_loss, self.spike_loss_slow,self.entropy_loss ]):
        #    self.train_op = adam_optimizer.compute_gradients(self.task_loss + self.aux_loss + self.spike_loss_slow - self.entropy_loss)
        with tf.control_dependencies([self.loss]):
            self.train_op = adam_optimizer.compute_gradients(self.loss)

        # Stabilizing weights
        if par['stabilization'] == 'pathint':
            # Zenke method
            self.pathint_stabilization(adam_optimizer, previous_weights_mu_minus_1)

        elif par['stabilization'] == 'EWC':
            # Kirkpatrick method
            self.EWC()

        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()

        self.reset_weights()

        self.make_recurrent_weights_positive()


    def reset_weights(self):

        reset_weights = []

        for var in self.variables:
            if 'b' in var.op.name:
                # reset biases to 0
                reset_weights.append(tf.assign(var, var*0.))
            elif 'W' in var.op.name:
                # reset weights to initial-like conditions
                new_weight = initialize_weight(var.shape, par['connection_prob'])
                reset_weights.append(tf.assign(var,new_weight))

        self.reset_weights = tf.group(*reset_weights)

    def make_recurrent_weights_positive(self):

        reset_weights = []
        for var in self.variables:
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

    # train the convolutional layers with the CIFAR-10 dataset
    # otherwise, it will load the convolutional weights from the saved file
    if (par['task'] == 'cifar' or par['task'] == 'imagenet') and par['train_convolutional_layers']:
        convolutional_layers.ConvolutionalLayers()

    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    # input_data, target_data, gating, mask

    x  = tf.placeholder(tf.float32, [par['batch_size'], par['n_input']], 'stim')
    target   = tf.placeholder(tf.float32, [par['batch_size'], par['n_output']], 'out')
    #mask   = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')
    #gating = tf.placeholder(tf.float32, [par['n_hidden']], 'gating')

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

    # Fast Model run session
    with tf.Session(config=config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Fast_Model(x, target)
            #slow_model = Slow_Model(x, target, gating, mask)

        sess.run(tf.global_variables_initializer())
        t_start = time.time()
        sess.run(model.reset_prev_vars)

        for task in range(0,par['n_tasks']):

            for i in range(par['n_train_batches']):

                # make batch of training data
                name, stim_real, stim_in, y_hat = stim.generate_trial(task, subset_dirs=par['subset_dirs'], subset_loc=par['subset_loc'])

                if par['stabilization'] == 'pathint':
                    _, _, loss, AL, spike_loss, ent_loss, fast_output = sess.run([model.train_op, \
                        model.update_small_omega, model.task_loss, model.aux_loss, model.spike_loss, \
                        model.entropy_loss, model.output], \
                        feed_dict = {x:stim_in, target:y_hat})
                    sess.run([model.reset_rnn_weights])
                    if loss < 0.005 and AL < 0.0004 + 0.0002*task:
                        break

                elif par['stabilization'] == 'EWC':
                    _, loss, AL = sess.run([model.train_op, model.task_loss, model.aux_loss], feed_dict = \
                        {x:stim_in, target:y_hat, gating:par['gating'][task], mask:mk})

                else:
                     _, loss, fast_output = sess.run([model.train_op, \
                        model.loss, model.output], \
                        feed_dict = {x:stim_in, target:y_hat})

                if i%50 == 0:
                    acc = get_perf(y_hat, fast_output)
                    print('Iter ', i, 'Task name ', name, ' accuracy', acc, ' loss ', loss)


            # Test all tasks at the end of each learning session
            num_reps = 10
            acc = 0
            if par['subset_loc']:
                grid_loc = np.zeros((par['n_neurons'],par['n_neurons']), dtype=np.float32)
                counter_loc = np.zeros((par['n_neurons'],par['n_neurons']), dtype=np.int32)
            if par['subset_dirs']:
                grid_dirs = np.zeros((1,par['num_motion_dirs']+1), dtype=np.float32)
                counter_dirs = np.zeros((1,par['num_motion_dirs']+1), dtype=np.int32)
            for (task_prime, r) in product(range(task+1), range(num_reps)):

                # make batch of training data
                name, stim_real, stim_in, y_hat = stim.generate_trial(task_prime, subset_dirs=False, subset_loc=False)
                fast_output = sess.run(model.output, feed_dict = {x:stim_in})

                if par['subset_loc']:
                    grid_loc, counter_loc = heat_map(stim_real, y_hat, fast_output, grid_loc, counter_loc,loc=True)
                if par['subset_dirs']:
                    grid_dirs, counter_dirs = heat_map(stim_real, y_hat, fast_output, grid_dirs, counter_dirs,loc=False)
                acc += get_perf(y_hat, fast_output)

            print("Testing accuracy: ", acc/num_reps)

            if par['subset_loc']:
                counter_loc[counter_loc == 0] = 1
                plt.imshow(grid_loc/counter_loc, cmap='inferno')
                plt.colorbar()
                plt.clim(0,1)
                plt.show()
            if par['subset_dirs']:
                counter_dirs[counter_dirs == 0] = 1
                plt.imshow(grid_dirs/counter_dirs, cmap='inferno')
                plt.colorbar()
                plt.clim(0,1)
                plt.show()

            # Reset the Adam Optimizer, and set the previous parater values to their current values
            sess.run(model.reset_adam_op)
            sess.run(model.reset_prev_vars)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)

            # reset weights between tasks if called upon
            if par['reset_weights']:
                sess.run(model.reset_weights)




        if par['save_analysis']:
            save_results = {'task': task, 'accuracy': accuracy, 'accuracy_full': accuracy_full, \
                            'accuracy_grid': accuracy_grid, 'big_omegas': big_omegas, 'par': par}
            pickle.dump(save_results, open(par['save_dir'] + save_fn, 'wb'))

        print('\nFast Model execution complete.\n')

        """# Slow network session
        # sess.run(tf.global_variables_initializer())
        sess.run(slow_model.reset_prev_vars)

        for task in range(0,par['n_tasks']):

            for i in range(par['n_train_batches_slow']):

                # make batch of training data
                name, stim_in, y_hat, mk, _ = stim.generate_trial(task, training=True)

                if par['distillation']:
                    # getting fast output from the trained fast network
                    fast_output,_ = sess.run([model.output, model.syn_x_hist], feed_dict = {x:stim_in, gating:par['gating'][task_prime]})
                    fast_output = softmax(np.array(fast_output))

                    if par['stabilization'] == 'pathint':
                        _, _, loss, AL, spike_loss_slow, ent_loss, output = sess.run([slow_model.train_op, \
                            slow_model.update_small_omega, slow_model.task_loss, slow_model.aux_loss, slow_model.spike_loss_slow, \
                            slow_model.entropy_loss, slow_model.output], \
                            feed_dict = {x:stim_in, target:fast_output, gating:par['gating'][task], mask:mk})
                        sess.run([slow_model.reset_rnn_weights])
                        if loss < 0.005 and AL < 0.0004 + 0.0002*task:
                            break

                    elif par['stabilization'] == 'EWC':
                        _, loss, AL = sess.run([slow_model.train_op, slow_model.task_loss, slow_model.aux_loss], feed_dict = \
                            {x:stim_in, target:fast_output, gating:par['gating'][task], mask:mk})

                    else:
                         _, loss, spike_loss_slow, info_loss, ent_loss, output = sess.run([slow_model.train_op, \
                            slow_model.task_loss, slow_model.spike_loss_slow, slow_model.info_loss,\
                            slow_model.entropy_loss, slow_model.output], \
                            feed_dict = {x:stim_in, target:fast_output, gating:par['gating'][task], mask:mk})

                    if i%100 == 0:
                        acc = get_perf(y_hat, output, mk)
                        print('Iter ', i, 'Task name ', name, 'accuracy', acc, ' loss ', loss,  'spike loss', spike_loss_slow, \
                            ' info_loss', info_loss, ' entropy loss', ent_loss)

                else:
                    if par['stabilization'] == 'pathint':
                        _, _, loss, AL, spike_loss_slow, ent_loss, output = sess.run([slow_model.train_op, \
                            slow_model.update_small_omega, slow_model.task_loss, slow_model.aux_loss, slow_model.spike_loss_slow, \
                            slow_model.entropy_loss, slow_model.output], \
                            feed_dict = {x:stim_in, target:y_hat, gating:par['gating'][task], mask:mk})
                        sess.run([slow_model.reset_rnn_weights])
                        if loss < 0.005 and AL < 0.0004 + 0.0002*task:
                            break

                    elif par['stabilization'] == 'EWC':
                        _, loss, AL = sess.run([slow_model.train_op, slow_model.task_loss, slow_model.aux_loss], feed_dict = \
                            {x:stim_in, target:y_hat, gating:par['gating'][task], mask:mk})

                    else:
                         _, loss, spike_loss_slow, ent_loss, output = sess.run([slow_model.train_op, \
                            slow_model.task_loss, slow_model.spike_loss_slow, \
                            slow_model.entropy_loss, slow_model.output], \
                            feed_dict = {x:stim_in, target:y_hat, gating:par['gating'][task], mask:mk})

                    if i%100 == 0:
                        acc = get_perf(y_hat, output, mk)
                        print('Iter ', i, 'Task name ', name, ' accuracy', acc, ' loss ', loss,  'spike loss', spike_loss_slow, \
                            ' entropy loss', ent_loss)

            # Test all tasks at the end of each learning session
            num_reps = 10
            acc = 0
            for (task_prime, r) in product(range(task+1), range(num_reps)):

                # make batch of training data
                name, stim_in, y_hat, mk, _ = stim.generate_trial(task_prime, training=False)

                output,_ = sess.run([slow_model.output, slow_model.syn_x_hist], feed_dict = {x:stim_in, gating:par['gating'][task_prime]})
                acc += get_perf(y_hat, output, mk)
                accuracy_grid_slow[task,task_prime] += acc/num_reps

            print("Testing accuracy: ", acc/num_reps)

            # Update big omegaes, and reset other values before starting new task
            if par['stabilization'] == 'pathint':
                big_omegas = sess.run([slow_model.update_big_omega, slow_model.big_omega_var])
            elif par['stabilization'] == 'EWC':
                for n in range(par['EWC_fisher_num_batches']):
                    name, stim_in, y_hat, mk, _ = stim.generate_trial(task)
                    big_omegas = sess.run([slow_model.update_big_omega,slow_model.big_omega_var], feed_dict = \
                        {x:stim_in, target:y_hat, gating:par['gating'][task], mask:mk})



            # Reset the Adam Optimizer, and set the previous parater values to their current values
            sess.run(slow_model.reset_adam_op)
            sess.run(slow_model.reset_prev_vars)
            if par['stabilization'] == 'pathint':
                sess.run(slow_model.reset_small_omega)

            # reset weights between tasks if called upon
            if par['reset_weights']:
                sess.run(slow_model.reset_weights)




        if par['save_analysis']:
            save_results = {'task': task, 'accuracy': accuracy, 'accuracy_full': accuracy_full, \
                            'accuracy_grid': accuracy_grid_slow, 'big_omegas': big_omegas, 'par': par}
            pickle.dump(save_results, open(par['save_dir'] + save_fn, 'wb'))

    print('\nSlow Model execution complete.')
"""
def softmax(x):
    temp = np.exp(x/par['T'])
    s = [np.sum(temp, axis=2) for i in range(par['n_output'])]
    return temp / np.stack(s, axis=2)
    #return np.divide(temp, np.stack(np.sum(temp, axis=2)))

def heat_map(input, target, output, grid, counter,loc=True):

    for b in range(par['batch_size']):
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

def get_perf(target, output):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when target[:,:,-1] is not 0
    """
    return np.sum(np.float32((np.absolute(target[:,0] - output[:,0]) < par['tol']) * (np.absolute(target[:,1] - output[:,1]) < par['tol'])))/par['batch_size']

    #return np.sum([np.float32((t[0] - o[0]) < tol and (t[1] - o[1]) < tol) for (t, o) in zip(target, output)]/par['batch_size']

#main('testing')
