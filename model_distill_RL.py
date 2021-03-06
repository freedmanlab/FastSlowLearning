import tensorflow as tf
import numpy as np
import stimulus
import AdamOpt
import pickle
import matplotlib.pyplot as plt
from parameters_RL import par
from itertools import product
import os, sys, time

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print('TensorFlow version:\t', tf.__version__)
print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")

"""
Model setup and execution
"""
class Model:

    def __init__(self, input_data, target_data, pol_target_data, gating, pred_val, actual_action, advantage, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=0)
        self.target_data = tf.unstack(target_data, axis=0)
        self.pol_target_data = tf.unstack(pol_target_data, axis=0)
        self.gating = tf.reshape(gating, [1,-1])
        self.pred_val = tf.unstack(pred_val, axis=0)
        self.actual_action = tf.unstack(actual_action, axis=0)
        self.advantage = tf.unstack(advantage, axis=0)
        self.W_ei = tf.constant(par['EI_matrix'])


        self.time_mask = tf.unstack(mask, axis=0)


        # Build the TensorFlow graph
        self.rnn_cell_loop()

        # Train the model
        self.optimize()


    def rnn_cell_loop(self):


        self.W_ei = tf.constant(par['EI_matrix'])
        self.h = [] # RNN activity
        self.h_d = []
        self.pol_out = [] # policy output
        self.pol_d_out = [] # distill policy output
        self.val_out = [] # value output
        self.syn_x = [] # STP available neurotransmitter, currently not in use
        self.syn_u = [] # STP calcium concentration, currently not in use

        # we will add the first element to these lists since we need to input the previous action and reward
        # into the RNN
        self.action = []
        self.reward = []
        self.reward.append(tf.constant(np.zeros((par['batch_size'], par['n_val']), dtype = np.float32)))
        self.mask = []
        self.mask.append(tf.constant(np.ones((par['batch_size'], 1), dtype = np.float32)))

        """
        Initialize weights and biases
        """
        self.define_vars()

        h = tf.constant(par['h_init'])
        h_d = tf.constant(par['h_d_init'])
        syn_x = tf.constant(par['syn_x_init'])
        syn_u = tf.constant(par['syn_u_init'])
        syn_d_x = tf.constant(par['syn_x_init'])
        syn_d_u = tf.constant(par['syn_u_init'])


        """
        Loop through the neural inputs to the RNN, indexed in time
        """
        for rnn_input, target, time_mask in zip(self.input_data, self.target_data, self.time_mask):

            h, h_d, syn_x, syn_u, syn_d_x, syn_d_u, action, pol_out, pol_d_out, val_out, mask, reward  = self.rnn_cell(rnn_input, h, h_d, syn_x, syn_u, \
                syn_d_x, syn_d_u, self.reward[-1], self.mask[-1], target, time_mask)

            self.h.append(h)
            self.h_d.append(h_d)
            self.syn_x.append(syn_x)
            self.syn_u.append(syn_u)
            self.action.append(action)
            self.pol_out.append(pol_out)
            self.pol_d_out.append(pol_d_out)
            self.val_out.append(val_out)
            self.mask.append(mask)
            self.reward.append(reward)

        self.mask = self.mask[1:]
        # actions will produce a reward on the next time step
        self.reward = self.reward[1:]


    def rnn_cell(self, x, h, h_d, syn_x, syn_u, syn_d_x, syn_d_u, prev_reward, mask, target, time_mask):

        #self.define_vars(reuse = True)

        # Modify the recurrent weights if using excitatory/inhibitory neurons
        if par['EI']:
            self.W_rnn = tf.matmul(self.W_ei, tf.nn.relu(self.W_rnn))

        h, syn_x, syn_u = self.recurrent_cell(h, syn_x, syn_u, x, distill = False)
        h_d, syn_d_x, syn_d_u = self.recurrent_cell(h_d, syn_d_x, syn_d_u, x, distill = True)

        # calculate the policy output and choose an action
        pol_out = tf.matmul(h, self.W_pol_out) + self.b_pol_out
        pol_d_out = tf.matmul(h_d, self.W_d_pol_out) + self.b_d_pol_out
        #random_exp = tf.random_uniform(shape = (par['batch_size'], 1))
        #random_exp = tf.cast(random_exp > self.explore_prob, tf.float32)
        #pol_out_exp = pol_out*random_exp

        action_index = tf.multinomial(pol_out, 1)
        action = tf.one_hot(tf.squeeze(action_index), par['n_pol'])
        #pol_out = tf.nn.softmax(pol_out, dim = 1) # needed for optimize
        val_out = tf.matmul(h, self.W_val_out) + self.b_val_out

        # if previous reward was non-zero, then end the trial, unless the new trial signal cue is on
        continue_trial = tf.cast(tf.equal(prev_reward, 0.), tf.float32)
        mask *= continue_trial
        reward = tf.reduce_sum(action*target, axis = 1, keep_dims = True)*mask*time_mask

        return h, h_d, syn_x, syn_u, syn_d_x, syn_d_u, action, pol_out, pol_d_out, val_out, mask, reward


    def optimize(self):

        epsilon = 1e-7
        self.variables = [var for var in tf.trainable_variables() if not '_d_' in var.op.name]
        self.d_variables = [var for var in tf.trainable_variables() if '_d_' in var.op.name]
        #self.variables_val = [var for var in tf.trainable_variables() if 'val' in var.op.name]
        adam_optimizer = AdamOpt.AdamOpt(self.variables, learning_rate = par['learning_rate'])
        adam_optimizer_d = AdamOpt.AdamOpt(self.d_variables, learning_rate = par['learning_rate'])
        #adam_optimizer_val = AdamOpt.AdamOpt(self.variables_val, learning_rate = 10.*par['learning_rate'])

        self.previous_weights_mu_minus_1 = {}
        reset_prev_vars_ops = []
        self.big_omega_var = {}
        aux_losses = []

        for var in self.variables:
            self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            if not 'val' in var.op.name:
                # don't stabilizae the value weights or biases
                aux_losses.append(par['omega_c']*tf.reduce_sum(tf.multiply(self.big_omega_var[var.op.name], \
                    tf.square(self.previous_weights_mu_minus_1[var.op.name] - var) )))
            reset_prev_vars_ops.append( tf.assign(self.previous_weights_mu_minus_1[var.op.name], var ) )

        self.aux_loss = tf.add_n(aux_losses)

        self.pol_out_sm = [tf.nn.softmax(pol_out, dim = 1) for pol_out in self.pol_out]

        self.spike_loss = par['spike_cost']*tf.reduce_mean(tf.stack([mask*time_mask*tf.reduce_mean(h) \
            for (h, mask, time_mask) in zip(self.h, self.mask, self.time_mask)]))


        self.pol_loss = -tf.reduce_mean(tf.stack([advantage*time_mask*mask*act*tf.log(epsilon + pol_out) \
            for (pol_out, advantage, act, mask, time_mask) in zip(self.pol_out_sm, self.advantage, \
            self.actual_action, self.mask, self.time_mask)]))

        self.d_loss = tf.reduce_mean([mask*tf.nn.softmax_cross_entropy_with_logits(logits = y, \
            labels = target, dim=1) for y, target, mask in zip(self.pol_d_out, self.pol_target_data, self.mask)])

        self.spike_loss_d = par['spike_cost']*tf.reduce_mean(tf.stack([mask*time_mask*tf.reduce_mean(h) \
            for (h, mask, time_mask) in zip(self.h_d, self.mask, self.time_mask)]))



        self.entropy_loss = -par['entropy_cost']*tf.reduce_mean(tf.stack([tf.reduce_sum(time_mask*mask*pol_out*tf.log(epsilon+pol_out), axis = 1) \
            for (pol_out, mask, time_mask) in zip(self.pol_out_sm, self.mask, self.time_mask)]))


        self.val_loss = 0.5*tf.reduce_mean(tf.stack([time_mask*mask*tf.square(val_out - pred_val) \
            for (val_out, mask, time_mask, pred_val) in zip(self.val_out[:-1], self.mask, self.time_mask, self.pred_val[:-1])]))



        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        with tf.control_dependencies([self.pol_loss, self.aux_loss, self.spike_loss, self.val_loss]):
            self.train_op = adam_optimizer.compute_gradients(self.pol_loss + self.val_loss + \
                self.aux_loss + self.spike_loss - self.entropy_loss)
        self.train_op_d = adam_optimizer_d.compute_gradients(self.d_loss + self.spike_loss_d)


        # Stabilizing weights
        if par['stabilization'] == 'pathint':
            # Zenke method
            self.pathint_stabilization(adam_optimizer)

        elif par['stabilization'] == 'EWC':
            # Kirkpatrick method
            self.EWC()

        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()

        self.make_recurrent_weights_positive()
        #self.reset_zeroed_weights()



    def make_recurrent_weights_positive(self):

        reset_weights = []
        for var in self.variables:
            if 'W_rnn' in var.op.name:
                # make all negative weights slightly positive
                reset_weights.append(tf.assign(var,tf.maximum(1e-9, var)))

        self.reset_rnn_weights = tf.group(*reset_weights)

    def EWC(self):

        # Kirkpatrick method
        var_list = [var for var in tf.trainable_variables() if not 'val' in var.op.name]
        epsilon = 1e-6
        fisher_ops = []
        opt = tf.train.GradientDescentOptimizer(learning_rate = 1.0)
        log_p_theta = tf.stack([mask*time_mask*action*tf.log(epsilon + pol_out) for (pol_out, action, mask, time_mask) in \
            zip(self.pol_out,self.action, self.mask, self.time_mask)], axis = 0)

        grads_and_vars = opt.compute_gradients(log_p_theta, var_list = var_list)
        for grad, var in grads_and_vars:
            fisher_ops.append(tf.assign_add(self.big_omega_var[var.op.name], \
                grad*grad/par['EWC_fisher_num_batches']))

        self.update_big_omega = tf.group(*fisher_ops)


    def pathint_stabilization(self, adam_optimizer):
        # Zenke method

        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  1.0)
        self.small_omega_var = {}
        small_omega_var_div = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        initialize_prev_weights_ops = []


        self.previous_reward = tf.Variable(-tf.ones([]), trainable=False)
        self.current_reward = tf.Variable(-tf.ones([]), trainable=False)

        reward_stacked = tf.stack(self.reward, axis = 0)
        current_reward = tf.reduce_mean(tf.reduce_sum(reward_stacked, axis = 0))
        self.update_current_reward = tf.assign(self.current_reward, current_reward)
        self.update_previous_reward = tf.assign(self.previous_reward, self.current_reward)

        for var in self.variables:

            self.small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            small_omega_var_div[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append( tf.assign(self.small_omega_var[var.op.name], self.small_omega_var[var.op.name]*0.0 ) )

            update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.abs(self.small_omega_var[var.op.name]), \
            	(par['omega_xi'] + small_omega_var_div[var.op.name]))))


        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        # This is called every batch
        self.delta_grads = adam_optimizer.return_delta_grads()
        delta_reward = self.current_reward - self.previous_reward
        for grad,var in zip(self.delta_grads, self.variables):
            update_small_omega_ops.append(tf.assign_add(self.small_omega_var[var.op.name], self.delta_grads[var.op.name]*delta_reward))
            update_small_omega_ops.append(tf.assign_add(small_omega_var_div[var.op.name], tf.abs(self.delta_grads[var.op.name]*delta_reward)))
        self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!



    def recurrent_cell(self, h, syn_x, syn_u, x, distill = False):

        if par['synapse_config'] == 'std_stf':
            syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
            syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
            syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
            syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
            h_post = syn_u*syn_x*h
        else:
            h_post = h

        if distill:
            h = self.gating*tf.nn.relu((1-par['alpha_neuron'])*h + par['alpha_neuron']*(tf.matmul(x, self.W_d_in) + \
                tf.matmul(h_post, self.W_d_rnn) + self.b_d_rnn) + tf.random_normal(h.shape, 0, 4*par['noise_rnn'], dtype=tf.float32))
        else:
            h = self.gating*tf.nn.relu((1-par['alpha_neuron'])*h + par['alpha_neuron']*(tf.matmul(x, self.W_in) + \
                tf.matmul(h_post, self.W_rnn) + self.b_rnn) + tf.random_normal(h.shape, 0, par['noise_rnn'], dtype=tf.float32))


        return h, syn_x, syn_u


    def define_vars(self):

        # W_in0, and W_in1 are feedforward weights whose input is the convolved image, and projects onto the RNN
        # W_reward_pos, W_reward_neg project the postive and negative part of the reward from the previous time point onto the RNN
        # W_action projects the action from the previous time point onto the RNN
        # Wnn projects the activity of the RNN from the previous time point back onto the RNN (i.e. the recurrent weights)
        # W_pol_out projects from the RNN onto the policy output neurons
        # W_val_out projects from the RNN onto the value output neuron

        with tf.variable_scope('recurrent_pol'):
            self.W_in = tf.get_variable('W_in', initializer = par['W_in_init'])
            self.b_rnn = tf.get_variable('b_rnn', initializer = par['b_rnn_init'])
            self.W_rnn = tf.get_variable('W_rnn', initializer = par['W_rnn_init'])
            #self.W_reward_pos = tf.get_variable('W_reward_pos', initializer = par['W_reward_pos_init'])
            #self.W_reward_neg = tf.get_variable('W_reward_neg', initializer = par['W_reward_neg_init'])
            self.W_pol_out = tf.get_variable('W_pol_out', initializer = par['W_pol_out_init'])
            self.b_pol_out = tf.get_variable('b_pol_out', initializer = par['b_pol_out_init'])
            #self.W_action = tf.get_variable('W_action', initializer = par['W_action_init'])

            self.W_val_out = tf.get_variable('W_val_out', initializer = par['W_val_out_init'])
            self.b_val_out = tf.get_variable('b_val_out', initializer = par['b_val_out_init'])

            self.W_d_in = tf.get_variable('W_d_in', initializer = par['W_d_in_init'])
            self.b_d_rnn = tf.get_variable('b_d_rnn', initializer = par['b_d_rnn_init'])
            self.W_d_rnn = tf.get_variable('W_d_rnn', initializer = par['W_d_rnn_init'])
            self.W_d_pol_out = tf.get_variable('W_d_pol_out', initializer = par['W_d_out_init'])
            self.b_d_pol_out = tf.get_variable('b_d_pol_out', initializer = par['b_d_out_init'])




def main(gpu_id = None, save_fn = 'test.pkl'):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """
    Reset TensorFlow before running anything
    """
    tf.reset_default_graph()

    """
    Create the stimulus class to generate trial paramaters and input activity
    """
    stim = stimulus.MultiStimulus()


    """
    Define all placeholder
    """
    x, target, pol_target, mask, pred_val, actual_action, advantage, mask, gating, = generate_placeholders()

    config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True

    print_key_params()

    with tf.Session(config = config) as sess:

        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, target, pol_target, gating, pred_val, actual_action, advantage, mask)

        sess.run(tf.global_variables_initializer())

        # keep track of the model performance across training
        model_performance = {'reward': [], 'entropy_loss': [], 'val_loss': [], 'pol_loss': [], 'spike_loss': [], 'trial': [], 'task': []}
        reward_matrix = np.zeros((par['n_tasks'], par['n_tasks']))
        accuracy_full = []

        sess.run(model.reset_prev_vars)

        for task in range(2, par['n_tasks']):
        #for task in [0,3]:
            accuracy_above_threshold = 0

            task_start_time = time.time()

            for i in range(par['n_train_batches']):

                # make batch of training data
                name, input_data, desired_output, mk, reward_data = stim.generate_trial(task)
                mk = mk[..., np.newaxis]

                """
                Run the model
                """
                pol_out_list, val_out_list, h_list, action_list, mask_list, reward_list = sess.run([model.pol_out, model.val_out, model.h, model.action, \
                    model.mask, model.reward], {x: input_data, target: reward_data, mask: mk, gating:par['gating'][task]})


                """
                Unpack all lists, calculate predicted value and advantage functions
                """
                val_out, reward, adv, act, predicted_val, stacked_mask = stack_vars(pol_out_list, val_out_list, reward_list, action_list, mask_list, mk)

                T = 4
                #print(reward.shape)
                #print(act.shape)
                #print(stacked_mask.shape)
                pol_out_stacked = np.stack(pol_out_list)

                mk_pol = stacked_mask*mk*np.abs(adv)*10.
                pol_out_sm_neg = pol_out_stacked - 999*act
                pol_out_stacked = (adv>=0)*pol_out_stacked + (adv<0)*pol_out_sm_neg
                pol_out_sm = np.exp(pol_out_stacked/T)/np.sum(np.exp(pol_out_stacked/T), axis = 2, keepdims = True)


                _, pol_d_out_list = sess.run([model.train_op_d, model.pol_d_out], {x: input_data, target: reward_data, \
                    pol_target: pol_out_sm, mask: mk_pol, gating:par['gating'][task]})

                acc_d = get_perf(desired_output, pol_d_out_list, mk)

                """
                Calculate and apply gradients
                """
                if par['stabilization'] == 'pathint':
                    _, _, pol_loss, val_loss, aux_loss, spike_loss, ent_loss = sess.run([model.train_op, \
                         model.update_current_reward, model.pol_loss, model.val_loss, model.aux_loss, model.spike_loss, \
                        model.entropy_loss], feed_dict = {x:input_data, target:reward_data, \
                        gating:par['gating'][task], mask:mk, pred_val: predicted_val, actual_action: act, advantage:adv})
                    if i>0:
                        sess.run([model.update_small_omega])
                    sess.run([model.update_previous_reward])



                elif par['stabilization'] == 'EWC':
                    _, pol_loss,val_loss, aux_loss, spike_loss, ent_loss = sess.run([model.train_op, model.pol_loss, \
                        model.val_loss, model.aux_loss, model.spike_loss, model.entropy_loss], feed_dict = \
                        {x:input_data, target:reward_data, gating:par['gating'][task], mask:mk, pred_val: predicted_val, \
                        actual_action: act, advantage:adv})

                acc = np.mean(np.sum(reward>0,axis=0))
                if acc > 0.99:
                    accuracy_above_threshold += 1
                if accuracy_above_threshold >= 3000:
                    break

                sess.run([model.reset_rnn_weights])
                if i%10 == 0:
                    #print('Iter ', i, 'Task name ', name, ' accuracy', acc, ' aux loss', aux_loss, 'spike_loss', spike_loss, ' h > 0 ', above_zero, 'mean h', np.mean(h_stacked))
                    print('Iter ', i, 'Task name ', name, ' accuracy', acc, acc_d, ' aux loss', aux_loss, 'time ', np.around(time.time() - task_start_time))



            # Update big omegaes, and reset other values before starting new task
            if par['stabilization'] == 'pathint':
                """
                _, reset_masks = sess.run([model.reset_shunted_weights, model.reset_masks], feed_dict = \
                    {x:input_data, target: reward_data, gating:par['gating'][task], mask:mk})
                for i in range(len(reset_masks)):
                    print('Mean reset masks ', np.mean(reset_masks[i]))
                """
                big_omegas = sess.run([model.update_big_omega, model.big_omega_var])


            elif par['stabilization'] == 'EWC':
                for n in range(par['EWC_fisher_num_batches']):
                    name, input_data, _, mk, reward_data = stim.generate_trial(task)
                    mk = mk[..., np.newaxis]
                    big_omegas = sess.run([model.update_big_omega,model.big_omega_var], feed_dict = \
                        {x:input_data, target: reward_data, gating:par['gating'][task], mask:mk})

            # Test all tasks at the end of each learning session
            num_reps = 10
            for (task_prime, r) in product(range(par['n_tasks']), range(num_reps)):

                # make batch of training data
                name, input_data, _, mk, reward_data = stim.generate_trial(task_prime)
                mk = mk[..., np.newaxis]

                reward_list = sess.run([model.reward], feed_dict = {x:input_data, target: reward_data, \
                    gating:par['gating'][task_prime], mask:mk, explore_prob: 0.})
                # TODO: figure out what's with the extra dimension at index 0 in reward
                reward = np.squeeze(np.stack(reward_list))
                reward_matrix[task,task_prime] += np.mean(np.sum(reward>0,axis=0))/num_reps

            print('Accuracy grid after task {}:'.format(task))
            print(reward_matrix[task,:])
            results = {'reward_matrix': reward_matrix, 'par': par}
            pickle.dump(results, open(par['save_dir'] + save_fn, 'wb') )
            print('Analysis results saved in ', save_fn)
            print('')

            # Reset the Adam Optimizer, and set the previous parater values to their current values
            sess.run(model.reset_adam_op)
            sess.run(model.reset_prev_vars)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)



def get_perf(target, output, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when target[:,:,-1] is not 0
    """
    output = np.stack(output, axis=0)
    #print(target.shape, output.shape, mask.shape)


    mk = mask*np.reshape(target[:,:,-1] == 0, (par['num_time_steps'], par['batch_size'], 1))

    target = np.argmax(target, axis = 2)
    output = np.argmax(output, axis = 2)

    return np.sum(np.float32(target == output)*np.squeeze(mk))/np.sum(mk)


def stack_vars(pol_out_list, val_out_list, reward_list, action_list, mask_list, trial_mask):


    pol_out = np.stack(pol_out_list)
    val_out = np.stack(val_out_list)
    stacked_mask = np.stack(mask_list)*trial_mask
    reward = np.stack(reward_list)

    #val_out_stacked = np.vstack((np.zeros((1,par['batch_size'],par['n_val'])), val_out)) # option 1
    val_out_stacked = np.vstack((val_out,np.zeros((1,par['batch_size'],par['n_val'])))) # option 2

    terminal_state = np.float32(reward != 0) # this assumes that the trial ends when a reward other than zero is received
    pred_val = reward + par['discount_rate']*val_out_stacked[1:,:,:]*(1-terminal_state)
    adv = pred_val - val_out_stacked[:-1,:,:]
    #adv = reward - val_out
    act = np.stack(action_list)

    return val_out, reward, adv, act, pred_val, stacked_mask

def append_model_performance(model_performance, reward, entropy_loss, pol_loss, val_loss, trial_num):

    reward = np.mean(np.sum(reward,axis = 0))/par['trials_per_sequence']
    model_performance['reward'].append(reward)
    model_performance['entropy_loss'].append(entropy_loss)
    model_performance['pol_loss'].append(pol_loss)
    model_performance['val_loss'].append(val_loss)
    model_performance['trial'].append(trial_num)

    return model_performance

def generate_placeholders():

    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], 1])
    x = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_input']])  # input data
    target = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_pol']])  # target data
    pol_target = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_pol']])  # target data
    pred_val = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_val'], ])
    actual_action = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_pol']])
    advantage  = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_size'], par['n_val']])
    gating = tf.placeholder(tf.float32, [par['n_hidden']], 'gating')

    return x, target, pol_target, mask, pred_val, actual_action, advantage, mask, gating

def eval_weights():

    # TODO: NEEDS FIXING!
    with tf.variable_scope('rnn_cell', reuse=True):
        W_in = tf.get_variable('W_in')
        W_rnn = tf.get_variable('W_rnn')
        b_rnn = tf.get_variable('b_rnn')

    with tf.variable_scope('output', reuse=True):
        W_out = tf.get_variable('W_out')
        b_out = tf.get_variable('b_out')

    weights = {
        'w_in'  : W_in.eval(),
        'w_rnn' : W_rnn.eval(),
        'w_out' : W_out.eval(),
        'b_rnn' : b_rnn.eval(),
        'b_out'  : b_out.eval()
    }

    return weights

def print_results(iter_num, model_performance):

    reward = np.mean(np.stack(model_performance['reward'])[-par['iters_between_outputs']:])
    pol_loss = np.mean(np.stack(model_performance['pol_loss'])[-par['iters_between_outputs']:])
    val_loss = np.mean(np.stack(model_performance['val_loss'])[-par['iters_between_outputs']:])
    entropy_loss = np.mean(np.stack(model_performance['entropy_loss'])[-par['iters_between_outputs']:])

    print('Iter. {:4d}'.format(iter_num) + ' | Reward {:0.4f}'.format(reward) +
      ' | Pol loss {:0.4f}'.format(pol_loss) + ' | Val loss {:0.4f}'.format(val_loss) +
      ' | Entropy loss {:0.4f}'.format(entropy_loss))

def print_key_params():

    key_info = ['synapse_config','spike_cost','weight_cost','entropy_cost','omega_c','omega_xi',\
        'constrain_input_weights','num_sublayers','n_hidden','noise_rnn_sd','learning_rate',\
        'discount_rate', 'mask_duration', 'stabilization','gating_type', 'gate_pct',\
        'fix_break_penalty','wrong_choice_penalty','correct_choice_reward','include_rule_signal']
    print('Paramater info...')
    for k in key_info:
        print(k, ': ', par[k])
