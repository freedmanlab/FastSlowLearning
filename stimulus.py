import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from parameters import par
from parameters_RL import par

print('Using \'Multistim\' stimulus file.')

class MultiStimulus:

    def __init__(self):

        # Shape configuration
        self.input_shape    = [par['batch_size'],par['n_input'] ]
        self.output_shape   = [par['batch_size'],par['n_output']]
        '''
        self.stimulus_shape = [par['num_time_steps'], par['batch_size'],par['num_motion_tuned'] ]
        self.response_shape = [par['num_time_steps'], par['batch_size'],par['num_motion_dirs'] ]
        self.fixation_shape = [par['num_time_steps'], par['batch_size'],par['num_fix_tuned'] ]
        self.rule_shape = [par['num_time_steps'], par['batch_size'],par['num_rule_tuned'] ]
        self.mask_shape     = [par['num_time_steps'], par['batch_size']]
        '''
        # Motion and stimulus configuration
        self.motion_dirs    = np.linspace(0,2*np.pi-2*np.pi/par['num_motion_dirs'],par['num_motion_dirs'])

        '''
        self.stimulus_dirs  = np.linspace(0,2*np.pi-2*np.pi/(par['num_motion_tuned']//2),(par['num_motion_tuned']//2))
        self.modality_size  = (par['num_motion_tuned'])//2

        self.fix_time = 400

        # Go task stuff
        self.go_delay = np.array([200,400,800])//par['dt']

        # DM task stuff
        self.dm_c_set = np.array([-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08])*4.
        self.dm_stim_lengths = np.array([200,400,800])//par['dt']

        # DM Dly task stuff
        self.dm_dly_c_set = np.array([-0.32, -0.16, -0.08, 0.08, 0.16, 0.32])*4.
        self.dm_dly_delay = np.array([200, 400, 800])//par['dt']

        # Matching task stuff
        self.match_delay = np.array([200, 400, 800])//par['dt']
        '''
        # Initialize task interface
        self.get_tasks()
        self.task_id = 0
        self.task_order = np.arange(len(self.task_types))

        self.rule_signal_factor = 1. if par['include_rule_signal'] else 0.


    def circ_tuning(self, theta):

        #return np.maximum(0, par['tuning_height']*(np.exp(par['kappa']*np.cos(theta-self.stimulus_dirs[:,np.newaxis])) - 1)/np.exp(par['kappa']))

        ang_dist = np.angle(np.exp(1j*theta - 1j*self.stimulus_dirs[:,np.newaxis]))
        return par['tuning_height']*np.exp(-0.5*(8*ang_dist/np.pi)**2)





    def get_tasks(self):
        if par['task'] == 'multistim':
            self.task_types = [
                [self.task_go, 'go', 0],
                [self.task_go, 'rt_go', 0],
                [self.task_go, 'dly_go', 0],

                [self.task_go, 'go', np.pi],
                [self.task_go, 'rt_go', np.pi],
                [self.task_go, 'dly_go', np.pi],

                [self.task_dm, 'dm1'],
                [self.task_dm, 'dm2'],
                [self.task_dm, 'ctx_dm1'],
                [self.task_dm, 'ctx_dm2'],
                [self.task_dm, 'multsen_dm'],

                [self.task_dm_dly, 'dm1_dly'],
                [self.task_dm_dly, 'dm2_dly'],
                [self.task_dm_dly, 'ctx_dm1_dly'],
                [self.task_dm_dly, 'ctx_dm2_dly'],
                [self.task_dm_dly, 'multsen_dm_dly'],

                [self.task_matching, 'dms'],
                [self.task_matching, 'dmc'],
                [self.task_matching, 'dnms'],
                [self.task_matching, 'dnmc']
            ]

        elif par['task'] == 'twelvestim':
            self.task_types = [
                [self.task_go, 'go', 0],
                [self.task_go, 'dly_go', 0],

                [self.task_dm, 'dm1'],
                [self.task_dm, 'dm2'],
                [self.task_dm, 'ctx_dm1'],
                [self.task_dm, 'ctx_dm2'],
                [self.task_dm, 'multsen_dm'],

                [self.task_dm_dly, 'dm1_dly'],
                [self.task_dm_dly, 'dm2_dly'],
                [self.task_dm_dly, 'ctx_dm1_dly'],
                [self.task_dm_dly, 'ctx_dm2_dly'],
                [self.task_dm_dly, 'multsen_dm_dly']
            ]
        else:
            raise Exception('Multistimulus task type \'{}\' not yet implemented.'.format(par['task']))

        return self.task_types


    def generate_trial(self, current_task, subset_dirs, subset_loc):

        self.trial_info = {
            'input'          : np.zeros([par['batch_size'], 5]),
            'neural_input'   : np.random.normal(par['input_mean'], par['noise_in'], size=self.input_shape),
            'desired_output' : np.zeros(self.output_shape, dtype=np.float32),
            }

        #self.trial_info['train_mask'][:par['dead_time']//par['dt'], :] = 0

        if par['num_rule_tuned'] > 0:
            rule_signal = np.zeros((1,1,par['num_rule_tuned']))
            rule_signal[0,0,current_task] = par['tuning_height']
            self.trial_info['neural_input'][:, :, -par['num_rule_tuned']:] += rule_signal*self.rule_signal_factor

        task = self.task_types[current_task]    # Selects a task from the list
        task[0](*task[1:], subset_dirs, subset_loc)                      # Generates that task into trial_info
        # Returns the trial info and the task name

        # give -1 for breaking fixation, -0.01/+2 for choosing incorrectly/correctly

        return task[1], self.trial_info['input'], self.trial_info['neural_input'], self.trial_info['desired_output']

    def task_go(self, variant='go', offset=0, subset_dirs=False, subset_loc=False):

        # t = np.random.choice(3, par['batch_size'])
        # x = np.zeros(par['batch_size'])
        # y = np.zeros(par['batch_size'])
        # dir = np.zeros(par['batch_size'])
        # dir_ind = np.zeros(par['batch_size'])

        # m = (t!=2).astype(int)
        # fix = (t!=0).astype(int)

        # resp = np.zeros([par['batch_size'], par['num_motion_dirs']+1, par['n_neurons'], par['n_neurons']])
        # resp[m,par['num_motion_dirs'],4:6,4:6] = 0
        # if subset_dirs:
        #     x = np.random.choice(5, par['batch_size'])
        #     y = np.random.choice(5, par['batch_size'])
        #     dir_ind = np.random.randint(par['num_motion_dirs']-1, par['batch_size'])
        # else:
        #     x = np.random.choice(par['n_neurons'], par['batch_size'])
        #     y = np.random.choice(par['n_neurons'], par['batch_size'])
        #     dir_ind = np.random.randint(par['num_motion_dirs'], par['batch_size'])
        # dir = self.motion_dirs[dir_ind]

        # for b in range(par['batch_size']):
        #     if m[b]:
        #         for mn in range(par['num_motion_dirs']):
        #             spatial = np.exp(-1/2 * (np.square(x[b] - np.transpose(np.array([np.arange(par['n_neurons'])]*par['n_neurons']))) + np.square(y[b] - np.array([np.arange(par['n_neurons'])]*par['n_neurons']))))
        #             ang_dist = np.angle(np.exp(1j*dir - 1j*self.motion_dirs[mn]))
        #             motion = np.exp(-1/2 * np.square(ang_dist))
        #             resp[b,mn,:,:] = motion * spatial

        # self.trial_info['input'] = np.array([x, y, dir_ind, m, fix])
        # self.trial_info['neural_input'] = np.reshape(resp, (par['batch_size']))
        # self.trial_info['desired_output'] = np.array([np.cos(dir), np.sin(dir)]) * m * (1-fix)


        t = np.random.choice(4, par['batch_size'])
        x, y, dir, dir_ind = 0, 0, 0, 0

        # 0 = [motion, no fixation], 1 = [motion, fixation], 2 = [no motion, fixation]
        for b in range(par['batch_size']):
            if t[b] == 0:
                m = 1
                fix = 0
            elif t[b] == 1:
                m = 1
                fix = 1
            elif t[b] == 2:
                m = 0
                fix = 1
            else:
                m = 1
                fix = 0

            m = 1
            fix = 0

            mult_motion = np.random.choice([0.75, 1.0, 1.5]) if par['variable_stim'] else 1
            mult_fix = np.random.choice([0.75, 1.0, 1.5]) if par['variable_stim'] else 1

            resp = np.zeros([par['num_motion_dirs']+1, par['n_neurons'], par['n_neurons']])
            if fix:
                resp[par['num_motion_dirs'],3:7,3:7] = np.float32(1) * mult_fix
                dir_ind = 8
            if m:
                if subset_loc:
                    x = np.random.randint(5,par['n_neurons'])
                    y = np.random.randint(5,par['n_neurons'])
                else:
                    x = np.random.randint(par['n_neurons'])
                    y = np.random.randint(par['n_neurons'])

                dir_ind = np.random.randint(par['num_motion_dirs']-2) if subset_dirs else np.random.randint(par['num_motion_dirs'])
                dir = self.motion_dirs[dir_ind]

                for mn in range(par['num_motion_dirs']):
                    spatial = np.exp(-1/2 * (np.square(x - np.transpose(np.array([np.arange(par['n_neurons'])]*par['n_neurons']))) + np.square(y - np.array([np.arange(par['n_neurons'])]*par['n_neurons']))))
                    ang_dist = np.angle(np.exp(1j*dir - 1j*self.motion_dirs[mn]))
                    motion = np.exp(-1/2 * np.square(ang_dist))
                    resp[mn,:,:] = motion * spatial * mult_motion

                if fix:
                    dir_ind = 8

            self.trial_info['input'][b] = np.array([x, y, dir_ind, m, fix])
            self.trial_info['neural_input'][b] = np.reshape(resp, (1,-1))
            self.trial_info['desired_output'][b] = np.array([np.cos(dir), np.sin(dir)]) * m * (1-fix)

            # self.trial_info['neural_input'][b] += np.random.normal(0, 0.1, size=900)

        return self.trial_info


    def task_dm(self, variant='dm1',training=True):

        # Create trial stimuli
        stim_dir1 = np.random.choice(self.motion_dirs, [1, par['batch_size']])
        stim_dir2 = (stim_dir1 + np.pi/2 + np.random.choice(self.motion_dirs[::2], [1, par['batch_size']])/2)%(2*np.pi)

        stim1 = self.circ_tuning(stim_dir1)
        stim2 = self.circ_tuning(stim_dir2)

        # Determine the strengths of the stimuli in each modality
        c_mod1 = np.random.choice(self.dm_c_set, [1, par['batch_size']])
        c_mod2 = np.random.choice(self.dm_c_set, [1, par['batch_size']])
        mean_gamma = 0.8 + 0.4*np.random.rand(1, par['batch_size'])
        gamma_s1_m1 = mean_gamma + c_mod1
        gamma_s2_m1 = mean_gamma - c_mod1
        gamma_s1_m2 = mean_gamma + c_mod2
        gamma_s2_m2 = mean_gamma - c_mod2

        # Determine response directions and convert to output indices
        resp_dir_mod1 = np.where(gamma_s1_m1 > gamma_s2_m1, stim_dir1, stim_dir2)
        resp_dir_mod2 = np.where(gamma_s1_m2 > gamma_s2_m2, stim_dir1, stim_dir2)
        resp_dir_sum  = np.where(gamma_s1_m1 + gamma_s1_m2 > gamma_s2_m1 + gamma_s2_m2, stim_dir1, stim_dir2)

        #print('stim dirs 1', 180*stim_dir1[0,:10]/np.pi)
        #print('stim dirs 2', 180*stim_dir2[0,:10]/np.pi)
        #print('gamma_s1_m1', gamma_s1_m1[0,:10])
        #print('gamma_s2_m1', gamma_s2_m1[0,:10])
        #print('resp_dir_mod1', 180*resp_dir_mod1[0,:10]/np.pi)

        resp_dir_mod1 = np.round(par['num_motion_dirs']*resp_dir_mod1/(2*np.pi))
        resp_dir_mod2 = np.round(par['num_motion_dirs']*resp_dir_mod2/(2*np.pi))
        resp_dir_sum  = np.round(par['num_motion_dirs']*resp_dir_sum/(2*np.pi))

        #print('resp_dir_mod1', resp_dir_mod1[0,:10])

        # Apply stimuli to modalities and build appropriate response
        if variant == 'dm1':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = np.zeros_like(stim1)
            resp_dirs = resp_dir_mod1
        elif variant == 'dm2':
            modality1 = np.zeros_like(stim1)
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'ctx_dm1':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod1
        elif variant == 'ctx_dm2':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'multsen_dm':
            modality1 = gamma_s1_m1*stim1 + gamma_s2_m1*stim2
            modality2 = gamma_s1_m2*stim1 + gamma_s2_m2*stim2
            resp_dirs = resp_dir_sum
        else:
            raise Exception('Bad task variant.')

        resp = np.zeros([par['num_motion_dirs'], par['batch_size']])
        for b in range(par['batch_size']):
            resp[np.int16(resp_dirs[0,b]%par['num_motion_dirs']),b] = 1

        # Setting up arrays
        fixation = np.zeros(self.fixation_shape)
        response = np.zeros(self.response_shape)
        stimulus = np.zeros(self.stimulus_shape)
        mask     = np.ones(self.mask_shape)
        mask[:par['dead_time']//par['dt'],:] = 0

        # Identify stimulus onset for each trial and build each trial from there
        stim_onset = self.fix_time//par['dt']
        stim_off   = stim_onset + np.random.choice(self.dm_stim_lengths, par['batch_size'])
        #resp_time  = stim_off + 500//par['dt']
        for b in range(par['batch_size']):
            #fixation[:resp_time[b],b,:] = 1
            fixation[:stim_off[b],b,:] = 1
            stimulus[stim_onset:stim_off[b],b,:] = np.transpose(np.concatenate([modality1[:,b], modality2[:,b]], axis=0)[:,np.newaxis])
            #response[resp_time[b]:,b,:] = np.transpose(resp[:,b,np.newaxis])
            #mask[resp_time[b]:resp_time[b]+par['mask_duration']//par['dt'],b] = 0
            response[stim_off[b]:,b,:] = np.transpose(resp[:,b,np.newaxis])
            mask[stim_off[b]:stim_off[b]+par['mask_duration']//par['dt'],b] = 0

        # Tweak the fixation array
        #stim_fix = fixation
        #resp_fix = fixation[0:1,:,:]

        # Merge activies and fixations into single vector
        stimulus = np.concatenate([stimulus, fixation], axis=2)
        response = np.concatenate([response, fixation[:,:,0:1]], axis=2)

        self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
        self.trial_info['desired_output'] = response
        self.trial_info['train_mask'] = mask

        """
        plt.subplot(2,2,1)
        plt.imshow(self.trial_info['neural_input'][:,0,:], aspect = 'auto')
        plt.subplot(2,2,2)
        plt.imshow(self.trial_info['desired_output'][:,0,:], aspect = 'auto')
        plt.subplot(2,2,3)
        plt.imshow(self.trial_info['train_mask'][:,:], aspect = 'auto')
        plt.show()
        """


        return self.trial_info


    def task_dm_dly(self, variant='dm1', training=True):

        # Create trial stimuli
        stim_dir1 = 2*np.pi*np.random.rand(1, par['batch_size'])
        stim_dir2 = (stim_dir1 + np.pi/2 + np.pi*np.random.rand(1, par['batch_size']))%(2*np.pi)
        stim1 = self.circ_tuning(stim_dir1)
        stim2 = self.circ_tuning(stim_dir2)

        # Determine the strengths of the stimuli in each modality
        c_mod1 = np.random.choice(self.dm_dly_c_set, [1, par['batch_size']])
        c_mod2 = np.random.choice(self.dm_dly_c_set, [1, par['batch_size']])
        mean_gamma = 0.8 + 0.4*np.random.rand(1, par['batch_size'])
        gamma_s1_m1 = mean_gamma + c_mod1
        gamma_s2_m1 = mean_gamma - c_mod1
        gamma_s1_m2 = mean_gamma + c_mod2
        gamma_s2_m2 = mean_gamma - c_mod2

        # Determine the delay for each trial
        delay = np.random.choice(self.dm_dly_delay, [1, par['batch_size']])

        # Determine response directions and convert to output indices
        resp_dir_mod1 = np.where(gamma_s1_m1 > gamma_s2_m1, stim_dir1, stim_dir2)
        resp_dir_mod2 = np.where(gamma_s1_m2 > gamma_s2_m2, stim_dir1, stim_dir2)
        resp_dir_sum  = np.where(gamma_s1_m1 + gamma_s1_m2 > gamma_s2_m1 + gamma_s2_m2, stim_dir1, stim_dir2)

        resp_dir_mod1 = np.round(par['num_motion_dirs']*resp_dir_mod1/(2*np.pi))
        resp_dir_mod2 = np.round(par['num_motion_dirs']*resp_dir_mod2/(2*np.pi))
        resp_dir_sum  = np.round(par['num_motion_dirs']*resp_dir_sum/(2*np.pi))

        # Apply stimuli to modalities and build appropriate response
        if variant == 'dm1_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = np.zeros_like(stim1)
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = np.zeros_like(stim2)
            resp_dirs = resp_dir_mod1
        elif variant == 'dm2_dly':
            modality1_t1 = np.zeros_like(stim1)
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = np.zeros_like(stim2)
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'ctx_dm1_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod1
        elif variant == 'ctx_dm2_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_mod2
        elif variant == 'multsen_dm_dly':
            modality1_t1 = gamma_s1_m1*stim1
            modality2_t1 = gamma_s1_m2*stim1
            modality1_t2 = gamma_s2_m1*stim2
            modality2_t2 = gamma_s2_m2*stim2
            resp_dirs = resp_dir_sum
        else:
            raise Exception('Bad task variant.')

        resp = np.zeros([par['num_motion_dirs'], par['batch_size']])
        for b in range(par['batch_size']):
            resp[np.int16(resp_dirs[0,b]%par['num_motion_dirs']),b] = 1

        # Setting up arrays
        fixation = np.zeros(self.fixation_shape)
        response = np.zeros(self.response_shape)
        stimulus = np.zeros(self.stimulus_shape)
        mask     = np.ones(self.mask_shape)
        mask[:par['dead_time']//par['dt'],:] = 0

        # Identify stimulus onset for each trial and build each trial from there
        stim_on1   = self.fix_time//par['dt']
        stim_off1  = (self.fix_time+300)//par['dt']
        stim_on2   = delay + stim_off1
        stim_off2  = stim_on2 + 300//par['dt']
        resp_time  = stim_off2 + 0//par['dt']
        for b in range(par['batch_size']):
            fixation[:resp_time[0,b],b,:] = 1
            stimulus[stim_on1:stim_off1,b,:] = np.concatenate([modality1_t1[:,b], modality2_t1[:,b]], axis=0)[np.newaxis,:]
            stimulus[stim_on2[0,b]:stim_off2[0,b],b] = np.concatenate([modality1_t2[:,b], modality2_t2[:,b]], axis=0)[np.newaxis,:]
            response[resp_time[0,b]:,b,:] = resp[np.newaxis,:,b]
            mask[resp_time[0,b]:resp_time[0,b]+par['mask_duration'],b] = 0

        # Merge activies and fixations into single vectors
        stimulus = np.concatenate([stimulus, fixation], axis=2)
        response = np.concatenate([response, fixation[:,:,0:1]], axis=2)    # Duplicates starting fixation on output

        self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
        self.trial_info['desired_output'] = response
        self.trial_info['train_mask'] = mask

        return self.trial_info


    def task_matching(self, variant='dms', training=True):

        # Determine matches, and get stimuli
        if variant in ['dms', 'dnms']:
            stim1 = np.random.choice(self.motion_dirs, par['batch_size'])
            nonmatch = (stim1 + np.random.choice(self.motion_dirs[1:], par['batch_size']))%(2*np.pi)

            match = np.random.choice(np.array([True, False]), par['batch_size'])
            stim2 = np.where(match, stim1, nonmatch)

        elif variant in ['dmc', 'dnmc']:
            stim1 = np.random.choice(self.motion_dirs, par['batch_size'])
            stim2 = np.random.choice(self.motion_dirs, par['batch_size'])

            stim1_cat = np.logical_and(np.less(0, stim1), np.less(stim1, np.pi))
            stim2_cat = np.logical_and(np.less(0, stim2), np.less(stim2, np.pi))
            match = np.logical_not(np.logical_xor(stim1_cat, stim2_cat))
        else:
            raise Exception('Bad variant.')

        # Establishing stimuli
        stimulus1 = self.circ_tuning(stim1)
        stimulus2 = self.circ_tuning(stim2)

        # Convert to response
        stim1_int = np.round(par['num_motion_dirs']*stim1/(2*np.pi))
        stim2_int = np.round(par['num_motion_dirs']*stim2/(2*np.pi))

        if variant in ['dms', 'dmc']:
            resp = np.where(match, stim1_int, -1)
        elif variant in ['dnms', 'dnmc']:
            resp = np.where(match, -1, stim2_int)
        else:
            raise Exception('Bad variant.')

        # Setting up arrays
        modality_choice = np.random.choice(np.array([0,1], dtype=np.int16), [2, par['batch_size']])
        modalities = np.zeros([2, par['num_time_steps'], par['batch_size'], par['num_motion_tuned']//2])
        fixation = np.zeros(self.fixation_shape)
        response = np.zeros(self.response_shape)
        stimulus = np.zeros(self.stimulus_shape)
        mask     = np.ones(self.mask_shape)
        mask[:par['dead_time']//par['dt'],:] = 0

        # Decide timings and build each trial
        stim1_on  = 300//par['dt']
        stim1_off = 600//par['dt']
        stim2_on  = stim1_off + np.random.choice(self.match_delay, par['batch_size'])
        stim2_off = stim2_on + 300//par['dt']
        resp_time = stim2_off
        resp_fix  = np.copy(fixation[:,:,0:1])

        for b in range(par['batch_size']):
            fixation[:resp_time[b],b,:] = 1
            modalities[modality_choice[0,b],stim1_on:stim1_off,b,:] = stimulus1[np.newaxis,:,b]
            modalities[modality_choice[1,b],stim2_on[b]:stim2_off[b],b,:] = stimulus2[np.newaxis,:,b]
            mask[resp_time[b]:resp_time[b]+par['mask_duration']//par['dt'],b] = 0
            if not resp[b] == -1:
                response[resp_time[b]:,b,int(resp[b])] = 1
            else:
                resp_fix[:,b,:] = 1

        # Merge activies and fixations into single vectors)
        stimulus = np.concatenate([modalities[0], modalities[1], fixation], axis=2)
        response = np.concatenate([response, np.maximum(resp_fix, fixation[:,:,0:1])], axis=2)

        self.trial_info['neural_input'][:,:,:par['num_motion_tuned']+par['num_fix_tuned']] += stimulus
        self.trial_info['desired_output'] = response
        self.trial_info['train_mask'] = mask

        return self.trial_info

def visualization(stim_real, x_hat, y_sample, full_output, iter):
    for b in range(10):
        z = np.reshape(x_hat[b], (par['num_motion_dirs']+1,par['n_neurons'],par['n_neurons']))
        y_sample_dir = int(stim_real[b,2])
        motion = int(stim_real[b,3])
        fix = int(stim_real[b,4])
        vmin = np.min(z)
        vmax = np.max(z)
        coord = y_sample[b]
        output = full_output[b]

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7,7))
        # fig.suptitle("y_sample_dir: "+str(y_sample_dir)+" motion: "+str(motion)+" fix: "+str(fix))
        fig.suptitle("y_sample: "+str(round(coord[0],3))+","+str(round(coord[1],3))+" y_sample_dir: "+str(y_sample_dir) \
            + "\nm: "+str(motion)+" fix: "+str(fix)+"\nfull_output: "+str(round(output[0],3))+","+str(round(output[1],3)))
        i = 0
        for ax in axes.flat:
            im = ax.imshow(z[i,:,:], vmin=vmin, vmax=vmax, cmap='inferno')
            i += 1
        cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
        plt.margins(tight=True)
        # plt.savefig("./savedir/iter_"+str(iter)+"_"+str(b)+".png")
        # plt.show()
        plt.close()

### EXAMPLE ###
# stim = MultiStimulus()
# name, stim_real, stim_in, y_hat = stim.generate_trial(0, subset_dirs=par['subset_dirs_ff'], subset_loc=par['subset_loc_ff'])
# visualization(stim_real, stim_in, y_hat, y_hat, 0)

"""
st = MultiStimulus()
for i in range(len(st.task_types)):
    print(i, st.task_types[i][1:])
    t, trial_info = st.generate_trial(i)

    s = trial_info['neural_input']
    r = trial_info['desired_output']
    m = trial_info['train_mask']

    print(t)
    print(s.shape)
    print(r.shape)
    print(m.shape)

    fig, axarr = plt.subplots(8, 3, sharex=True, sharey=True)
    for i in range(8):
        axarr[i,0].imshow(s[:,:,i])
        axarr[i,1].imshow(r[:,:,i])
        axarr[i,2].plot(m[:,i])

    plt.show()
quit()
"""
