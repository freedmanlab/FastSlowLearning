import numpy as np
from parameters_RL import *
import model_fastslow
import sys, os
import pickle


def try_model(save_fn,gpu_id):

    try:
        # Run model
        model_fastslow.main(save_fn, gpu_id)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

###############################################################################
###############################################################################
###############################################################################

# Second argument will select the GPU to use
# Don't enter a second argument if you want TensorFlow to select the GPU/CPU
try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None


print('Task GO - Synaptic Stabilization = SI - Gating = 0%')
save_fn = 'go.pkl'
try_model(save_fn, gpu_id)
quit()
