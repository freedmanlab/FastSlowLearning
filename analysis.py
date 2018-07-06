import numpy as np
import matplotlib.pyplot as plt
from parameters_RL import *
import pickle
import stimulus
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import product

def heat_map(input, target, output, grid, counter,loc=True,ff=True):

    num_total = par['batch_size'] if ff else par['n_ys']
    for b in range(num_total):
        x, y, dir, m, fix = input[b]
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

def get_perf_angle(stim_real, output):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when target[:,:,-1] is not 0
    """
    ang_diff = []
    for b in range(par['n_ys']):
        m = stim_real[b,3]
        if m != 0:
            target_ang = np.linspace(0,2*np.pi-2*np.pi/(par['num_motion_tuned']//2),(par['num_motion_tuned']//2))[int(stim_real[b,2])]
            ang_hat = np.arctan(output[b,1]/output[b,0])
            ang_diff.append(np.absolute(target_ang - ang_hat) < par['ang_tol'])

    return np.mean(ang_diff)

def visualization(stim_real, x_hat, iter):
    for b in range(10):
        z = np.reshape(x_hat[b], (par['num_motion_dirs']+1,par['n_neurons'],par['n_neurons']))
        y_sample_dir = int(stim_real[b,2])
        motion = int(stim_real[b,3])
        fix = int(stim_real[b,4])
        vmin = np.min(z)
        vmax = np.max(z)

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7,7))
        fig.suptitle("y_sample_dir: "+str(y_sample_dir)+" motion: "+str(motion)+" fix: "+str(fix))
        i = 0
        for ax in axes.flat:
            im = ax.imshow(z[i,:,:], vmin=vmin, vmax=vmax, cmap='inferno')
            i += 1
        cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        plt.colorbar(im, cax=cax, **kw)
        plt.margins(tight=True)
        plt.savefig("./savedir/iter_"+str(iter)+"_"+str(b)+".png")
        # plt.show()
        plt.close()

def x_hat_perf(stim_real, stim_in, x_hat):
    # get direction
    # dir_x = int(np.where(x_hat[b].reshape((9,10,10))==np.max(x_hat[b]))[0]) #for b in range(par['batch_size'])]
    # ang = np.linspace(0,2*np.pi-2*np.pi/(par['num_motion_tuned']//2),(par['num_motion_tuned']//2))[dir_x]
    # target = [np.cos(ang), np.sin(ang)]


    print("x_hat direction\ty_sample direction")
    for b in range(10):#par['batch_size']):
        m = int(stim_real[b,3])
        fix = int(stim_real[b,4])
        if m != 0 and fix !=0:
            # stim = np.reshape(stim_in[b], (9,10,10))
            z = np.reshape(x_hat[b], (9,10,10)) # I've made many mistakes in the past with reshape... make sure the dimensions and order are correct!
            # # z = z[:8,:,:] # I'm guessing the first 8 indices give motion direction
            # # v = np.exp(1j*np.arange(8)*np.pi/8) # will give you a vector of 8 directions along unit circle in complex coordinates
            # # v = np.reshape(v,(8,1,1))
            # # x_dir = np.angle(np.sum(z*v))
            x_dir = int(np.where(z==np.max(x_hat[b]))[0])
            x_x = int(np.where(z==np.max(x_hat[b]))[1])
            x_y = int(np.where(z==np.max(x_hat[b]))[2])
            # dir = int(stim_real[b,2])
            # x = int(stim_real[b,0])
            # y = int(stim_real[b,1])
            # print(x_dir, "\t", dir)
            # plt.subplot(1,2,1)
            # plt.imshow(z[x_dir,:,:], cmap='inferno')
            # plt.subplot(1,2,2)
            # plt.imshow(stim[dir,:,:], cmap='inferno')
            # plt.colorbar()
            # plt.title("y_sample\tx: "+str(x)+" y: "+str(y)+" dir: "+str(dir)+" m: "+str(m)+"\nx_hat\tx: "+str(x_x)+" y: "+str(x_y)+" dir: "+str(x_dir)+" m: "+str(m))
            # plt.show()

            x = int(stim_real[b,0])
            y = int(stim_real[b,1])
            dir = int(stim_real[b,2])
            m = int(stim_real[b,3])

            stim = np.reshape(stim_in[b], (10,10,10))
            stim2 = np.sum(stim,axis=1)
            stim2 = np.sum(stim2,axis=1)
            hat = np.reshape(x_hat[b], (10,10,10))
            hat2 = hat
            hat2[8,:,:] = 0
            hat2 = np.sum(hat2,axis=1)
            hat2 = np.sum(hat2,axis=1)

            plt.figure()
            #plt.subplot(2,1,1)
            #plt.imshow(stim[:,x,y], cmap='inferno')


        # for n in range(par['n_neurons']):
            # for m in range(par['n_neurons']):
                # plt.subplot(10,10,(n+1)*9+m+1)
            #plt.figure()
            # plt.imshow([stim[:,x,y],hat[:,x,y]], cmap='inferno')
            #plt.subplot(2,1,2)
            plt.imshow([stim2,hat2], cmap='inferno')
            plt.title("y_sample x: "+str(x)+" y: "+str(y)+" dir: "+str(dir)+" m: "+str(m)+"\nx_hat x: "+str(x_x)+" y: "+str(x_y)+" dir: "+str(x_dir)+" m: "+str(m))
            plt.colorbar()
            plt.show()
        # plt.subplot(2,1,2)
        # plt.imshow(hat[dir], cmap='inferno')
        # plt.colorbar()
        # plt.title("x: "+str(x)+" y: "+str(y)+" dir: "+str(dir)+" m: "+str(m))
        # # plt.savefig("./spooky/iter_"+str(i)+"/"+str(b)+".png")
        # plt.clim(0,10)
        # plt.show()
        # plt.close()

def test(stim, model, task, sess, x, ys, ff):
    print("FF: ", ff)
    num_reps = 10
    acc = 0
    subset_loc = ((ff and par['subset_loc_ff']) or (not ff and par['subset_loc']))
    subset_dirs = ((ff and par['subset_dirs_ff']) or (not ff and par['subset_dirs']))

    if subset_loc:
        grid_loc = np.zeros((par['n_neurons'],par['n_neurons']), dtype=np.float32)
        counter_loc = np.zeros((par['n_neurons'],par['n_neurons']), dtype=np.int32)
    if subset_dirs:
        grid_dirs = np.zeros((1,par['num_motion_dirs']+2), dtype=np.float32)
        counter_dirs = np.zeros((1,par['num_motion_dirs']+2), dtype=np.int32)

    for (task_prime, r) in product(range(task+1), range(num_reps)):
        # make batch of training data
        name, stim_real, stim_in, y_hat = stim.generate_trial(task_prime, subset_dirs=False, subset_loc=False)
        if ff:
            output = sess.run(model.ff_output, feed_dict = {x:stim_in})
        else:
            index = np.random.choice(np.arange(par['batch_size']), size=par['n_ys'])
            stim_real = stim_real[index]
            stim_in = stim_in[index]
            y_hat = y_hat[index]
            # y_hat = np.array([[1.0,0.0]] * par['n_ys'])

            output, x_hat = sess.run([model.full_output, model.x_hat], feed_dict = {ys:y_hat})
            if r == 0:
                visualization(stim_real, x_hat, 0)
            # x_hat_perf(stim_real, stim_in, x_hat, par['n_train_batches_full'])

        if subset_loc:
            grid_loc, counter_loc = heat_map(stim_real, y_hat, output, grid_loc, counter_loc, loc=True, ff=ff)
        if subset_dirs:
            grid_dirs, counter_dirs = heat_map(stim_real, y_hat, output, grid_dirs, counter_dirs, loc=False, ff=ff)
        acc += get_perf(y_hat, output, ff)

    print("Testing accuracy: ", acc/num_reps, "\n")

    if subset_loc:
        counter_loc[counter_loc == 0] = 1
        plt.imshow(grid_loc/counter_loc, cmap='inferno')
        plt.colorbar()
        plt.clim(0,1)
        if ff:
            plt.savefig("./savedir/FF_subset_loc_"+str(par['tol'])+"_new.png")
        else:
            plt.savefig("./savedir/Full_model_subset_loc_"+str(par['tol'])+"_new.png")
        # plt.show()
    if subset_dirs:
        counter_dirs[counter_dirs == 0] = 1
        plt.imshow(grid_dirs/counter_dirs, cmap='inferno')
        plt.colorbar()
        plt.clim(0,1)
        if ff:
            plt.savefig(("./savedir/FF_subset_dirs_"+str(par['tol'])+"_new.png"))
        else:
            plt.savefig("./savedir/Full_model_subset_dirs_"+str(par['tol'])+"_new.png")
        # plt.show()
