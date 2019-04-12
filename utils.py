"""
Various helping functions
"""
import pickle
import tensorflow as tf
import os
import scipy.io
import numpy as np
import subprocess
import glob


class FieldObj:
    pass


def extract_hps(w_path):
    _, _, _, _, _, exp_name, _, dataset, hps = w_path.split('/')
    nn_type = hps.split('nn_type=')[1].split(' ')[0]
    nn_type = 'fc1' if nn_type == 'mnist1' else nn_type  # correction for old runs
    lmbd = float(hps.split('lmbd=')[1].split(' ')[0])
    gamma_rb = float(hps.split('gamma_rb=')[1].split(' ')[0])
    gamma_db = float(hps.split('gamma_db=')[1].split(' ')[0])
    return exp_name, dataset, hps, nn_type, lmbd, gamma_rb, gamma_db


def get_max_epoch_in_tb(exp_name, dataset, hps_to_select):
    tb_pattern = 'exps/{}/tb/{}/{}'.format(exp_name, dataset, hps_to_select)
    path_tb_model = sorted(glob.glob(tb_pattern))[-1]  # take the last events file
    events_fname = os.listdir(path_tb_model + '/test/')[0]

    max_epoch = 0
    for e in tf.train.summary_iterator(path_tb_model + '/test/' + events_fname):
        if e.step > max_epoch:
            max_epoch = e.step
    print('max_epoch {}, restored_model {}'.format(max_epoch, path_tb_model))
    return max_epoch


class Logger:
    def __init__(self):
        self.lst_this_run = []
        self.lst_whole_exp = []

    def add(self, string):
        self.lst_this_run.append(string)
        print(string)

    def clear(self):
        self.lst_this_run = []

    def to_file(self, folder, this_run_file):
        if not os.path.exists(folder):
            os.makedirs(folder)
        if this_run_file is not None:
            with open(folder + this_run_file, 'w') as f:
                f.write('\n'.join(self.lst_this_run))


def save(var, f_name):
    with open(f_name, 'ab+') as file_write:
        pickle.dump(var, file_write)


def read(f_name):
    with open(f_name, 'rb') as file_read:
        return pickle.load(file_read)


def create_folders(folders):
    for folder in folders:
        current_folder = ''
        for component in folder.split('/')[:-1]:  # the last element of the list is ''
            current_folder += component + '/'
            if not os.path.exists(folder):
                os.makedirs(folder)


def create_hps_str(hps):
    # We can't take all hps for file names, so we select the most important ones
    hyperparam_str = "dataset={} model={} p_norm={} lmbd={} at_frac={} pgd_eps={} pgd_niter={} frac_perm={} loss={}".\
        format(hps.dataset, hps.model, hps.p, hps.lmbd, hps.at_frac, hps.pgd_eps, hps.pgd_niter, hps.frac_perm, hps.loss)
    return hyperparam_str


def save_results(log, saver, sess, metrics, epoch, hps, hps_str, cur_timestamp, base_path):
    # Example: exps/at_l2_basic_arch/logs/mnist/
    file_name = '{}_{}'.format(cur_timestamp, hps_str)
    logs_path = '{}/{}/{}/{}/'.format(base_path, hps.exp_name, file_name, 'logs')
    models_path = '{}/{}/{}/{}/'.format(base_path, hps.exp_name, file_name, 'models')
    mat_path = '{}/{}/{}/{}/'.format(base_path, hps.exp_name, file_name, 'mat')
    metrics_path = '{}/{}/{}/{}/'.format(base_path, hps.exp_name, file_name, 'metrics')

    create_folders([logs_path, models_path, mat_path, metrics_path])
    np.savetxt(metrics_path + file_name, np.array(metrics))  # save optimization metrics for future plots
    saver.save(sess, models_path + file_name, global_step=epoch)  # save TF model for future real robustness test
    log.to_file(logs_path, file_name)

    vars = tf.trainable_variables()
    var_val_dict = dict([(var.name, val) for var, val in zip(vars, sess.run(vars))])
    scipy.io.savemat(mat_path + file_name, mdict=var_val_dict)


def avg_tensor_list(tensor_list):
    tensors = tf.stack(axis=0, values=tensor_list)
    return tf.reduce_mean(tensors, axis=0)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

