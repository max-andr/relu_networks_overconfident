import time
time_start = time.time()
import tensorflow as tf
import numpy as np
import utils
import data
import models
import argparse
import regularizers
import tensorboard as tb
import utils_tf
from datetime import datetime

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib  as mpl
import pandas as pd
from collections import OrderedDict
import os

np.set_printoptions(suppress=True, precision=4)
tf.logging.set_verbosity(tf.logging.WARN)


parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--gpus', nargs='+', type=int, default=[2], help='GPU indices.')
parser.add_argument('--gpu_memory', type=float, default=0.15,
                    help='GPU memory fraction to use. Deadline time: 0.45 for cifar10')
parser.add_argument('--exp_name', type=str, default='test',
                    help='Name of the experiment, which is used to save the results/metrics/model in a certain folder.')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='moons, mnist, cifar10, gts (German traffic roadsign dataset), imagenet')
parser.add_argument('--model', type=str, default='lenet',
                    help='NN type: fc1, lenet, resnet, vgg_small, vgg_large, resnet_small, resnet_middle, resnet_large,'
                         ' densenet_large')
parser.add_argument('--activation_type', type=str, default='relu', help='softplus, relu or sigmoid')
parser.add_argument('--n_epochs', type=int, default=120, help='Number of epochs.')
parser.add_argument('--lmbd', type=float, default=0.0, help='Lambda for the weight decay.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--opt', type=str, default='adam', help='Optimization method: adam, sgd, sgdm')
parser.add_argument('--frac_perm', type=float, default=0.5,
                    help='Fraction of permuted images in a batch (if at_frac > 0).')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size, which is also used for evaluating train/validation/test error by parts.')
parser.add_argument('--pgd_eps', type=float, default=0.3,
                    help='The radius of Lp-ball in PGD attack used in adversarial training (AT).')
parser.add_argument('--pgd_niter', type=int, default=5,
                    help='Number of iterations in PGD attack used in adversarial training (AT).')
parser.add_argument('--at_frac', type=float, default=0.0,
                    help='Fraction of adv. or rubbish examples in a batch [0..1].')
parser.add_argument('--lowpass', action='store_true', default=True,
                    help='Whether to apply lowpass filter during training and evaluation.')
parser.add_argument('--augm', action='store_true', default=False,
                    help='Data augmentation: rotation, mirroring (not for mnist and gts).')
parser.add_argument('--p', type=str, default='inf',
                    help='P-norm of adv. examples for adv. training and evaluatiomn: 2 or inf')
parser.add_argument('--loss', type=str, default='max_conf',
                    help='uniform or small maximum confidence: uniform_conf or max_conf')

parser.add_argument('--model_folder', type=str, default=None,
                    help='The folder of the stored model. It has .data, .index, .meta and checkpoint files.')
parser.add_argument('--model_name', type=str, default=None,
                    help='The model from the folder that is to be used. Ends in something like -100')
parser.add_argument('--pgd_step', type=float,
                    help='Step size of the pgd attack.')



hps = parser.parse_args()  # returns a Namespace object, new fields can be set like hps.abc = 10
hps_str = utils.create_hps_str(hps)
hps.seed = 1
hps.p = {'1': 1, '2': 2, 'inf': np.inf}[hps.p]
hps.q = {1: np.inf, 2: 2, np.inf: 1}[hps.p]  # q norm is used in the denominator of MMR

model_folder=hps.model_folder
model_name=hps.model_name

base_path = '/home/jb/ML/rub_ex_master/exps'
cur_timestamp = str(datetime.now())[:-7]  # to get rid of milliseconds
tb_base_folder = '{}/{}/{}_{}/{}/'.format(base_path, hps.exp_name, cur_timestamp, hps_str, 'tb')
tb_train, tb_test = ['{}/{}'.format(tb_base_folder, s) for s in ['train', 'test']]
n_batches = 'all'  # how many batches from train and test set to take (we need this for debugging purposes)
batch_size_per_gpu = hps.batch_size // len(hps.gpus)

log = utils.Logger()
log.add('The script started on GPUs {} with hyperparameters: {} and pgd step size {}'.format(hps.gpus, hps_str, hps.pgd_step))
hps.activation = utils_tf.f_activation(hps.activation_type)
dataset = data.datasets_dict[hps.dataset](hps.batch_size, hps.augm)
if hps.dataset == 'mnist':
    rub_dataset_names = ['fmnist', 'emnist', 'cifar10_gray']
elif hps.dataset == 'cifar10':
    rub_dataset_names = ['svhn', 'cifar100', 'lsun_classroom']  # We also evaluated on the custom dataset 'imagenet_minus_cifar10' which we only have locally.
elif hps.dataset == 'svhn':
    rub_dataset_names = ['cifar10', 'cifar100','lsun_classroom']  #'imagenet_minus_cifar10', 
elif hps.dataset == 'cifar100':
    rub_dataset_names = ['cifar10', 'svhn','lsun_classroom']  #'imagenet_minus_cifar10', 
else:
    raise ValueError('for this dataset rubbish datasets are not defined')
rub_datasets = [data.datasets_dict[ds_name](hps.batch_size, augm_flag=False) for ds_name in rub_dataset_names if ds_name != 'lsun_classroom']
if 'lsun_classroom' in rub_dataset_names: # Here we take 10000 examples from the training dataset, as the test set is small.
    rub_datasets.append(data.datasets_dict['lsun_classroom'](hps.batch_size, augm_flag=False,  test_only=False)) 
    
hps.n_train, hps.n_test, hps.n_classes = dataset.n_train, dataset.n_test, dataset.n_classes
hps.height, hps.width, hps.n_colors = dataset.height, dataset.width, dataset.n_colors
channel_mean, channel_std = np.array([[[[0.485, 0.456, 0.406]]]]), np.array([[[[0.229, 0.224, 0.225]]]])

# Define the computational graph
graph = tf.Graph()
with graph.as_default(), tf.device('/gpu:0'):
    #tf.set_random_seed(1) #does not ensure reproducibility
    tf_n_updates = tf.Variable(0, trainable=False)

    flag_train = tf.placeholder(tf.bool, name='is_training')
    at_flag_tf = tf.placeholder(tf.bool, name='at_flag')
    rub_flag_tf = tf.placeholder(tf.bool, name='rub_flag')
    adv_flag_tf = tf.placeholder(tf.bool, name='adv_flag')
    max_conf_flag_tf = tf.placeholder(tf.bool, name='max_conf_flag')
    at_frac_tf = tf.placeholder(tf.float32, name='at_frac')
    pgd_niter_tf = tf.placeholder_with_default(0, shape=())  # if just placeholder is used - weird bug in tf.while
    pgd_stepsize_tf = tf.to_float(hps.pgd_step)
    x_in = tf.placeholder(tf.float32, [None, hps.height, hps.width, hps.n_colors])
    y_in = tf.placeholder(tf.int32, [None, ])
    hps.n_ex = tf.shape(x_in)[0]

    tower_grads = []
    imgs_per_gpu = hps.batch_size // len(hps.gpus)
    losses, regs, loss_upds, reg_upds = [], [], [], []
    acc_rates, avg_confs, acc_rate_upds, avg_conf_upds = [], [], [], []
    loss_clean_metric, loss_clean_metric_upd, loss_out_metric, loss_out_metric_upd, loss_total_metric, loss_total_metric_upd, reg_metric, reg_metric_upd = [], [], [], [], [], [], [], []
    losses_clean_metric, losses_clean_metric_upds, losses_out_metric, losses_out_metric_upds, losses_total_metric, losses_total_metric_upds, regs_metric, regs_metric_upds = [], [], [], [], [], [], [], []    
    logits_list, probs_list, labels_list = [], [], []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len(hps.gpus)):
            with tf.device('/gpu:%d' % i), tf.name_scope('tower_%d' % i) as scope:
                id_from, id_to = i * imgs_per_gpu, i * imgs_per_gpu + imgs_per_gpu
                x, y = x_in[id_from:id_to], y_in[id_from:id_to]
                y = tf.one_hot(y, hps.n_classes)  # needed for rubbish adversarial training
                print('Batch on gpu {}: from {} to {}, {}'.format(i, id_from, id_to, x))

                model = models.models_dict[hps.model](hps)
                n_clean = tf.cast(batch_size_per_gpu, tf.int32)
                n_adv = tf.cast(at_frac_tf * batch_size_per_gpu, tf.int32)
                x_adv, y_adv = utils_tf.gen_adv(model, tf.concat([x, x, x], axis=0)[:n_adv], tf.concat([y, y, y], axis=0)[:n_adv], hps.p, hps.pgd_eps, pgd_niter_tf,
                                                pgd_stepsize_tf, rub_flag_tf, adv_flag_tf, hps.frac_perm,
                                                hps.lowpass, max_conf_flag_tf)
                # During training or if x_adv is empty: use clean and adversarial/rubbish examples.
                # During evaluation: use adversarial/rubbish examples only (which may still be clean iff
                # rub_flag_tf: False and adv_flag_tf: False).
                x_c, y_c = tf.cond(tf.logical_or(flag_train, tf.equal(tf.shape(x_adv)[0], 0)),
                               lambda: (tf.concat([x_adv, x], axis=0), tf.concat([y_adv, y], axis=0)),
                               lambda: (x_adv, y_adv))

                # Reuse the model declared for adv. examples here, and also for the next tower.
                tf.get_variable_scope().reuse_variables()

                logits_c = model.get_logits(x_c, flag_train=flag_train)
                probs_c = tf.nn.softmax(logits_c)
                logits_adv =  logits_c[:n_adv]
                logits = logits_c[n_adv:]
                maxclass = tf.argmax(logits_adv, axis=-1)
                loss_adv = tf.cond(max_conf_flag_tf,
                                        lambda: -model.get_loss(logits_adv, tf.one_hot(maxclass, logits.shape[-1])),
                                        lambda:  model.get_loss(logits_adv, y_c[:n_adv])
                                        )
                loss_clean = model.get_loss(logits, y_c[n_adv:])
                reg_plain = regularizers.weight_decay()
                
                loss_ce = (loss_clean + loss_adv)/tf.to_float(n_clean + n_adv)
                
                loss_tower = loss_ce + hps.lmbd * reg_plain

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                # Here are just different metrics for tensorboard
                loss_clean_metric, loss_clean_metric_upd = tf.metrics.mean(loss_clean/tf.to_float(n_clean), name='loss_clean_metric')
                loss_out_metric, loss_out_metric_upd = tf.metrics.mean(loss_adv/tf.to_float(n_adv), name='loss_out_metric')
                loss_total_metric, loss_total_metric_upd = tf.metrics.mean(loss_tower, name='loss_total_metric')
                reg_metric, reg_metric_upd = tf.metrics.mean(reg_plain, name='reg_metric')

                losses_clean_metric, losses_clean_metric_upds = losses_clean_metric + [loss_clean_metric], losses_clean_metric_upds + [loss_clean_metric_upd]
                losses_out_metric, losses_out_metric_upds = losses_out_metric + [loss_out_metric], losses_out_metric_upds + [loss_out_metric_upd]
                losses_total_metric, losses_total_metric_upds = losses_total_metric + [loss_total_metric], losses_total_metric_upds + [loss_total_metric_upd]
                regs_metric, regs_metric_upds = regs_metric + [reg_metric], regs_metric_upds + [reg_metric_upd]

                acc_rate, acc_rate_upd = tf.metrics.mean(tf.nn.in_top_k(
                    predictions=logits_c, targets=tf.argmax(y_c, 1), k=1),
                    name='top1_err_metric')
                avg_conf, avg_conf_upd = tf.metrics.mean(tf.reduce_max(probs_c, axis=1), name='avg_conf_metric')
                acc_rates, acc_rate_upds = acc_rates + [acc_rate], acc_rate_upds + [acc_rate_upd]
                avg_confs, avg_conf_upds = avg_confs + [avg_conf], avg_conf_upds + [avg_conf_upd]

                #Lists of the predictions
                logits_list += [logits_c]
                probs_list += [probs_c]
                labels_list += [y_c]           
                
    grads_vars = utils_tf.average_gradients(tower_grads)
    # Average error rates
    mean_loss_clean = tf.reduce_mean(tf.stack(losses_clean_metric))
    mean_loss_out = tf.reduce_mean(tf.stack(losses_out_metric))
    mean_loss_total = tf.reduce_mean(tf.stack(losses_total_metric))
    mean_reg = tf.reduce_mean(tf.stack(regs_metric))
    err_rate = 1 - tf.reduce_mean(tf.stack(acc_rates))
    avg_conf = tf.reduce_mean(tf.stack(avg_confs))
    mean_loss_clean_upd, mean_loss_out_upd, mean_loss_total_upd, mean_reg_upd = tf.group(*losses_clean_metric_upds), tf.group(*losses_out_metric_upds), tf.group(*losses_total_metric_upds), tf.group(*regs_metric_upds)
    err_rate_upd, avg_conf_upd = tf.group(*acc_rate_upds), tf.group(*avg_conf_upds)

    train_writer = tf.summary.FileWriter(tb_train, flush_secs=30)
    test_writer = tf.summary.FileWriter(tb_test, flush_secs=30)

    metric_vars = [var for var in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES) if 'metric' in var.op.name]
    init_metric_vars = tf.variables_initializer(var_list=metric_vars)
    saver = tf.train.Saver()

    tb_summaries_op = tf.summary.merge_all()

    gpu_options = tf.GPUOptions(visible_device_list=str(hps.gpus)[1:-1], per_process_gpu_memory_fraction=hps.gpu_memory)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)


pgd_eval_niter = hps.pgd_niter
def get_all_preds(X_tf, Y_tf, sess, batch_iterator, flag_train_tf,
                    init_metric_vars, feed_dict):
    """Get all predictions for a dataset by running it in large batches."""
    sess.run(init_metric_vars)
    all_probs, all_labels = [], []
    for batch_x, batch_y in batch_iterator:
        bprobs_list, blabels_list = sess.run([probs_list, labels_list], feed_dict={X_tf: batch_x, Y_tf: batch_y, flag_train: False, **feed_dict})
        all_probs += bprobs_list
        all_labels += blabels_list
    return np.concatenate(np.array(all_probs), axis=0), np.concatenate(np.array(all_labels), axis=0)
def tru(a):
    '''If rubbish inputs are labeled with C*[1/C], they will be False. This is meant as in true positives and false positives, where true means in-distribution.'''
    return(np.isin(a[:,0], [0,1]))
def max_conf(a):
    '''Returns the highest probability in a prediction.'''
    return np.max(a,axis=1)


imported_meta = tf.train.import_meta_graph(model_folder + model_name + ".meta")

#All the predictions are run in a session and made available as numpy arrays.
with tf.Session(graph=graph, config=config) as sess:
    imported_meta.restore(sess, model_folder + model_name)
    metrics = []
    
    ops_track = [err_rate, avg_conf, mean_loss_clean, mean_loss_out, mean_loss_total, mean_reg]
    ops_upd = [err_rate_upd, avg_conf_upd, mean_loss_clean_upd, mean_loss_out_upd, mean_loss_total_upd, mean_reg_upd]
    
    sess.run(init_metric_vars)

    # test - clean
    test_data_iter = dataset.get_test_batches(n_batches=n_batches, shuffle=False)
    feed_dict = {rub_flag_tf: False, adv_flag_tf: False, max_conf_flag_tf: False, at_frac_tf: 0.0, pgd_epsilon_tf: 0.0, pgd_niter_tf: 0}
    err_rate_test, avg_conf_test, mean_loss_clean_test, mean_loss_out_test, mean_loss_total_test, mean_reg_test = utils_tf.eval_in_batches(
        x_in, y_in, ops_track, ops_upd, sess, test_data_iter, flag_train, init_metric_vars, feed_dict)
    print('Error rate on the clean test set: ', err_rate_test)
    print('Mean max. confidence on the clean test set: ', avg_conf_test)

    #get preds - clean
    test_data_iter = dataset.get_test_batches(n_batches=n_batches, shuffle=False)
    feed_dict = {rub_flag_tf: False, adv_flag_tf: False, max_conf_flag_tf: False, at_frac_tf: 0.0, pgd_epsilon_tf: 0.0, pgd_niter_tf: 0}
    probs_clean, lbls_clean = get_all_preds(
        x_in, y_in, sess, test_data_iter, flag_train, init_metric_vars, feed_dict)
    tru_clean = tru(lbls_clean)
    conf_clean = max_conf(probs_clean)
        
    #get preds - noise
    test_data_iter = dataset.get_test_batches(n_batches=n_batches, shuffle=False)
    feed_dict_noise = {rub_flag_tf: True, adv_flag_tf: False, max_conf_flag_tf: False, at_frac_tf: 1.0, pgd_epsilon_tf: 0.0, pgd_niter_tf: 0}
    probs_noise, lbls_noise = get_all_preds(
        x_in, y_in, sess, test_data_iter, flag_train, init_metric_vars, feed_dict_noise)
    tru_noise = tru(lbls_noise)
    conf_noise = max_conf(probs_noise)
    
    # get preds - noise + adv
    test_data_iter = dataset.get_test_batches(n_batches=n_batches, shuffle=False)
    feed_dict_noise_adv = {rub_flag_tf: True, adv_flag_tf: True, max_conf_flag_tf: hps.loss=='max_conf', at_frac_tf: 1.0, pgd_epsilon_tf: hps.pgd_eps, pgd_niter_tf: pgd_eval_niter}
    probs_noise_adv, lbls_noise_adv = get_all_preds(
        x_in, y_in, sess, test_data_iter, flag_train, init_metric_vars, feed_dict_noise_adv)
    tru_noise_adv = tru(lbls_noise_adv)
    conf_noise_adv = max_conf(probs_noise_adv)
     
    # get preds - noise + adv 2.5
    test_data_iter = dataset.get_test_batches(n_batches=n_batches, shuffle=False)
    feed_dict_noise_adv_more = {rub_flag_tf: True, adv_flag_tf: True, max_conf_flag_tf: hps.loss=='max_conf', at_frac_tf: 1.0, pgd_epsilon_tf: hps.pgd_eps, pgd_niter_tf: 2.5*pgd_eval_niter}
    probs_noise_adv_more, lbls_noise_adv_more = get_all_preds(
        x_in, y_in, sess, test_data_iter, flag_train, init_metric_vars, feed_dict_noise_adv_more)
    tru_noise_adv_more = tru(lbls_noise_adv_more)
    conf_noise_adv_more = max_conf(probs_noise_adv_more)

    # get preds - adv
    test_data_iter = dataset.get_test_batches(n_batches=n_batches, shuffle=False)
    feed_dict_adv = {rub_flag_tf: False, adv_flag_tf: True, max_conf_flag_tf: False, at_frac_tf: 1.0, pgd_epsilon_tf: hps.pgd_eps, pgd_niter_tf: pgd_eval_niter}
    probs_adv, lbls_adv = get_all_preds(
        x_in, y_in, sess, test_data_iter, flag_train, init_metric_vars, feed_dict_adv)
    tru_adv = tru(lbls_adv)
    conf_adv = max_conf(probs_adv)

    #get preds - dsets
    def get_preds(dset):
        test_data_iter = dset.get_test_batches(n_batches=n_batches, shuffle=False)
        feed_dict = {rub_flag_tf: False, adv_flag_tf: False, max_conf_flag_tf: False, at_frac_tf: 0.0, pgd_epsilon_tf: 0.0, pgd_niter_tf: 0}
        probs_dset, lbls_dset = get_all_preds(
            x_in, y_in, sess, test_data_iter, flag_train, init_metric_vars, feed_dict)
        return probs_dset, lbls_dset
    
        tru_dset = tru(lbls_dset) * False
        conf_dset = max_conf(probs_dset)
        return tru_dset, conf_dset
    
    probs_rdsets = dict([])
    lbls_rdsets = dict([])
    tru_rdsets = dict([])
    conf_rdsets = dict([])
    for rdset, rdset_name in zip(rub_datasets, rub_dataset_names):
        probs_rdsets[rdset_name], lbls_rdsets[rdset_name] = get_preds(rdset)
        tru_rdsets[rdset_name] = tru(lbls_rdsets[rdset_name]) * False
        conf_rdsets[rdset_name] = max_conf(probs_rdsets[rdset_name])

log.add('Evaluation values collected in {:.2f} min ({})\n\n'.format((time.time() - time_start) / 60, hps_str))

save_folder = model_folder[:-7] + 'evals2/'
np.savez(save_folder + 'probs', probs_clean=probs_clean, probs_noise=probs_noise, probs_noise_adv=probs_noise_adv, probs_noise_adv_more=probs_noise_adv_more, probs_adv=probs_adv, probs_rdsets=probs_rdsets)
np.savez(save_folder + 'lbls', lbls_clean=lbls_clean, lbls_noise=lbls_noise, lbls_noise_adv=lbls_noise_adv, lbls_noise_adv_more=lbls_noise_adv_more, lbls_adv=lbls_adv, lbls_rdsets=lbls_rdsets)
np.savez(save_folder + 'cleanperf', err_rate_test=err_rate_test, avg_conf_test=avg_conf_test, mean_loss_clean_test=mean_loss_clean_test, mean_loss_out_test=mean_loss_out_test, mean_loss_total_test=mean_loss_total_test, mean_reg_test=mean_reg_test)