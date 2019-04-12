import argparse
import tensorflow as tf
import numpy as np
import time
import utils
import data
import models
import regularizers
import utils_tf
from datetime import datetime

np.set_printoptions(suppress=True, precision=4)


def write_summary(writer, vals, names, cur_iter):
    for val, name in zip(vals, names):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=val)
        writer.add_summary(summary, cur_iter)

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
parser.add_argument('--augm', action='store_true', default=True,
                    help='Data augmentation: rotation, mirroring (not for mnist and gts).')
parser.add_argument('--p', type=str, default='inf',
                    help='P-norm of adv. examples for adv. training and evaluatiomn: 2 or inf')
parser.add_argument('--loss', type=str, default='max_conf',
                    help='uniform or small maximum confidence: uniform_conf or max_conf')



hps = parser.parse_args()  # returns a Namespace object, new fields can be set like hps.abc = 10
hps_str = utils.create_hps_str(hps)
hps.seed = 1
hps.p = {'1': 1, '2': 2, 'inf': np.inf}[hps.p]
hps.q = {1: np.inf, 2: 2, np.inf: 1}[hps.p]  # q norm is used in the denominator of MMR

base_path = 'exps' # The models will be stored under this folder.
cur_timestamp = str(datetime.now())[:-7]  # to get rid of milliseconds
tb_base_folder = '{}/{}/{}_{}/{}/'.format(base_path, hps.exp_name, cur_timestamp, hps_str, 'tb')
tb_train, tb_test = ['{}/{}'.format(tb_base_folder, s) for s in ['train', 'test']]
n_batches = 'all'  # how many batches from train and test set to take (we need this for debugging purposes)
batch_size_per_gpu = hps.batch_size // len(hps.gpus)

log = utils.Logger()
log.add('The script started on GPUs {} with hyperparameters: {}'.format(hps.gpus, hps_str))

hps.activation = utils_tf.f_activation(hps.activation_type)

dataset = data.datasets_dict[hps.dataset](hps.batch_size, hps.augm)
# The following datasets are used for evaluation during training. Feel free to comment out those that are not available. 
if hps.dataset == 'mnist':
    rub_dataset_names = ['fmnist', 'emnist', 'cifar10_gray']
elif hps.dataset == 'cifar10':
    rub_dataset_names = ['lsun_classroom', 'svhn', 'cifar100']#, 'imagenet_minus_cifar10'] imagenet_minus_cifar10 only works if the dataset is available locally.
elif hps.dataset == 'svhn':
    rub_dataset_names = ['cifar10', 'lsun_classroom', 'cifar100']#, 'imagenet_minus_cifar10']
elif hps.dataset == 'cifar100':
    rub_dataset_names = ['cifar10', 'svhn', 'lsun_classroom']#, 'imagenet_minus_cifar10']
else:
    raise ValueError('for this dataset rubbish datasets are not defined')
rub_datasets = [data.datasets_dict[ds_name](hps.batch_size, augm_flag=False) for ds_name in rub_dataset_names]
hps.n_train, hps.n_test, hps.n_classes = dataset.n_train, dataset.n_test, dataset.n_classes
hps.height, hps.width, hps.n_colors = dataset.height, dataset.width, dataset.n_colors
channel_mean, channel_std = np.array([[[[0.485, 0.456, 0.406]]]]), np.array([[[[0.229, 0.224, 0.225]]]])

# Setting up the learning rate decay
n_iter_per_epoch = hps.n_train // hps.batch_size
n_iter_total = hps.n_epochs * n_iter_per_epoch
decay1 = round(0.5 * n_iter_per_epoch * hps.n_epochs)
decay2 = round(0.75 * n_iter_per_epoch * hps.n_epochs)
decay3 = round(0.90 * n_iter_per_epoch * hps.n_epochs)
hps.lr_decay_n_updates = [decay1, decay2, decay3]
hps.lr_decay_coefs = [hps.lr, hps.lr / 10, hps.lr / 100, hps.lr / 1000]

# Define the computational graph
graph = tf.Graph()
with graph.as_default(), tf.device('/gpu:0'):
    tf_n_updates = tf.Variable(0, trainable=False)

    flag_train = tf.placeholder(tf.bool, name='is_training')
    at_flag_tf = tf.placeholder(tf.bool, name='at_flag')
    rub_flag_tf = tf.placeholder(tf.bool, name='rub_flag')
    adv_flag_tf = tf.placeholder(tf.bool, name='adv_flag')
    max_conf_flag_tf = tf.placeholder(tf.bool, name='max_conf_flag')
    at_frac_tf = tf.placeholder(tf.float32, name='at_frac')
    pgd_niter_tf = tf.placeholder_with_default(0, shape=())  # if just placeholder is used there is a bug in tf.while
    pgd_stepsize_tf = tf.cond(tf.not_equal(pgd_niter_tf, 0), lambda: 1.0 * hps.pgd_eps / tf.to_float(pgd_niter_tf),
                              lambda: 0.0)
    x_in = tf.placeholder(tf.float32, [None, hps.height, hps.width, hps.n_colors])
    y_in = tf.placeholder(tf.int32, [None, ])
    hps.n_ex = tf.shape(x_in)[0]

    lr_tf = tf.train.piecewise_constant(tf_n_updates, hps.lr_decay_n_updates, hps.lr_decay_coefs)

    opt_dict = {'sgd': tf.train.GradientDescentOptimizer(learning_rate=lr_tf),
                'momentum': tf.train.MomentumOptimizer(learning_rate=lr_tf, momentum=0.9),
                'adam': tf.train.AdamOptimizer(learning_rate=lr_tf)}
    optimizer = opt_dict[hps.opt]

    tower_grads = []
    imgs_per_gpu = hps.batch_size // len(hps.gpus)
    losses, regs, loss_upds, reg_upds = [], [], [], []
    acc_rates, avg_confs, acc_rate_upds, avg_conf_upds = [], [], [], []
    loss_clean_metric, loss_clean_metric_upd, loss_out_metric, loss_out_metric_upd, loss_total_metric, loss_total_metric_upd, reg_metric, reg_metric_upd = [], [], [], [], [], [], [], []
    losses_clean_metric, losses_clean_metric_upds, losses_out_metric, losses_out_metric_upds, losses_total_metric, losses_total_metric_upds, regs_metric, regs_metric_upds = [], [], [], [], [], [], [], []    
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len(hps.gpus)):
            with tf.device('/gpu:%d' % i), tf.name_scope('tower_%d' % i) as scope:
                id_from, id_to = i * imgs_per_gpu, i * imgs_per_gpu + imgs_per_gpu
                x, y = x_in[id_from:id_to], y_in[id_from:id_to]
                y = tf.one_hot(y, hps.n_classes)
                print('Batch on gpu {}: from {} to {}, {}'.format(i, id_from, id_to, x))

                model = models.models_dict[hps.model](hps)
                n_clean = tf.cast(batch_size_per_gpu, tf.int32)
                n_adv = tf.cast(at_frac_tf * batch_size_per_gpu, tf.int32)
                x_adv, y_adv = utils_tf.gen_adv(model, tf.concat([x, x, x], axis=0)[:n_adv], tf.concat([y, y, y], axis=0)[:n_adv], hps.p, hps.pgd_eps, pgd_niter_tf,
                                                pgd_stepsize_tf, rub_flag_tf, adv_flag_tf, hps.frac_perm,
                                                hps.lowpass, max_conf_flag_tf)
                # x_adv can be noise, adv. noise, adv. samples, or empty, depending on the feed_dict.
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
                logits_adv =  logits_c[:n_adv] #due to the concatenation above, the first n_adv values belong to x_adv and the rest is clean
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
                with tf.control_dependencies(update_ops):  # update_ops is executed before compute_gradients()
                    # Calculate the gradients for the batch of data on this tower.
                    grads_vars_in_tower = optimizer.compute_gradients(loss_tower)

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

                # Keep track of the gradients across all towers.
                tower_grads.append(grads_vars_in_tower)

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

    train = optimizer.apply_gradients(grads_vars, global_step=tf_n_updates, name='train_step')

    train_writer = tf.summary.FileWriter(tb_train, flush_secs=30)
    test_writer = tf.summary.FileWriter(tb_test, flush_secs=30)

    metric_vars = [var for var in tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES) if 'metric' in var.op.name]
    init_metric_vars = tf.variables_initializer(var_list=metric_vars)
    saver = tf.train.Saver()

    tb_summaries_op = tf.summary.merge_all()

    gpu_options = tf.GPUOptions(visible_device_list=str(hps.gpus)[1:-1], per_process_gpu_memory_fraction=hps.gpu_memory)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

with tf.Session(graph=graph, config=config) as sess:
    sess.run(tf.global_variables_initializer())  # run 'init' op
    metrics = []
    log.add('Session started with hyperparameters: {} \n'.format(hps_str))
    time_start = time.time()

    stats_names = ['err_rate', 'max_conf', 'loss', 'reg']  # for tensorboard
    ops_track = [err_rate, avg_conf, mean_loss_clean, mean_loss_out, mean_loss_total, mean_reg]
    ops_upd = [err_rate_upd, avg_conf_upd, mean_loss_clean_upd, mean_loss_out_upd, mean_loss_total_upd, mean_reg_upd]
    counter_summaries = 0
    for epoch in range(1, hps.n_epochs + 1):
        sess.run(init_metric_vars)  # important to reset the metrics variables

        for batch_x, batch_y in dataset.get_train_batches(n_batches=n_batches, shuffle=True):
            sess.run([train] + ops_upd,
                     feed_dict={x_in: batch_x, y_in: batch_y, flag_train: True, rub_flag_tf: True, adv_flag_tf: True, max_conf_flag_tf: hps.loss=='max_conf',
                                at_frac_tf: hps.at_frac, pgd_niter_tf: hps.pgd_niter})
        log.add('Epoch {} training is done, {:.2f} sec from start'.format(epoch, time.time() - time_start))
        eval_stats = {}
        if epoch % 2 == 0 or epoch <= 5 or epoch == hps.n_epochs:
            # Evaluating different metrics once per epoch
            err_rate_train, avg_conf_train, mean_loss_clean_train, mean_loss_out_train, mean_loss_total_train, mean_reg_train = sess.run([err_rate, avg_conf, mean_loss_clean, mean_loss_out, mean_loss_total, mean_reg])

            # test - clean
            test_data_iter = dataset.get_test_batches(n_batches=n_batches, shuffle=False)
            feed_dict = {rub_flag_tf: False, adv_flag_tf: False, max_conf_flag_tf: False, at_frac_tf: 0.0, pgd_niter_tf: 0}
            err_rate_test, avg_conf_test, mean_loss_clean_test, mean_loss_out_test, mean_loss_total_test, mean_reg_test = utils_tf.eval_in_batches(
                x_in, y_in, ops_track, ops_upd, sess, test_data_iter, flag_train, init_metric_vars, feed_dict)

            # Store the main train/test summaries
            write_summary(train_writer, [err_rate_train, mean_loss_clean_train, mean_loss_out_train, mean_loss_total_train, mean_reg_train, avg_conf_train],
                             ['main/error', 'main/loss_clean', 'main/loss_out', 'main/loss_total', 'main/reg', 'main/avg_conf'],
                             epoch)
            write_summary(test_writer, [err_rate_test, mean_loss_clean_test, mean_loss_out_test, mean_loss_total_test, mean_reg_test, avg_conf_test],
                             ['main/error', 'main/loss_clean', 'main/loss_out', 'main/loss_total', 'main/reg', 'main/avg_conf'],
                             epoch)
            metrics.append([err_rate_train, mean_loss_clean_train, mean_loss_out_train, mean_loss_total_train, err_rate_test, mean_loss_clean_test, mean_loss_out_test, mean_loss_total_test])

            # Evaluate only on 2 batches (256 examples) for all epochs, except the last (there full test set evaluation)
            n_adv_batches = 2 if epoch != hps.n_epochs else 'all'
            n_image_evals = 3 + len(rub_dataset_names)  # how many times we evaluate images

            # test - noise
            test_data_iter = dataset.get_test_batches(n_batches=5*n_adv_batches, shuffle=False)
            feed_dict = {rub_flag_tf: True, adv_flag_tf: False, max_conf_flag_tf: False, at_frac_tf: 1.0, pgd_niter_tf: 0}
            eval_stats['test_noise_plain'] = utils_tf.eval_in_batches(
                x_in, y_in, ops_track[:2], ops_upd[:2], sess, test_data_iter, flag_train, init_metric_vars, feed_dict)
            
            # test - noise + adv constant 80
            test_data_iter = dataset.get_test_batches(n_batches=5*n_adv_batches, shuffle=False)
            feed_dict = {rub_flag_tf: True, adv_flag_tf: True, max_conf_flag_tf: hps.loss=='max_conf', at_frac_tf: 1.0, pgd_niter_tf: 80}
            eval_stats['test_noise_adv_80'] = utils_tf.eval_in_batches(
                x_in, y_in, ops_track[:2], ops_upd[:2], sess, test_data_iter, flag_train, init_metric_vars, feed_dict)
                     
            # test - noise + adv same as training
            test_data_iter = dataset.get_test_batches(n_batches=5*n_adv_batches, shuffle=False)
            feed_dict = {rub_flag_tf: True, adv_flag_tf: True, max_conf_flag_tf: hps.loss=='max_conf', at_frac_tf: 1.0, pgd_niter_tf: hps.pgd_niter}
            eval_stats['test_noise_adv_train'] = utils_tf.eval_in_batches(
                x_in, y_in, ops_track[:2], ops_upd[:2], sess, test_data_iter, flag_train, init_metric_vars, feed_dict)
            
            # test - adv
            test_data_iter = dataset.get_test_batches(n_batches=n_adv_batches, shuffle=False)
            feed_dict = {rub_flag_tf: False, adv_flag_tf: True, max_conf_flag_tf: False, at_frac_tf: 1.0, pgd_niter_tf: 40}
            eval_stats['test_adv'] = utils_tf.eval_in_batches(
                x_in, y_in, ops_track[:2], ops_upd[:2], sess, test_data_iter, flag_train, init_metric_vars, feed_dict)
            
            # test - other datasets
            tb_tabs = ['test_{}_plain'.format(rub_dataset_name) for rub_dataset_name in rub_dataset_names]
            for rub_dataset, rub_dataset_name, tb_tab in zip(rub_datasets, rub_dataset_names, tb_tabs):
                test_data_iter = rub_dataset.get_test_batches(n_batches=n_adv_batches, shuffle=False)
                feed_dict = {rub_flag_tf: False, adv_flag_tf: False, max_conf_flag_tf: False, at_frac_tf: 0.0, pgd_niter_tf: 0}
                eval_stats[tb_tab] = utils_tf.eval_in_batches(
                    x_in, y_in, ops_track[:2], ops_upd[:2], sess, test_data_iter, flag_train, init_metric_vars,
                    feed_dict)
                
            msg = 'Epoch: {:d}  test err: {:.3%}  train err: {:.3%}  max conf plain: {:.3%}  ' \
                  'max conf noise plain: {:.3%} max conf noise adv n=n_train: {:.3%}  max conf noise adv n=80: {:.3%}  max conf adv: {:.3%}  '.format(
                epoch, err_rate_test, err_rate_train, avg_conf_test, eval_stats['test_noise_plain'][1],
                eval_stats['test_noise_adv_train'][1], eval_stats['test_noise_adv_80'][1], eval_stats['test_adv'][1])
            rub_datasets_msgs = ['max conf {}: {:.3%}'.format(rub_dt_name, eval_stats[tb_tab][1])
                                 for rub_dt_name, tb_tab in zip(rub_dataset_names, tb_tabs)]
            log.add(msg + '  '.join(rub_datasets_msgs))

            for key in eval_stats.keys():
                write_summary(test_writer, eval_stats[key],
                                 [key + '/' + stats_names[i] for i in range(len(stats_names))], epoch)

        if epoch % 10 == 0 or epoch == hps.n_epochs:
            utils.save_results(log, saver, sess, metrics, epoch, hps, hps_str, cur_timestamp, base_path)

    for writer in [train_writer, test_writer]:
        writer.close()

log.add('Worker done in {:.2f} min ({})\n\n'.format((time.time() - time_start) / 60, hps_str))
