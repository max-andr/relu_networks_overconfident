import time
time_start = time.time()
import numpy as np
import argparse
from datetime import datetime

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib  as mpl
import pandas as pd
from collections import OrderedDict
import os

def tru(a):
    '''If rubbish inputs are labeled with C*[1/C], they will be False. This is meant as in true positives and false positives, where true means in-distribution.'''
    return(np.isin(a[:,0], [0,1]))
def max_conf(a):
    '''Returns the highest probability in a prediction.'''
    return np.max(a,axis=1)

def check_adv_robustness(probs_adv, lbls_adv):
    '''For adversarial samples, we have two interesting statistics. Firstly, we check how robust the model is, which means for how many inputs the correct label of the original sample is predicted. If the model predicts uniform probability for all classes, this does not count as the correct class. Since many argmax functions return the 0th index for a completely uniform prediction, we set a threshold of uniform + 1%*uniform_value so no input with label 0 is considered resisted by chance.
    Secondly, we check if the model detects adversarials. For this, we only look at the confidences of samples that are not resisted.
    '''
    threshold_rej = (1.01)/lbls_adv.shape[-1]
    probs_adv_rej = np.insert(probs_adv, 10, threshold_rej, axis=1) #If all predictions are uniform, the 10th will be the argmax and thus not the original label.

    classpreds_adv = np.argmax(probs_adv_rej, axis=1)
    classlbls_adv = np.argmax(lbls_adv, axis=1)

    corrects_adv = (classpreds_adv == classlbls_adv)

    resistance_adv = np.sum(corrects_adv)/len(classlbls_adv)

    probs_adv_not_resisted = probs_adv[(corrects_adv==False)]
    confs_adv_not_resisted = max_conf(probs_adv_not_resisted)
    tru_adv_not_resisted = corrects_adv[(corrects_adv==False)]
    return resistance_adv, probs_adv_not_resisted, confs_adv_not_resisted, tru_adv_not_resisted

def roc(tru_set, conf_set, tru_clean, conf_clean):
    tru_with_clean = np.concatenate([tru_set, tru_clean])
    conf_with_clean = np.concatenate([conf_set, conf_clean])
    return roc_curve(tru_with_clean, conf_with_clean, pos_label=True), roc_auc_score(tru_with_clean, conf_with_clean)

def fpr_at_95_tpr(conf_t, conf_f):
    TPR = 95
    PERC = np.percentile(conf_t, 100-TPR)
    FP = np.sum(conf_f >=  PERC)
    FPR = np.sum(conf_f >=  PERC)/len(conf_f)
    return FPR, PERC
    
np.set_printoptions(suppress=True, precision=4)


parser = argparse.ArgumentParser(description='Specify data location.')
parser.add_argument('--model_folder', type=str, default=None,
                    help='The folder of the stored model. Above it, there is an eval folder with all the relevant files.')
parser.add_argument('--save_all_evals', type=str, default='evals.csv',
                    help='File to add the evaluation results to.')

hps = parser.parse_args()  # returns a Namespace object, new fields can be set like hps.abc = 10
model_folder = hps.model_folder
model_name = model_folder.split('/')[-3]
save_folder = model_folder[:-7] + 'evals/'

probs = np.load(save_folder + 'probs.npz')
probs_clean, probs_noise, probs_noise_adv, probs_noise_adv_more, probs_adv, probs_rdsets = probs['probs_clean'], probs['probs_noise'], probs['probs_noise_adv'], probs['probs_noise_adv_more'], probs['probs_adv'], probs['probs_rdsets'].item()
rub_dataset_names = rub_dataset_names = sorted(list(probs_rdsets.keys()))
conf_clean, conf_noise, conf_noise_adv, conf_noise_adv_more, conf_adv = max_conf(probs_clean), max_conf(probs_noise), max_conf(probs_noise_adv), max_conf(probs_noise_adv_more), max_conf(probs_adv)
conf_rdsets = dict([])
for rdset_name in rub_dataset_names:
    conf_rdsets[rdset_name] = max_conf(probs_rdsets[rdset_name])

lbls = np.load(save_folder + 'lbls.npz')
lbls_clean, lbls_noise, lbls_noise_adv, lbls_noise_adv_more, lbls_adv, lbls_rdsets = lbls['lbls_clean'], lbls['lbls_noise'], lbls['lbls_noise_adv'], lbls['lbls_noise_adv_more'], lbls['lbls_adv'], lbls['lbls_rdsets'].item()
tru_clean, tru_noise, tru_noise_adv, tru_noise_adv_more, tru_adv = tru(lbls_clean), tru(lbls_noise), tru(lbls_noise_adv), tru(lbls_noise_adv_more), tru(lbls_adv)
tru_rdsets = dict([])
for rdset_name in rub_dataset_names:
    tru_rdsets[rdset_name] = tru(lbls_rdsets[rdset_name])*False

cleanperf = np.load(save_folder + 'cleanperf.npz')
err_rate_test, avg_conf_test, mean_loss_clean_test, mean_loss_out_test, mean_loss_total_test, mean_reg_test = cleanperf['err_rate_test'], cleanperf['avg_conf_test'], cleanperf['mean_loss_clean_test'], cleanperf['mean_loss_out_test'], cleanperf['mean_loss_total_test'], cleanperf['mean_reg_test'] 


resistance_adv, probs_adv_not_resisted, confs_adv_not_resisted, tru_adv_not_resisted = check_adv_robustness(probs_adv, lbls_adv)

label_size = 15
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 

print('conf_clean')
plt.hist(conf_clean, bins=100, range=(0, 1), log=True, label= str(conf_clean.shape[0])+' clean test images')
plt.ylim(ymin=0.9, ymax=11000)
plt.savefig(save_folder + 'conf_clean', bbox_inches='tight')
plt.show()

for rdset_name in rub_dataset_names:
    print(rdset_name)
    plt.hist(conf_rdsets[rdset_name], bins=100, range=(0, 1), log=True, label= str(conf_rdsets[rdset_name].shape[0]).upper() + ' ' + rdset_name + ' images')
    #plt.legend()
    plt.ylim(ymin=0.9, ymax=11000)
    plt.savefig(save_folder + 'conf_' + rdset_name, bbox_inches='tight')
    plt.show()
    
print('conf_noise')
plt.hist(conf_noise, bins=100, range=(0, 1), log=True, label= str(conf_noise.shape[0])+' noise samples')
plt.ylim(ymin=0.9, ymax=11000)
plt.savefig(save_folder + 'conf_noise', bbox_inches='tight')
plt.show()

print('conf_noise_adv')
plt.hist(conf_noise_adv, bins=100, range=(0, 1), log=True, label= str(conf_noise_adv.shape[0])+' noise + adv. PGD samples')
plt.ylim(ymin=0.9, ymax=11000)
plt.savefig(save_folder + 'conf_noise_adv', bbox_inches='tight')
plt.show()

print('conf_noise_adv_more')
plt.hist(conf_noise_adv_more, bins=100, range=(0, 1), log=True, label= str(conf_noise_adv_more.shape[0])+' noise + adv. PGD samples')
plt.ylim(ymin=0.9, ymax=11000)
plt.savefig(save_folder + 'conf_noise_adv_more', bbox_inches='tight')
plt.show()

print('conf_adv')
plt.hist(conf_adv, bins=100, range=(0, 1), log=True, label= str(conf_adv.shape[0])+' test img. adv. PGD samples')
plt.ylim(ymin=0.9, ymax=11000)
plt.savefig(save_folder + 'conf_adv', bbox_inches='tight')
plt.show()


(fpr_noise, tpr_noise, thresholds_noise), auc_score_noise = roc(tru_noise, conf_noise, tru_clean, conf_clean)
(fpr_noise_adv, tpr_noise_adv, thresholds_noise_adv), auc_score_noise_adv = roc(tru_noise_adv, conf_noise_adv, tru_clean, conf_clean)
(fpr_noise_adv_more, tpr_noise_adv_more, thresholds_noise_adv_more), auc_score_noise_adv_more = roc(tru_noise_adv_more, conf_noise_adv_more, tru_clean, conf_clean)
(fpr_adv, tpr_adv, thresholds_adv), auc_score_adv = roc(tru_adv_not_resisted, confs_adv_not_resisted, tru_clean, conf_clean)

fpr_rdsets = dict([])
tpr_rdsets = dict([])
thresholds_rdsets = dict([])
auc_score_rdsets = dict([])
for rdset in rub_dataset_names:
    (fpr_rdsets[rdset], tpr_rdsets[rdset], thresholds_rdsets[rdset]), auc_score_rdsets[rdset] = roc(tru_rdsets[rdset], conf_rdsets[rdset], tru_clean, conf_clean)


plt.plot(fpr_noise, tpr_noise, label='Noise', linewidth=3.5, linestyle='-', alpha=1)
#plt.plot(fpr_noise_adv, tpr_noise_adv, label='Adv. Noise 80', linewidth=2.5, linestyle='-', alpha=1)
plt.plot(fpr_noise_adv_more, tpr_noise_adv_more, label='Adv. Noise', linewidth=2.5, linestyle='-', alpha=1)
plt.plot(fpr_adv, tpr_adv, label='Adv. Samples', linewidth=3, alpha=1)

for rdset in rub_dataset_names:
    plt.plot(fpr_rdsets[rdset], tpr_rdsets[rdset], label=rdset.upper(), linewidth=3, linestyle='--', alpha=1)
plt.grid()
plt.legend()
plt.savefig(save_folder + 'rocs')
plt.show()


print('ROC AUC score Noise: ', auc_score_noise)
print('ROC AUC score Noise + adv 80: ', auc_score_noise_adv)
print('ROC AUC score Noise + adv 200: ', auc_score_noise_adv_more)
print('ROC AUC score adv: ', auc_score_adv)
for rdset in rub_dataset_names:
    print('ROC AUC score {0}: '.format(rdset), auc_score_rdsets[rdset])
def fpr_at_95_tpr(conf_t, conf_f):
    TPR = 95
    PERC = np.percentile(conf_t, 100-TPR)
    FP = np.sum(conf_f >=  PERC)
    FPR = np.sum(conf_f >=  PERC)/len(conf_f)
    return FPR, PERC
    #print('TPR is at {0}% for confidence {1}.'.format(TPR, PERC))
    #print('False Positives: {0:.2f}%'.format(FPR*100))
noise95, conf_tpr95 = fpr_at_95_tpr(conf_clean, conf_noise)
noise_adv95, _ = fpr_at_95_tpr(conf_clean, conf_noise_adv)
noise_adv_more95, _ = fpr_at_95_tpr(conf_clean, conf_noise_adv_more)
adv95, _ = fpr_at_95_tpr(conf_clean, confs_adv_not_resisted)

frp95_rdsets = dict([])
for rdset in rub_dataset_names:
    frp95_rdsets[rdset], _ = fpr_at_95_tpr(conf_clean, conf_rdsets[rdset])
TPR = 95
print('TPR is at {0}% for confidence {1}.'.format(TPR, conf_tpr95))
fpr95_saves = ['conf_tpr95', 'noise95', 'noise_adv95', 'noise_adv_more95', 'adv95']
fpr95_list = []
for v in fpr95_saves:
    fpr95_list.append((v,locals()[v]))
for rdset in rub_dataset_names:
    fpr95_list.append((rdset+'95', frp95_rdsets[rdset]))
fpr95_dict = OrderedDict(fpr95_list)
import pandas as pd
from collections import OrderedDict
import os
mmc_list = [('Mean maximum confidence on clean', np.mean(conf_clean)),
('Mean maximum confidence on noise', np.mean(conf_noise)),
('Mean maximum confidence on noise_adv (noise + pgd)', np.mean(conf_noise_adv)),
('Mean maximum confidence on noise_adv_more (noise + pgd)', np.mean(conf_noise_adv_more)),
('Mean maximum confidence on adv (clean + pgd not_resisted)', np.mean(confs_adv_not_resisted))]

for rdset in rub_dataset_names:
    mmc_list.append(('Mean maximum confidence on ' + rdset, np.mean(conf_rdsets[rdset])))
mmc_dict = OrderedDict(mmc_list)
eval_saves = ['model_name', 'err_rate_test', 'auc_score_fmnist', 'auc_score_emnist', 'auc_score_noise', 'auc_score_noise_adv', 'auc_score_noise_adv_more', 'auc_score_adv', 'resistance_adv']
eval_saves_general = ['model_name', 'err_rate_test', 'resistance_adv']
eval_saves_auc = ['auc_score_noise', 'auc_score_noise_adv', 'auc_score_noise_adv_more', 'auc_score_adv', 'resistance_adv']
vars_list_general = []
for v in eval_saves_general:
    vars_list_general.append((v,locals()[v]))

    
vars_list_auc = []
for v in eval_saves_auc:
    vars_list_auc.append((v,locals()[v]))
for rdset in rub_dataset_names:
    vars_list_auc.append(('auc_score_' + rdset, auc_score_rdsets[rdset]))
vars_dict = OrderedDict(vars_list_general)
vars_dict.update(vars_list_auc)
vars_dict.update(mmc_dict)
vars_dict.update(fpr95_dict)
df = pd.DataFrame(vars_dict, index=[model_name])
save_all_evals = hps.save_all_evals
if os.path.exists(save_all_evals):
    df_old = pd.read_csv(save_all_evals)
    df_new = df_old.append(df)
else:
    df_new = df
df_new.to_csv(save_all_evals, index=False)

print('Plots and csv completed in {:.2f} sec\n'.format((time.time() - time_start)))
