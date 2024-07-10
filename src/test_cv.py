import os
import logging
import numpy as np
import random
import pickle

import torch
import copy

# Env
from networks import define_net
from data_loaders import *
from options import parse_args
from train_test import train, test


### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

### 2. Initializes Data
ignore_missing_histype = 1 if 'grad' in opt.task else 0
ignore_missing_moltype = 1 if 'omic' in opt.mode else 0
use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_patches else ('_', 'all_st')
use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''

data_cv_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results = []


# This is used to extract the GCN features
if opt.task == 'surv':
    data_cv_new = pickle.load(open('../data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_patches_512_1_0_1_rnaseq_agg.pkl', 'rb'))
elif opt.task == 'grad':
    data_cv_new = pickle.load(open('../data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_patches_512_1_1_1_agg.pkl', 'rb'))


data_cv_splits_new = data_cv_new['cv_splits']
   

### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
    print("*******************************************")
    print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
    print("*******************************************")
    load_path = os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k))
    model_ckpt = torch.load(load_path, map_location=device)

    #### Loading Env
    model_state_dict = model_ckpt['model_state_dict']
    if hasattr(model_state_dict, '_metadata'): del model_state_dict._metadata

    model = define_net(opt, None)
    if isinstance(model, torch.nn.DataParallel): model = model.module

    print('Loading the model from %s' % load_path)
    model.load_state_dict(model_state_dict)


    ### 3.2 Evalutes Train + Test Error, and Saves Model
    
    # all modes will inccur

    
    if opt.mode == 'graph':
        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, name_dict = test(opt, model, data, 'train', device)    
        
        new_ls = []
        for i in range(0,len(data_cv_splits_new[k]['train']['x_patname']),1):
            new_ls.append(name_dict[data_cv_splits_new[k]['train']['x_patname'][i]])

        data_cv_splits_new[k]['train']['x_grph'] = copy.deepcopy(new_ls)


    loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, name_dict = test(opt, model, data, 'test', device)
    
    if opt.mode == 'graph':
        new_ls = []
        for i in range(0,len(data_cv_splits_new[k]['test']['x_patname']),1):
            new_ls.append(name_dict[data_cv_splits_new[k]['test']['x_patname'][i]])

        data_cv_splits_new[k]['test']['x_grph'] = copy.deepcopy(new_ls)


    
    
    
    if opt.task == 'surv':
        print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
        logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
        results.append(cindex_test)
    elif opt.task == 'grad':
        print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
        logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
        results.append(grad_acc_test)

    ### 3.3 Saves Model
    pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))
    print('The prediction result has been saved to:',os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)))    


print('Split Results:', results)
print("Average:", np.array(results).mean())
# pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))

if opt.mode == 'graph':
    with open('../data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_patches_512_1_0_1_rnaseq_agg.pkl', 'wb') as file:
        pickle.dump(data_cv_new, file)
