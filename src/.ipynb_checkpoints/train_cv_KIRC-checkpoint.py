import os
import logging
import numpy as np
import random
import pickle

import torch

# Env
from data_loaders import *
from options import parse_args
from train_test import train, test


### 1. Initializes parser and device
opt = parse_args()
print(opt.use_vgg_features)
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


if opt.mode == 'path':
    data_cv_path = '%s/splits/KIRC_st_0.pkl' % (opt.dataroot)
else:
    data_cv_path = '%s/splits/KIRC_st_1.pkl' % (opt.dataroot)




print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['split']


results = []

### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
    if k in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
        print("*******************************************")
        print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
        print("*******************************************")

        ### 3.1 Trains Model
        # print(data)
        
        '''
        signal = 1 
        while signal == 1:
            try:    
                model, optimizer, metric_logger = train(opt, data, device, k)
                signal = 0
            except:
                print('training failure, redo it again !')
        '''
        
        
        model, optimizer, metric_logger = train(opt, data, device, k)


        ### 3.2 Evalutes Train + Test Error, and Saves Model
        loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train, _ = test(opt, model, data, 'train', device)
        loss_test,  cindex_test,  pvalue_test,  surv_acc_test,  grad_acc_test,  pred_test,  _ = test(opt, model, data, 'test',  device)

        if opt.task == 'surv':
            print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
            logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
            print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
            logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
            results.append(cindex_test)
        elif opt.task == 'grad':
            print("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
            logging.info("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
            print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
            logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
            results.append(grad_acc_test)

        ### 3.3 Saves Model
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            model_state_dict = model.module.cpu().state_dict()
        else:
            model_state_dict = model.cpu().state_dict()

        torch.save({
            'split':k,
            'opt': opt,
            'epoch': opt.niter+opt.niter_decay,
            'data': data,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metric_logger}, 
            os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d.pt' % (opt.model_name, k)))

        print()

        # pickle.dump(pred_train, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_train.pkl' % (opt.model_name, k, use_patch)), 'wb'))
        pickle.dump(pred_test, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)), 'wb'))
        print('The test metrics has been saved to:', os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_%d%spred_test.pkl' % (opt.model_name, k, use_patch)))


print('Split Results:', results)
print("Average:", np.array(results).mean())
# pickle.dump(results, open(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))