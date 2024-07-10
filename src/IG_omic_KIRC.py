import torch
from captum.attr import IntegratedGradients

from networks import define_net

import os
import logging
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import torch
from tqdm import tqdm
import seaborn as sns


# Env
from data_loaders import *
from options import parse_args
from train_test import train, test
from data_loaders import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader
from utils import unfreeze_unimodal, CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, mixed_collate, count_parameters
from networks import MaxNet, define_act_layer

### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')


ignore_missing_histype = 1 if 'grad' in opt.task else 0
ignore_missing_moltype = 1 if 'omic' in opt.mode else 0
use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_patches else ('_', 'all_st')
use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''


data_cv_path = '%s/splits/KIRC_st_1.pkl' % (opt.dataroot)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['split']
data_pd = data_cv['all_dataset']


def define_net(opt, k):
    net = None
    act = define_act_layer(act_type=opt.act_type)
    init_max = True if opt.init_type == "max" else False

    if opt.mode == "path":
        net = get_vgg(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim)
    elif opt.mode == "graph":
        net = GraphNet(grph_dim=opt.grph_dim, dropout_rate=opt.dropout_rate, GNN=opt.GNN, use_edges=opt.use_edges, pooling_ratio=opt.pooling_ratio, act=act, label_dim=opt.label_dim, init_max=init_max)
    elif opt.mode == "omic":
        net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=init_max)
    elif opt.mode == "graphomic":
        net = GraphomicNet(opt=opt, act=act, k=k)
    elif opt.mode == "pathomic":
        net = PathomicNet(opt=opt, act=act, k=k)
    elif opt.mode == "pathgraphomic":
        net = PathgraphomicNet(opt=opt, act=act, k=k)
    elif opt.mode == "pathpath":
        net = PathpathNet(opt=opt, act=act, k=k)
    elif opt.mode == "graphgraph":
        net = GraphgraphNet(opt=opt, act=act, k=k)
    elif opt.mode == "omicomic":
        net = OmicomicNet(opt=opt, act=act, k=k)
    else:
        raise NotImplementedError('model [%s] is not implemented' % opt.model)
    return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
    
    
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """

    if init_type != 'max' and init_type != 'none':
        #print("Init Type:", init_type)
        init_weights(net, init_type, init_gain=init_gain)
    elif init_type == 'none':
        pass
        #print("Init Type: Not initializing networks.")
    elif init_type == 'max':
        pass
        #print("Init Type: Self-Normalizing Weights")
    return net



    
class NewModel(nn.Module):
    def __init__(self, base_model):
        super(NewModel, self).__init__()
        self.base_model = base_model

    def forward(self, x):

        _, pred = self.base_model(x_omic=x)

        return pred
    



def main(survive_type):
    total_attributions = None

    for k in range(1,16,1):
        data = data_cv_splits[k]
        model = define_net(opt, k) 

        custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, split='train', mode=opt.mode) if opt.use_vgg_features else PathgraphomicDatasetLoader(opt, data, split='train', mode=opt.mode)
        train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=100000, shuffle=False, collate_fn=mixed_collate)


        checkpoint_path = f'../checkpoints/TCGA_KIRC/surv_15/omic/omic_{k}.pt'

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


        model.load_state_dict(checkpoint['model_state_dict'])


        new_model = NewModel(model)

        train_iter = iter(train_loader)

        sample_batch = next(train_iter)

        inputs = sample_batch[3]
        
        name = sample_batch[0]


        ig = IntegratedGradients(new_model)

        new_model.eval()
        
        
        if survive_type == 'overall':
            attributions, delta = ig.attribute(inputs, target=0, return_convergence_delta=True)
        elif survive_type == 'short':
            indices = []
            for ii in tqdm(range(0,len(name),1)):
                if data_pd.loc[name[ii][0:12],'OS_month'] <= 12*3.5:
                    indices.append(True)
                else:
                    indices.append(False)
            attributions, delta = ig.attribute(inputs[indices], target=0, return_convergence_delta=True)
        elif survive_type == 'long':
            indices = []
            for ii in tqdm(range(0,len(name),1)):
                if data_pd.loc[name[ii][0:12],'OS_month'] > 12*3.5:
                    indices.append(True)
                else:
                    indices.append(False)
            attributions, delta = ig.attribute(inputs[indices], target=0, return_convergence_delta=True)


        if total_attributions is None:
            total_attributions = attributions
        else:
            total_attributions = np.vstack([total_attributions, attributions])



    column_names = np.loadtxt('../data/TCGA_KIRC/column_names_KIRC.txt', dtype=str)


    sorted_indices = np.argsort(np.abs((total_attributions).mean(axis = 0)))


    top_column_names = column_names[sorted_indices]  
    top_mean_attributions = (total_attributions).mean(axis = 0)[sorted_indices]
    top_attributions = total_attributions[:,sorted_indices]
    

    file_path = f'../Interpretation/TCGA_KIRC/IG/IG_omic_{survive_type}.pkl'

    data_to_save = {
        'top_column_names': top_column_names,
        #'top_attributions': top_attributions,
        'top_mean_attributions': top_mean_attributions
    }


    with open(file_path, 'wb') as f:
        pickle.dump(data_to_save, f)
        
    top_mean_attributions = (top_mean_attributions)/ np.abs(top_mean_attributions).max() 

    
    cmap = mcolors.LinearSegmentedColormap.from_list("RedCournflowerblue", ["cornflowerblue","red"])
    
    norm = plt.Normalize(-0.2,0.2)
    norm_values = norm(top_mean_attributions[-20:])
    
    colors = cmap(norm_values)
    
    plt.figure(figsize=(12, 8))
    ax = plt.barh(top_column_names[-20:], top_mean_attributions[-20:], color=colors,edgecolor='grey')
    
    sm = cm.ScalarMappable(cmap=cmap,norm=norm) 
    sm.set_array([])
    plt.colorbar(sm)
    
    plt.xlabel(f'IG Contribution')
    plt.xlim([-1.05,1.05])
    plt.title(f'IG Contributions for Top Features {survive_type} (Omic)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'../Interpretation/TCGA_KIRC/IG/IG_omic_{survive_type}.png')
    
    
main('short')
main('long')
main('overall')