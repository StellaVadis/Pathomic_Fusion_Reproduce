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


# Env
from data_loaders import *
from options import parse_args
from train_test import train, test
from data_loaders import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader
from utils import unfreeze_unimodal, CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, mixed_collate, count_parameters,dfs_freeze
from networks import MaxNet, define_act_layer, define_bifusion
from torch.nn import init, Parameter

### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')


ignore_missing_histype = 1 if 'grad' in opt.task else 0
ignore_missing_moltype = 1 if 'omic' in opt.mode else 0
use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_patches else ('_', 'all_st')
use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''


data_cv_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)

data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
data_pd = data_cv['data_pd']


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
        print("Init Type:", init_type)
        init_weights(net, init_type, init_gain=init_gain)
    elif init_type == 'none':
        print("Init Type: Not initializing networks.")
    elif init_type == 'max':
        print("Init Type: Self-Normalizing Weights")
    return net

class PathomicNet(nn.Module):
    def __init__(self, opt, act, k):
        super(PathomicNet, self).__init__()
        self.omic_net = MaxNet(input_dim=opt.input_size_omic, omic_dim=opt.omic_dim, dropout_rate=opt.dropout_rate, act=act, label_dim=opt.label_dim, init_max=False)

        if k is not None:
            pt_fname = '_%d.pt' % k
            best_omic_ckpt = torch.load(os.path.join(opt.checkpoints_dir, opt.exp_name, 'omic', 'omic'+pt_fname), map_location=torch.device('cpu'))
            self.omic_net.load_state_dict(best_omic_ckpt['model_state_dict'])
            print("Loading Models:\n", os.path.join(opt.checkpoints_dir, opt.exp_name, 'omic', 'omic'+pt_fname))

        self.fusion = define_bifusion(fusion_type=opt.fusion_type, skip=opt.skip, use_bilinear=opt.use_bilinear, gate1=opt.path_gate, gate2=opt.omic_gate, dim1=opt.path_dim, dim2=opt.omic_dim, scale_dim1=opt.path_scale, scale_dim2=opt.omic_scale, mmhid=opt.mmhid, dropout_rate=opt.dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act

        dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec = kwargs['x_path']

        omic_vec, _ = self.omic_net(x_omic=kwargs['x_omic'])
        

        features = self.fusion(path_vec, omic_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

    def __hasattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return True
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return True
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return True
        return False

     
class NewModel(nn.Module):
    def __init__(self, base_model):
        super(NewModel, self).__init__()
        # Assuming base_model is an instance of an existing model
        self.base_model = base_model

    def forward(self, x):
        # Forward pass through the base_model
        
        _, pred = self.base_model(x_omic=x[:,0:320], x_path = x[:,320:])

        return pred
    

    
# Assuming model is defined somewhere
def main(glioma_type):
    total_attributions = None

    for k in range(1,16,1):
        data = data_cv_splits[k]
        model = define_net(opt, k)  # Define your original model

        custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, split='train', mode=opt.mode) if opt.use_vgg_features else PathgraphomicDatasetLoader(opt, data, split='train', mode=opt.mode)
        train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=100000, shuffle=False, collate_fn=mixed_collate)


        # Specify the path to the checkpoint file
        checkpoint_path = f'../checkpoints/TCGA_GBMLGG/surv_15_rnaseq/pathomic_fusion/pathomic_fusion_{k}.pt'

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


        model.load_state_dict(checkpoint['model_state_dict'])



        train_iter = iter(train_loader)


        sample_batch = next(train_iter)


        inputs = torch.hstack([sample_batch[3],sample_batch[1]]).to(device)


        new_model = NewModel(model).to(device)


        ig = IntegratedGradients(new_model)

        new_model.eval()

        attributions, delta = ig.attribute(inputs[data_pd['Histomolecular subtype'][data_cv_splits[k]['train']['x_patname']] == glioma_type], target=0, return_convergence_delta=True)


        if total_attributions is None:
            total_attributions = attributions
        else:
            total_attributions = torch.vstack([total_attributions, attributions])



    column_names = np.loadtxt('../data/TCGA_GBMLGG/column_names_GBMLGG.txt', dtype=str)


    sorted_indices = np.argsort(torch.abs((total_attributions).mean(axis = 0)[0:320]).cpu())

    sorted_column_names = column_names[sorted_indices] 


    top_column_names = column_names[sorted_indices]  
    top_mean_attributions = (total_attributions).mean(axis = 0)[sorted_indices].cpu()
    top_attributions = total_attributions[:,sorted_indices].cpu()
    

    file_path = f'../Interpretation/TCGA_GBMLGG/IG/IG_pathomic_{glioma_type}.pkl'

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
    plt.xlim([-1.05,1.05])
    plt.xlabel(f'IG Contribution')
    plt.title(f'IG Contributions for Top Features {glioma_type} (Pathomic)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'../Interpretation/TCGA_GBMLGG/IG/IG_pathomic_{glioma_type}.png')
    
main('ODG')
main('idhwt_ATC')
main('idhmut_ATC')