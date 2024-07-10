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
# from torchcam.methods import GradCAM
from torchvision.transforms.functional import to_pil_image
import PIL

from utils import CoxLoss




# Env
from data_loaders import *
from options import parse_args
from train_test import train, test
from data_loaders import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader
from utils import unfreeze_unimodal, CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, mixed_collate, count_parameters
from networks import MaxNet, define_act_layer, get_vgg, define_reg



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
        # Assuming base_model is an instance of an existing model
        self.base_model = base_model

    def forward(self, x):
        # Forward pass through the base_model
        _, pred = self.base_model(x_path=x)
        #sig_logit = (pred + 3)/6
        #print(sig_logit)
        #logits = torch.zeros_like(sig_logit)
        #logits[:, 0] = torch.log(sig_logit[:, 0] / (1 - sig_logit[:, 0]))
        return pred
    

    
def custom_loss(pred , censor, survtime):
    loss_cox = CoxLoss(survtime, censor, pred, 'cpu') if opt.task == "surv" else 0
    print('loss_cox',loss_cox)
    loss_reg = define_reg(opt, model)
    loss = (loss_cox + opt.lambda_reg*loss_reg) 
    print(type(loss))
    return loss


'''
The code of GradCAM is provided b Chatgpt
'''
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input, target=None):
        self.model.zero_grad()
        pred = self.model(input)
        if target is None:
            target = pred

        loss = custom_loss(pred , censor, survtime)
        
        loss.backward()

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        weights = np.mean(gradients, axis=(2, 3))
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        # Only consider sample 0, discarding any other one
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        # cam = np.uint8(cam * 255)
        return cam
    
    
    
total_attributions = None


## 4,6,7,9
for k in [4,6,7,9]:
    
    data = data_cv_splits[k]
    model = define_net(opt, k)  



    custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, split='train', mode=opt.mode) if opt.use_vgg_features else PathgraphomicDatasetLoader(opt, data, split='train', mode=opt.mode)
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=70, shuffle=False, collate_fn=mixed_collate) ## 70


    checkpoint_path = f'../checkpoints/TCGA_GBMLGG/surv_15_rnaseq/path/path_{k}.pt'



    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


    model.load_state_dict(checkpoint['model_state_dict'])


    new_model = NewModel(model)


    train_iter = iter(train_loader)

    sample_batch = next(train_iter)

    inputs = sample_batch[1]

    censor = sample_batch[4]

    survtime = sample_batch[5]


    
    def disable_dropout(model):

        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0


    new_model.eval()  
    disable_dropout(new_model)


    for param in new_model.parameters():
        param.requires_grad = True


    target_layer = new_model.base_model.features[14]  

    cam = GradCAM(new_model, target_layer)
    heatmap = cam(inputs)

    overlay = np.array(to_pil_image(heatmap, mode='F').resize((512, 512), resample=PIL.Image.BICUBIC))

    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min()) 

    inputs = (inputs - inputs.min())/(inputs.max() - inputs.min())
    
    inputs_np = inputs[0].numpy().transpose((1, 2, 0))


    data_to_save = {
        'original': inputs_np,
        'mask': overlay
    }
    

    with open(f'../Interpretation/TCGA_GBMLGG/Gradcam/Gradcam_GBMLGG_{k}.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    
        
    
    # Load and Plot
    with open(f'../Interpretation/TCGA_GBMLGG/Gradcam/Gradcam_GBMLGG_{k}.pkl', 'rb') as file:
        data = pickle.load(file)
             
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(data['original'])
    axes[0].set_title('Original')

    axes[1].imshow(data['mask'], cmap='jet')
    axes[1].set_title('Gradient Heatmap')

    axes[2].imshow(data['original'])
    axes[2].imshow(data['mask'], cmap='jet', alpha=0.5, vmin=np.min(data['mask']), vmax=np.max(data['mask'])-0.5)
    axes[2].set_title('Overlay')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'../Interpretation/TCGA_GBMLGG/Gradcam/Gradcam_GBMLGG_{k}.png')
