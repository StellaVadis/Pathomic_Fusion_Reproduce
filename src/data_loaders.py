### data_loaders.py
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets, transforms
import random


################
# Dataset Loader
################
class PathgraphomicDatasetLoader(Dataset):
    def __init__(self, opt, data, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.X_patname = data[split]['x_patname']
        self.X_path = data[split]['x_path']
        self.X_grph = data[split]['x_grph']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        self.mode = mode
        
        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(opt.input_size_path),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        
        ## Naming Convention
        if self.X_grph[index].startswith('.') and not self.X_grph[index].startswith('..'):
            self.X_grph[index] = '..' + self.X_grph[index][1:]
            
            

            
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

        if self.mode == "path" or self.mode == 'pathpath':          
            if self.X_path[index].startswith('.') and not self.X_path[index].startswith('..'):
                path_fname = '..'+self.X_path[index][1:]
            else:
                path_fname = self.X_path[index]
            single_X_path = Image.open(path_fname).convert('RGB')
            return (self.X_patname[index], self.transforms(single_X_path), 0, 0, single_e, single_t, single_g)
        elif self.mode == "graph" or self.mode == 'graphgraph':
            single_X_grph = torch.load(self.X_grph[index])
            return (self.X_patname[index], 0, single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "omic" or self.mode == 'omicomic':
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.X_patname[index], 0, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathomic":
            if self.X_path[index].startswith('.') and not self.X_path[index].startswith('..'):
                path_fname = '..'+self.X_path[index][1:]
            else:
                path_fname = self.X_path[index]
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.X_patname[index], self.transforms(single_X_path), 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "graphomic":
            single_X_grph = torch.load(self.X_grph[index])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.X_patname[index], 0, single_X_grph, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathgraph":
            if self.X_path[index].startswith('.') and not self.X_path[index].startswith('..'):
                path_fname = '..'+self.X_path[index][1:]
            else:
                path_fname = self.X_path[index]
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            single_X_grph = torch.load(self.X_grph[index])
            return (self.X_patname[index], self.transforms(single_X_path), single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "pathgraphomic":
            if self.X_path[index].startswith('.') and not self.X_path[index].startswith('..'):
                path_fname = '..'+self.X_path[index][1:]
            else:
                path_fname = self.X_path[index]
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            single_X_grph = torch.load(self.X_grph[index])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.X_patname[index], self.transforms(single_X_path), single_X_grph, single_X_omic, single_e, single_t, single_g)

    def __len__(self):
        return len(self.X_path)


class PathgraphomicFastDatasetLoader(Dataset):
    def __init__(self, opt, data, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.X_patname = data[split]['x_patname']
        self.X_path = data[split]['x_path']
        self.X_grph = data[split]['x_grph']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        self.mode = mode
        self.split = split


    def __getitem__(self, index):
        
        
        ## Naming Convention
            
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

        if self.mode == "path" or self.mode == 'pathpath':
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            return (self.X_patname[index], single_X_path, 0, 0, single_e, single_t, single_g)
        elif self.mode == "graph" or self.mode == 'graphgraph':
            if self.X_grph[index].startswith('.') and not self.X_grph[index].startswith('..'):
                self.X_grph[index] = '..' + self.X_grph[index][1:]
            single_X_grph = torch.load(self.X_grph[index])
            return (self.X_patname[index], 0, single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "omic" or self.mode == 'omicomic':
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.X_patname[index], 0, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathomic":
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.X_patname[index], single_X_path, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "graphomic":
            if self.X_grph[index].startswith('.') and not self.X_grph[index].startswith('..'):
                self.X_grph[index] = '..' + self.X_grph[index][1:]
            single_X_grph = torch.load(self.X_grph[index])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.X_patname[index], 0, single_X_grph, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathgraph":
            if self.X_grph[index].startswith('.') and not self.X_grph[index].startswith('..'):
                self.X_grph[index] = '..' + self.X_grph[index][1:]
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            single_X_grph = torch.load(self.X_grph[index])
            return (self.X_patname[index], single_X_path, single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "pathgraphomic":
            if self.X_grph[index].startswith('.') and not self.X_grph[index].startswith('..'):
                self.X_grph[index] = '..' + self.X_grph[index][1:]
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            single_X_grph = torch.load(self.X_grph[index])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.X_patname[index], single_X_path, single_X_grph, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "agg_pathomic":
            if self.split == 'train':
                single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
                sample_list = self.X_path[index]
                # Randomly sample 5 elements from the list
                if len(sample_list) >= 5:
                    sampled_elements = random.sample(sample_list, 5)
                else:
                    # If the list has less than 4 elements, sample with replacement
                    sampled_elements = random.choices(sample_list, k=5)
                sampled_elements = [torch.tensor(elem) for elem in sampled_elements]
                single_X_path = torch.stack(sampled_elements).squeeze(1)
                return (self.X_patname[index], single_X_path, 0, single_X_omic, single_e, single_t, single_g)
            elif self.split == 'test':
                single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
                sample_list = self.X_path[index]
                sampled_elements = [torch.tensor(elem) for elem in sample_list]
                single_X_path = torch.stack(sampled_elements).squeeze(1)
                return (self.X_patname[index], single_X_path, 0, single_X_omic, single_e, single_t, single_g)
            
        elif self.mode == "agg_graphomic":
            if self.split == 'train':
                single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
                sample_list = self.X_grph[index]
                if len(sample_list) >= 5:
                    sampled_elements = random.sample(sample_list, 5)
                else:
                    sampled_elements = random.choices(sample_list, k=5)
                sampled_elements = [torch.tensor(elem) for elem in sampled_elements]
                single_X_grph = torch.stack(sampled_elements).squeeze(1)
                return (self.X_patname[index], 0, single_X_grph, single_X_omic, single_e, single_t, single_g)
            elif self.split == 'test':
                single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
                sample_list = self.X_grph[index]
                sampled_elements = [torch.tensor(elem) for elem in sample_list]
                single_X_grph = torch.stack(sampled_elements).squeeze(1)
                return (self.X_patname[index], 0, single_X_grph, single_X_omic, single_e, single_t, single_g)
            
        elif self.mode == "agg_pathgraphomic":
            if self.split == 'train':
                single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
                
                sample_list_grph = self.X_grph[index]
                if len(sample_list_grph) >= 5:
                    sampled_elements_grph = random.sample(sample_list_grph, 5)
                else:
                    sampled_elements_grph = random.choices(sample_list_grph, k=5)
                sampled_elements_grph = [torch.tensor(elem) for elem in sampled_elements_grph]
                single_X_grph = torch.stack(sampled_elements_grph).squeeze(1)
 
                sample_list_path = self.X_path[index]
                if len(sample_list_path) >= 5:
                    sampled_elements_path = random.sample(sample_list_path, 5)
                else:
                    sampled_elements_path = random.choices(sample_list_path, k=5)
                sampled_elements_path = [torch.tensor(elem) for elem in sampled_elements_path]
                single_X_path  = torch.stack(sampled_elements_path).squeeze(1)

                return (self.X_patname[index], single_X_path, single_X_grph, single_X_omic, single_e, single_t, single_g)

            elif self.split == 'test':
                single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
                
                sample_list_grph = self.X_grph[index]
                sampled_elements_grph = [torch.tensor(elem) for elem in sample_list_grph]
                single_X_grph = torch.stack(sampled_elements_grph).squeeze(1)

                sample_list_path = self.X_path[index]
                sampled_elements_path = [torch.tensor(elem) for elem in sample_list_path]
                single_X_path = torch.stack(sampled_elements_path).squeeze(1)                   
                
                return (self.X_patname[index], single_X_path, single_X_grph, single_X_omic, single_e, single_t, single_g)
                               

    def __len__(self):
        return len(self.X_path)