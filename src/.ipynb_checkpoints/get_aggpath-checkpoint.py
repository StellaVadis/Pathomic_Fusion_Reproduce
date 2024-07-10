import pickle
import copy
import argparse

## This function is used to get the path of aggregation splits

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', help='experiment name')

args = parser.parse_args()


if args.exp_name == 'surv_15': 
    file_path = '../data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_patches_512_1_0_1_rnaseq.pkl'
    dumped_path = '../data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_patches_512_1_0_1_rnaseq_agg.pkl'
    
elif args.exp_name == 'grad_15': 
    file_path = '../data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_patches_512_1_1_1.pkl'
    dumped_path = '../data/TCGA_GBMLGG/splits/gbmlgg15cv_all_st_patches_512_1_1_1_agg.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)
    
# Create the files for each patient
new_data = copy.deepcopy(data)

for i in range(1, 16):
    for key in new_data['cv_splits'][i]['train'].keys():
        new_data['cv_splits'][i]['test'][key] = []
        new_data['cv_splits'][i]['train'][key] = []
    
    
for split in range(1,16,1):
    for train_test in ['train','test']:
    
        dict = data['cv_splits'][split][train_test]

        new_dict = new_data['cv_splits'][split][train_test]

        temp_dict = {}


        for i in range(len(dict['x_patname'])):
            name = dict['x_patname'][i]
            if name not in temp_dict:
                temp_dict[name] = {
                    'x_path': [],
                    'x_grph': [],
                    'x_omic': dict['x_omic'][i],
                    'e': [],
                    't': [],
                    'g': []
                }
                
                
            
            temp_dict[name]['x_path'].append(dict['x_path'][i].reshape(1, -1))
            # temp_dict[name]['x_grph'].append(dict['x_grph'][i])
            #temp_dict[name]['x_omic'].append(dict['x_omic'][i])
            temp_dict[name]['e'].append(dict['e'][i])
            temp_dict[name]['t'].append(dict['t'][i])
            temp_dict[name]['g'].append(dict['g'][i])
            
        for name, attributes in temp_dict.items():
            new_dict['x_patname'].append(name)
            new_dict['x_path'].append(attributes['x_path'])
            new_dict['x_grph'].append(attributes['x_grph'])
            new_dict['x_omic'].append(attributes['x_omic'])
            new_dict['e'].append(attributes['e'][0])
            new_dict['t'].append(attributes['t'][0])
            new_dict['g'].append(attributes['g'][0])


with open(dumped_path, 'wb') as f:
    pickle.dump(new_data, f)
    



