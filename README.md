# Data Analysis Project: Pathomic Fusion
Original Paper: https://arxiv.org/pdf/1912.08937

# 1. Data Acquisition

## Clone the Repository
```bash
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/projects/dz357.git
cd dz357
```

## Download the Dataset and Checkpoints from Google Drive
https://drive.google.com/drive/folders/1yQm6WjsRbr_ghwQdxyCvAIOaOAViTL9S?usp=drive_link

## Move the Downloaded Dataset and Checkpoints to Appropriate space
```bash
PathomicFusion/
├── data/
│   ├── TCGA_GBMLGG/
│   └── TCGA_KIRC/
├── checkpoints/
│   ├── TCGA_GBMLGG/
│   └── TCGA_KIRC/
├── src/
├── Interpretation/
│   ├── TCGA_GBMLGG/
│   └── TCGA_KIRC/
├── report/
│   ├── report.pdf
│   └── summary.pdf
├── core/
├── Evaluation-GBMLGG.ipynb
└── Evaluation-KIRC.ipynb
```

data: Contain the datasets and splits
checkpoints: Contain the well-trained models, training and testing results

src: Contain the Codes

Interpretation: Contain the Histogram, Swarm Plot, Kaplan-Meier Plots, Integrated Gradients and Grad-CAM

core: Contain the sample aggregation codes, for Evaluation-GBMLGG.ipynb and Evaluation-KIRC.ipynb

Evaluation-GBMLGG.ipynb and Evaluation-KIRC.ipynb: Jupyter notebooks to aggregate the predictions of different samples from the same patient

## Install the Environment
```bash
pip install -r requirements.txt
```

## Move to Code Directory
```bash
cd src
```

# 2. Task 1: Pipeline for TCGA-GBMLGG (Survival Prediction)

## (Optional) Make Splits for SNN, SNN + SNN
```
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st --use_rnaseq 1
```

## SNN (Train + Test)
Although in SNN --use_vgg_features 0 --use_patches 0 is unrelated, but we still include them such that the splits .pkl files can be found.
```bash
python train_cv_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode omic --model_name omic --niter 0 --niter_decay 50 --batch_size 64 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --use_vgg_features 0 --use_patches 0  --gpu_ids 0 --use_rnaseq 1 --input_size_omic 320 --verbose 1
```

## SNN + SNN (Train + Test)
```bash
python train_cv_GBMLGG.py  --exp_name surv_15_rnaseq --task surv --mode omicomic --model_name omicomic_fusion --niter 2 --niter_decay 4 --batch_size 64 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --use_vgg_features 0 --use_patches 0 --gpu_ids 0 --use_rnaseq 1 --input_size_omic 320 --verbose 1
```

##  (Optional)  Make Splits for CNN (Train + Test)
```bash
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st
```

## CNN (Train)
Train the model on the full image, where the full image during the training is randomly cropped into (512,512) patches

```
python  train_cv_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --use_vgg_features 0 --use_patches 0 --gpu_ids 0
```

##  (Optional)  Make Splits for CNN (Test)
```bash
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st_patches_512
```

## CNN (Test)
t=Test the model on a stable dataset, where you can find the descriptions of all_st_512 in the author's page.

```bash
python test_cv.py  --exp_name surv_15_rnaseq --task surv --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --use_vgg_features 0 --use_patches 1 --gpu_ids 0
```

##  (Optional) Make Splits for GCN, GCN + GCN, CNN + CNN (Train + Test)
```bash
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --gpu_ids 0
```

## GCN (Train + Test)
```bash
python train_cv_GBMLGG.py  --exp_name surv_15_rnaseq --task surv --mode graph --model_name graph --niter 0 --niter_decay 50 --lr 0.002 --init_type max --reg_type none --lambda_reg 0 --use_vgg_features 1 --use_patches 1 --gpu_ids 0
```

## GCN + GCN (Train + Test)
```bash
python train_cv_GBMLGG.py  --exp_name surv_15_rnaseq --task surv --mode graphgraph --model_name graphgraph_fusion --niter 2 --niter_decay 4 --lr 0.002 --init_type max --reg_type none --lambda_reg 0 --use_vgg_features 1 --use_patches 1 --gpu_ids 0
```

## CNN + CNN (Train + Test)
```bash
python train_cv_GBMLGG.py  --exp_name surv_15_rnaseq --task surv --mode pathpath --model_name pathpath_fusion --niter 2 --niter_decay 4 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --use_vgg_features 1 --use_patches 1 --gpu_ids 0 
```

##  (Optional)  Make Splits for the FUSION model
```bash
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 0 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_rnaseq --gpu_ids 0 --use_rnaseq 1
```

## CNN + SNN (Train + Test)
```bash
python train_cv_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --use_patches 1 --gpu_ids 0 --omic_gate 0 --use_rnaseq 1 --input_size_omic 320
```

## GCN + SNN (Train + Test)
```bash
python train_cv_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode graphomic --model_name graphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --use_patches 1 --gpu_ids 0 --omic_gate 0 --grph_scale 2 --use_rnaseq 1 --input_size_omic 320
```

## CNN + GCN + SNN (Train + Test)
```bash
python train_cv_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode pathgraphomic --model_name pathgraphomic_fusion --niter 2 --niter_decay 2 --lr 0.0001 --beta1 0.5 --fusion_type pofusion_A --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --grph_scale 2 --use_rnaseq 1 --input_size_omic 320
```

We discover that running too many epochs can overfit the models.

## Get the Aggregated Path Data
```bash
python get_aggpath.py --exp_name surv_15
```

## Append the Aggregated Graph Data into the Path Dataset
I have embbed the code of appending the aggregated graph data into the test_cv.py
```bash
python test_cv.py  --exp_name surv_15_rnaseq --task surv --mode graph --model_name graph --niter 0 --niter_decay 50 --lr 0.002 --init_type max --reg_type none --lambda_reg 0 --use_vgg_features 1 --use_patches 1 --gpu_ids 0
```

## CNN + SNN hazard attention aggregate (Train + Test)
```bash
python train_cv_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode agg_pathomic --model_name agg_pathomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --use_rnaseq 1 --input_size_omic 320
```

## GCN + SNN hazard attention aggregate (Train + Test)
```
python train_cv_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode agg_graphomic --model_name agg_graphomic_fusion --niter 50 --niter_decay 0 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --use_rnaseq 1 --input_size_omic 320
```

## CNN + GCN + SNN hazard attention aggregate (Train + Test)
```
python train_cv_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode agg_pathgraphomic --model_name agg_pathgraphomic_fusion --niter 80 --niter_decay 0 --lr 0.0001 --beta1 0.5 --fusion_type pofusion_A --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --gpu_ids 0 --omic_gate 0 --grph_scale 2 --use_rnaseq 1 --input_size_omic 320 --batch_size 128 --lambda_cox 0.05
```

# 3. Task 2: Pipeline for TCGA-GBMLGG (Grade Prediction)

## (Optional) Make Splits for SNN, SNN + SNN 
```
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st
```

## SNN (Train + Test)
```bash
python train_cv_GBMLGG.py --exp_name grad_15 --task grad --mode omic --model_name omic --niter 0 --niter_decay 50 --batch_size 64 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --act LSM --label_dim 3 --gpu_ids 0 --use_vgg_features 0 --use_patches 0
```

## SNN + SNN (Train + Test)
```bash
python train_cv_GBMLGG.py --exp_name grad_15 --task grad --mode omicomic --model_name omicomic_fusion --niter 2 --niter_decay 4 --batch_size 64 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --act LSM --label_dim 3 --gpu_ids 0 --use_vgg_features 0 --use_patches 0
```

##  (Optional)  Make Splits for CNN (Train)
```bash
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 0 --roi_dir all_st
```

## CNN (Train)
```bash
python train_cv_GBMLGG.py  --exp_name grad_15 --task grad --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --use_vgg_features 0 --use_patches 0 --lambda_reg 0 --act LSM --label_dim 3 --gpu_ids 0
```

##  (Optional)  Make Splits for CNN (Test)
```bash
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 0 --use_vgg_features 0 --roi_dir all_st_patches_512
```

## CNN (Test)
```bash
python test_cv.py  --exp_name grad_15 --task grad --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --use_vgg_features 0 --use_patches 1 --lambda_reg 0 --act LSM --label_dim 3 --gpu_ids 0 
```

##  (Optional) Make Splits for GCN, GCN + GCN, CNN + CNN (Train + Test)
```bash
python make_splits.py --ignore_missing_moltype 0 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --gpu_ids 0
```


## GCN (Train + Test)
```bash
python train_cv_GBMLGG.py  --exp_name grad_15 --task grad --mode graph --model_name graph --niter 0 --niter_decay 50 --lr 0.002 --init_type max --reg_type none --lambda_reg 0  --act LSM --label_dim 3 --gpu_ids 0 --use_vgg_features 1 --use_patches 1
```

## GCN + GCN (Train + Test)
```bash
python train_cv_GBMLGG.py  --exp_name grad_15 --task grad --mode graphgraph --model_name graphgraph_fusion --niter 2 --niter_decay 4 --lr 0.002 --init_type max --reg_type none --lambda_reg 0 --use_vgg_features 1 --use_patches 1 --act LSM --label_dim 3 --gpu_ids 0
```

## CNN + CNN (Train + Test)
```bash
python train_cv_GBMLGG.py  --exp_name grad_15 --task grad --mode pathpath --model_name pathpath_fusion --niter 2 --niter_decay 4 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --act LSM --label_dim 3 --gpu_ids 0 --use_vgg_features 1 --use_patches 1
```

##  (Optional)  Make Splits for the FUSION model 
```bash
python make_splits.py --ignore_missing_moltype 1 --ignore_missing_histype 1 --use_vgg_features 1 --roi_dir all_st_patches_512 --exp_name surv_15_rnaseq --gpu_ids 0 
```

## CNN + SNN (Train + Test)
```bash
python train_cv_GBMLGG.py --exp_name grad_15 --task grad --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --use_patches 1 --gpu_ids 0 --path_gate 0 --omic_scale 2 --act LSM --label_dim 3
```

## GCN + SNN (Train + Test)
```bash
python train_cv_GBMLGG.py --exp_name grad_15 --task grad --mode graphomic --model_name graphomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --use_patches 1 --gpu_ids 0 --grph_gate 0 --omic_scale 2 --act LSM --label_dim 3
```

## CNN + GCN + SNN (Train + Test)
```bash
python train_cv_GBMLGG.py --exp_name grad_15 --task grad --mode pathgraphomic --model_name pathgraphomic_fusion --niter 0 --niter_decay 4 --lr 0.0001 --beta1 0.5 --fusion_type pofusion_B --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --use_patches 1 --gpu_ids 0 --path_gate 0 --act LSM --label_dim 3
```

# 4. Task 3: Pipeline for TCGA-KIRC (Survival Prediction)
## SNN (Train + Test)
```bash
python train_cv_KIRC.py --act_type Sigmoid --batch_size 64 --beta1 0.9 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 1 --init_gain 0.02 --init_type max --input_size_omic 362 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0003 --lr 0.002 --lr_policy linear --measure 1 --mmhid 64 --mode omic --model_name omic --niter 0 --niter_decay 30 --omic_dim 32 --omic_gate 1 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 1 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type all --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 1 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1 --use_rnaseq 0 --use_vgg_features 0 --verbose 1 --weight_decay 0.0005 --use_patches 0
```

## SNN + SNN (Train + Test)
```bash
python train_cv_KIRC.py --GNN GCN --act_type Sigmoid --batch_size 32 --beta1 0.5 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 1 --init_gain 0.02 --init_type none --input_size_omic 362 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0003 --lr 0.0001 --lr_policy linear --measure 1 --mmhid 64 --mode omicomic --model_name omicomic_fusion --niter 10 --niter_decay 20 --omic_dim 32 --omic_gate 1 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 1 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type omic --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 1 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1 --use_rnaseq 0 --use_vgg_features 0 --verbose 1 --weight_decay 0.0004  --use_patches 0
```

## GCN (Train + Test)
```bash
python train_cv_KIRC.py --GNN GCN --act_type Sigmoid --batch_size 128 --beta1 0.9 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 1 --init_gain 0.02 --init_type max --input_size_omic 80 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0 --lr 0.002 --lr_policy linear --measure 1 --mmhid 64 --mode graph --model_name graph --niter 0 --niter_decay 30 --omic_dim 32 --omic_gate 1 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 1 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type none --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 4 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1.0 --use_rnaseq 0 --use_vgg_features 1 --verbose 1 --weight_decay 0.0004 --use_patches 0
```

## GCN + GCN (Train + Test)
```bash
python train_cv_KIRC.py --GNN GCN --act_type Sigmoid --batch_size 128 --beta1 0.9 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 1 --init_gain 0.02 --init_type max --input_size_omic 80 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0 --lr 0.002 --lr_policy linear --measure 1 --mmhid 64 --mode graphgraph --model_name graphgraph_fusion --niter 2 --niter_decay 4 --omic_dim 32 --omic_gate 1 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 1 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type none --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 4 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1.0 --use_rnaseq 0 --use_vgg_features 1 --verbose 1 --weight_decay 0.0004 --use_patches 0
```

## CNN (Train + Test)
```bash
python train_cv_KIRC.py --GNN GCN --act_type Sigmoid --batch_size 8 --beta1 0.9 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 1 --init_gain 0.02 --init_type none --input_size_omic 80 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0 --lr 0.0005 --lr_policy linear --measure 1 --mmhid 64 --mode path --model_name path --niter 0 --niter_decay 30 --omic_dim 32 --omic_gate 1 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 1 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type none --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 13 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1 --use_rnaseq 0 --use_vgg_features 0 --verbose 1 --weight_decay 0.0004 –-use_patches 0
```

## CNN + CNN (Train + Test)
```bash
python train_cv_KIRC.py --GNN GCN --act_type Sigmoid --batch_size 32 --beta1 0.5 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 1 --init_gain 0.02 --init_type none --input_size_omic 80 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0003 --lr 0.0001 --lr_policy linear --measure 1 --mmhid 64 --mode pathpath --model_name pathpath_fusion --niter 10 --niter_decay 20 --omic_dim 32 --omic_gate 1 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 1 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type none --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 1 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1 --use_rnaseq 0 --use_vgg_features 1 --verbose 1 --weight_decay 0.0004  --use_patches 0
```

## CNN + SNN (Train + Test)
```bash
python train_cv_KIRC.py --GNN GCN --act_type Sigmoid --batch_size 32 --beta1 0.5 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 1 --init_gain 0.02 --init_type none --input_size_omic 362 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0003 --lr 0.0001 --lr_policy linear --measure 1 --mmhid 64 --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --omic_dim 32 --omic_gate 0 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 0 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type omic --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 1 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1 --use_rnaseq 1 --use_vgg_features 1 --verbose 1 --weight_decay 0.0004  --use_patches 0
```

## GCN + SNN (Train + Test)
```bash
python train_cv_KIRC.py --GNN GCN --act_type Sigmoid --batch_size 32 --beta1 0.5 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 2 --init_gain 0.02 --init_type none --input_size_omic 362 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0003 --lr 0.0001 --lr_policy linear --measure 1 --mmhid 64 --mode graphomic --model_name graphomic_fusion --niter 10 --niter_decay 20 --omic_dim 32 --omic_gate 0 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 1 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type omic --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 1 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1 --use_rnaseq 1 --use_vgg_features 1 --verbose 1 --weight_decay 0.0004  --use_patches 0
```

## SNN + GCN + CNN (Train + Test)
```bash
python train_cv_KIRC.py --GNN GCN --act_type Sigmoid --batch_size 32 --beta1 0.5 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion_A --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 2 --init_gain 0.02 --init_type none --input_size_omic 362 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0003 --lr 0.0001 --lr_policy linear --measure 1 --mmhid 64 --mode pathgraphomic --model_name pathgraphomic_fusion --niter 2 --niter_decay 4 --omic_dim 32 --omic_gate 0 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 1 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type omic --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 1 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1 --use_rnaseq 1 --use_vgg_features 1 --verbose 1 --weight_decay 0.0004  --use_patches 0
```

# 5. Interpretations
## 5.1 Plots
Run the Jupyter Notebook Evaluation-GBMLGG.ipynb and Evaluation-KIRC.ipynb to obtain the Histogram, Kaplan-Meier plots, and Swarm plots.

## 5.2 Integrated Gradients (IG)
### SNN GBMLGG
```bash
python IG_omic_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode omic --model_name omic --niter 0 --niter_decay 50 --batch_size 64 --reg_type all --init_type max --lr 0.002 --weight_decay 5e-4 --use_vgg_features 0 --use_patches 0  --gpu_ids 0 --use_rnaseq 1 --input_size_omic 320 --verbose 1
```

### SNN + CNN GBMLGG
```bash
python IG_pathomic_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --lr 0.0001 --beta1 0.5 --fusion_type pofusion --mmhid 64 --use_bilinear 1 --use_vgg_features 1 --use_patches 1 --gpu_ids 0 --omic_gate 0 --use_rnaseq 1 --input_size_omic 320
```

### SNN KIRC
```bash
python IG_omic_KIRC.py --act_type Sigmoid --batch_size 64 --beta1 0.9 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 1 --init_gain 0.02 --init_type max --input_size_omic 362 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0003 --lr 0.002 --lr_policy linear --measure 1 --mmhid 64 --mode omic --model_name omic --niter 0 --niter_decay 30 --omic_dim 32 --omic_gate 1 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 1 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type all --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 1 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1 --use_rnaseq 0 --use_vgg_features 0 --verbose 1 --weight_decay 0.0005 --use_patches 0
```

### SNN + CNN KIRC
```bash
python IG_pathomic_KIRC.py --GNN GCN --act_type Sigmoid --batch_size 32 --beta1 0.5 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 1 --init_gain 0.02 --init_type none --input_size_omic 362 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0003 --lr 0.0001 --lr_policy linear --measure 1 --mmhid 64 --mode pathomic --model_name pathomic_fusion --niter 10 --niter_decay 20 --omic_dim 32 --omic_gate 0 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 0 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type omic --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 1 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1 --use_rnaseq 1 --use_vgg_features 1 --verbose 1 --weight_decay 0.0004  --use_patches 0
```

## 5.3 Gradcam
### GBMLGG
```bash
python  Gradcam_path_GBMLGG.py --exp_name surv_15_rnaseq --task surv --mode path --model_name path --niter 0 --niter_decay 50 --batch_size 8 --lr 0.0005 --reg_type none --lambda_reg 0 --use_vgg_features 0 --use_patches 1 --gpu_ids 0
```

### KIRC
```bash
python Gradcam_path_KIRC.py --GNN GCN --act_type Sigmoid --batch_size 8 --beta1 0.9 --beta2 0.999 --checkpoints_dir ../checkpoints/TCGA_KIRC/ --dataroot ../data/TCGA_KIRC/ --dropout_rate 0.25 --end_k 15 --epoch_count 1 --exp_name surv_15 --final_lr 0.1 --finetune 1 --fusion_type pofusion --gpu_ids 0 --grph_dim 32 --grph_gate 1 --grph_scale 1 --init_gain 0.02 --init_type none --input_size_omic 80 --input_size_path 512 --label_dim 1 --lambda_cox 1 --lambda_nll 1 --lambda_reg 0.0 --lr 0.0005 --lr_policy linear --measure 1 --mmhid 64 --mode path --model_name path --niter 0 --niter_decay 30 --omic_dim 32 --omic_gate 1 --omic_scale 1 --optimizer_type adam --path_dim 32 --path_gate 1 --path_scale 1 --patience 0.005 --pooling_ratio 0.2 --print_every 0 --reg_type none --roi_dir KIRC_st --save_at 20 --skip 0 --start_k 13 --task surv --useRNA 0 --useSN 1 --use_bilinear 1 --use_edges 1 --use_rnaseq 0 --use_vgg_features 0 --verbose 1 --weight_decay 0.0004 –-use_patches 0
```

# Statement of Chatgpt
The following Codes involving the contribution of Chatgpt:
1. GradCAM classes in Gradcam_path_GBMGLL.py
2. GradCAM classes in Gradcam_path_KIRC.py
3. The pre-aggregate Bimodal and Trimodal Fusion in networks.py
The structure of the Self Attention refers to Chatgpt, and the original Network structure
We put the idea and structure into the Chatgpt to get the basic structure and then fix the structure and change the parameters and hyperparameters by ourselves
