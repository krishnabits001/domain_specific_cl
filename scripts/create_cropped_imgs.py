# script to crop the images into the target resolution and save them.
import numpy as np
import pathlib

import nibabel as nib

import os.path
from os import path

import sys
sys.path.append("<path_to_git_code>/git_code")

import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc','prostate_md'])

parse_config = parser.parse_args()
#parse_config = parser.parse_args(args=[])

if parse_config.dataset == 'acdc':
    print('load acdc configs')
    import experiment_init.init_acdc as cfg
    import experiment_init.data_cfg_acdc as data_list
elif parse_config.dataset == 'prostate_md':
    print('load prostate_md configs')
    import experiment_init.init_prostate_md as cfg
    import experiment_init.data_cfg_prostate_md as data_list
else:
    raise ValueError(parse_config.dataset)

######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)

if parse_config.dataset == 'acdc' :
    #print('set acdc orig img dataloader handle')
    orig_img_dt=dt.load_acdc_imgs
    start_id,end_id=1,101
elif parse_config.dataset == 'prostate_md':
    #print('set prostate_md orig img dataloader handle')
    orig_img_dt=dt.load_prostate_imgs_md
    start_id,end_id=0,48

# For loop to go over all available images
for index in range(start_id,end_id):
    if(index<10):
        test_id='00'+str(index)
    elif(index<100):
        test_id='0'+str(index)
    else:
        test_id=str(index)
    test_id_l=[test_id]
    
    if parse_config.dataset == 'acdc' :
        file_path=str(cfg.data_path_tr)+str(test_id)+'/patient'+str(test_id)+'_frame01.nii.gz'
        mask_path=str(cfg.data_path_tr)+str(test_id)+'/patient'+str(test_id)+'_frame01_gt.nii.gz'
    elif parse_config.dataset == 'prostate_md':
        file_path=str(cfg.data_path_tr)+str(test_id)+'/img.nii.gz'
        mask_path=str(cfg.data_path_tr)+str(test_id)+'/mask.nii.gz'
    
    #check if image file exists
    if(path.exists(file_path)):
        print('crop',test_id)
    else:
        print('continue',test_id)
        continue
    
    #check if mask exists
    if(path.exists(mask_path)):
        # Load the image &/mask
        img_sys,label_sys,pixel_size,affine_tst= orig_img_dt(test_id_l,ret_affine=1,label_present=1)
        # Crop the loaded image &/mask to target resolution
        cropped_img_sys,cropped_mask_sys = dt.preprocess_data(img_sys, label_sys, pixel_size)
    else:
        # Load the image &/mask
        img_sys,pixel_size,affine_tst= orig_img_dt(test_id_l,ret_affine=1,label_present=0)
        #dummy mask with zeros
        label_sys=np.zeros_like(img_sys)
        # Crop the loaded image &/mask to target resolution
        cropped_img_sys = dt.preprocess_data(img_sys, label_sys, pixel_size, label_present=0)
    
    #output directory to save cropped image &/mask
    save_dir_tmp=str(cfg.data_path_tr_cropped)+str(test_id)+'/'
    pathlib.Path(save_dir_tmp).mkdir(parents=True, exist_ok=True)

    if (parse_config.dataset == 'acdc') :             
        affine_tst[0,0]=-cfg.target_resolution[0]
        affine_tst[1,1]=-cfg.target_resolution[1]
    elif (parse_config.dataset == 'prostate_md') :   
        affine_tst[0,0]=cfg.target_resolution[0]
        affine_tst[1,1]=cfg.target_resolution[1]

    #Save the cropped image &/mask
    array_img = nib.Nifti1Image(cropped_img_sys, affine_tst)
    pred_filename = str(save_dir_tmp)+'img_cropped.nii.gz'
    nib.save(array_img, pred_filename)
    if(path.exists(mask_path)):
        array_mask = nib.Nifti1Image(cropped_mask_sys.astype(np.int16), affine_tst)
        pred_filename = str(save_dir_tmp)+'mask_cropped.nii.gz'
        nib.save(array_mask, pred_filename)

