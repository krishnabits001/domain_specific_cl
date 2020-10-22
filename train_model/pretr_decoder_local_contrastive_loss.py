import os

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

import matplotlib
matplotlib.use('Agg')

import numpy as np
#to make directories
import pathlib

import sys
sys.path.append('../')

from utils import *


import argparse
parser = argparse.ArgumentParser()
#data set type
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc','prostate_md','mmwhs'])
#no of training images
parser.add_argument('--no_of_tr_imgs', type=str, default='tr52', choices=['tr52','tr22','tr10'])
#combination of training images
parser.add_argument('--comb_tr_imgs', type=str, default='c1')
#learning rate of Enc net
parser.add_argument('--lr_reg', type=float, default=0.001)

# data aug - 0 - disabled, 1 - enabled
parser.add_argument('--data_aug', type=int, default=0, choices=[0,1])
# version of run
parser.add_argument('--ver', type=int, default=0)
# temperature_scaling factor
parser.add_argument('--temp_fac', type=float, default=0.1)
# bounding box dim - dimension of the cropped image. Ex. if bbox_dim=100, then 100 x 100 region is randomly cropped from original image of size W x W & then re-sized to W x W.
# Later, these re-sized images are used for pre-training using global contrastive loss.
parser.add_argument('--bbox_dim', type=int, default=100)

#data aug - 0 - disabled, 1 - enabled
parser.add_argument('--pretr_data_aug', type=int, default=0, choices=[0,1])
# bounding box dim - dimension of the cropped image. Ex. if bbox_dim=100, then 100 x 100 region is randomly cropped from original image of size W x W & then re-sized to W x W.
# Later, these re-sized images are used for pre-training using global contrastive loss.
parser.add_argument('--pretr_bbox_dim', type=int, default=100)
#no of training images
parser.add_argument('--pretr_no_of_tr_imgs', type=str, default='tr52', choices=['tr52','tr22','tr10'])
#combination of training images
parser.add_argument('--pretr_comb_tr_imgs', type=str, default='c1', choices=['c1'])
#no of iterations to run
parser.add_argument('--pretr_n_iter', type=int, default=10001)
#pretr version
parser.add_argument('--pretr_ver', type=int, default=0)

# type of global_loss_exp_no for global contrastive loss
# 0 - default loss formulation as in simCLR (sample images in a batch from all volumes)
# 1 - prevent negatives to be contrasted for images coming from corresponding partitions from other volumes for a given positive image.
# 2 - as in (1) + additionally match positive image to corresponding slice from similar partition in another volume
parser.add_argument('--global_loss_exp_no', type=int, default=2)

# type of local_loss_exp_no for Local contrastive loss
# 0 - default loss formulation. Sample local regions from two images. these 2 images are intensity transformed version of same image.
# 1 - (0) + sample local regions to match from 2 differnt images that are from 2 different volumes but they belong to corresponding local regions of similar partitions.
parser.add_argument('--local_loss_exp_no', type=int, default=0)
#no_of_partitions per volume
parser.add_argument('--n_parts', type=int, default=4)


#Load encoder weights from pre-trained contrastive loss model; 1-Yes, 0-No.
#Fine-tune which layers: 1- FT only dec. layer (enc. wgts are frozen), 2- FT all layers
#parser.add_argument('--pretr_dec', type=int, default=1)
#FT on best val loss model (0) or last step model from training (1)
#parser.add_argument('--load_model_type', type=int, default=1)
#Decoder Size: 0-small, 1-large
#parser.add_argument('--dec_size', type=int, default=1)
# 1 - crop + color aug. , 0 - for only color aug
#parser.add_argument('--aug_type', type=int, default=0)

# no. of local regions to consider in the feature map for local contrastive loss computation
parser.add_argument('--no_of_local_regions', type=int, default=13)

#no. of decoder blocks used. Here, 1 means 1 decoder block used, 2 is for 2 blocks,..., 5 is for all blocks aka full decoder.
parser.add_argument('--no_of_decoder_blocks', type=int, default=1)

#local_reg_size - 1 for 3x3 local region size in the feature map. <local_reg> -> flat -> w*flat -> 128 bit z vector matching;
#               - 0 for 1x1 local region size in the feature map
parser.add_argument('--local_reg_size', type=int, default=1)
# #nn_type - 1 for far away neighbouring fmap patches; 0 - for mostly nearby patches
# parser.add_argument('--nn_type', type=int, default=0)
#wgt_en - 1 for having extra weight layer on top of 'z' vector from local region.
#      - 0 for not having any weight layer.
parser.add_argument('--wgt_en', type=int, default=1)
#wgt_fac - for weighted loss of negative patch samples;
# 1 - for normal ratio(nearby samples contribute lower loss than far away samples)
# 2 - for inverse ratio(far away samples contribute lower loss than nearby samples)
#parser.add_argument('--wgt_fac_en', type=int, default=0)

#use both - global and local losses
#parser.add_argument('--both_loss_en', type=int, default=0)
#no. of neighbouring local regions sampled from the feature maps to act as negative samples in local contrastive loss
# for a given positive local region - currently 5 local regions are chosen from each feature map.
parser.add_argument('--no_of_neg_local_regions', type=int, default=5)

#overide the no. of negative (-ve) local neighbouring regions chosen for local loss computation- 4 for L^{D} (local_loss_exp_no=1) - due to memory issues
parser.add_argument('--no_of_neg_regs_override', type=int, default=4)

#batch_size value for local_loss
parser.add_argument('--bt_size', type=int,default=12)

#no of iterations to run
parser.add_argument('--n_iter', type=int, default=10001)

parse_config = parser.parse_args()
#parse_config = parser.parse_args(args=[])

if parse_config.dataset == 'acdc':
    print('load acdc configs')
    import experiment_init.init_acdc as cfg
    import experiment_init.data_cfg_acdc as data_list
elif parse_config.dataset == 'mmwhs':
    print('load mmwhs configs')
    import experiment_init.init_mmwhs as cfg
    import experiment_init.data_cfg_mmwhs as data_list
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

if parse_config.dataset == 'acdc':
    print('set acdc orig img dataloader handle')
    orig_img_dt=dt.load_acdc_imgs
elif parse_config.dataset == 'mmwhs':
    print('set mmwhs orig img dataloader handle')
    orig_img_dt=dt.load_mmwhs_imgs
elif parse_config.dataset == 'prostate_md':
    print('set prostate_md orig img dataloader handle')
    orig_img_dt=dt.load_prostate_imgs_md

#  load model object
from models import modelObj
model = modelObj(cfg)
#  load f1_utils object
from f1_utils import f1_utilsObj
f1_util = f1_utilsObj(cfg,dt)

######################################

######################################
# Restore the model and initialize the encoder with the pre-trained weights from last epoch
######################################
#define save_dir of pre-trained encoder model
save_dir=str(cfg.srt_dir)+'/models/'+str(parse_config.dataset)+'/trained_models/pretrain_encoder_with_global_contrastive_loss/'

save_dir=str(save_dir)+'/bt_size_'+str(parse_config.bt_size)+'/'

if(parse_config.pretr_data_aug==0):
    save_dir=str(save_dir)+'/no_data_aug/'
else:
    save_dir=str(save_dir)+'/with_data_aug/'

save_dir=str(save_dir)+'global_loss_exp_no_'+str(parse_config.global_loss_exp_no)+'_n_parts_'+str(parse_config.n_parts)+'/'

save_dir=str(save_dir)+'temp_fac_'+str(parse_config.temp_fac)+'/'

save_dir=str(save_dir)+str(parse_config.pretr_no_of_tr_imgs)+'/'+str(parse_config.pretr_comb_tr_imgs)+'_v'+str(parse_config.pretr_ver)+'/enc_bbox_dim_'+str(parse_config.pretr_bbox_dim)+'_n_iter_'+str(parse_config.pretr_n_iter)+'_lr_reg_'+str(parse_config.lr_reg)+'/'

print('save dir ',save_dir)
######################################
# Define Encoder(e) + g_1 network graph for pre-training
tf.reset_default_graph()
print('cfg.temp_fac',parse_config.lr_reg,parse_config.temp_fac)
ae = model.encoder_pretrain_net(learn_rate_seg=parse_config.lr_reg,temp_fac=parse_config.temp_fac, \
                        global_loss_exp_no=parse_config.global_loss_exp_no,n_parts=parse_config.n_parts)
######################################
# load pre-trained encoder weight variables and their values
mp_best=get_chkpt_file(save_dir)
print('load last epoch model from pre-training')
print('mp_best',mp_best)

saver_rnet = tf.train.Saver()
sess_rnet = tf.Session(config=config)
saver_rnet.restore(sess_rnet, mp_best)
print("Model restored")

#get all variable names and their values
print('Loading trainable vars')
variables_names = [v.name for v in tf.trainable_variables()]
var_values = sess_rnet.run(variables_names)
sess_rnet.close()
print('loaded encoder weight values from pre-trained model with global contrastive loss')
#print(tf.trainable_variables())
######################################

######################################
#define directory to save the pre-training model of decoder with encoder weights frozen (encoder weights obtained from earlier pre-training step)
save_dir=str(cfg.srt_dir)+'/models/'+str(parse_config.dataset)+'/trained_models/pretrain_decoder_with_local_contrastive_loss/'

save_dir=str(save_dir)+'/load_encoder_wgts_from_pretrained_model/'

if(parse_config.data_aug==0):
    save_dir=str(save_dir)+'/no_data_aug/'
else:
    save_dir=str(save_dir)+'/with_data_aug/'

save_dir=str(save_dir)+'local_loss_exp_no_'+str(parse_config.local_loss_exp_no)+'_global_loss_exp_no_'+str(parse_config.global_loss_exp_no)\
             +'_n_parts_'+str(parse_config.n_parts)+'/'

save_dir=str(save_dir)+'temp_fac_'+str(parse_config.temp_fac)+'/'

if(parse_config.local_reg_size==1):
    if(parse_config.wgt_en==1):
        save_dir=str(save_dir)+'local_reg_size_3x3_wgt_en/'
    else:
        save_dir=str(save_dir)+'local_reg_size_3x3_wgt_dis/'
else:
    save_dir=str(save_dir)+'local_reg_size_1x1_wgt_dis/'

save_dir=str(save_dir)+'no_of_decoder_blocks_'+str(parse_config.no_of_decoder_blocks)+'/'

save_dir=str(save_dir)+'no_of_local_regions_'+str(parse_config.no_of_local_regions)

if(parse_config.local_loss_exp_no==1):
    #parse_config.no_of_neg_local_regions=4
    parse_config.no_of_neg_regs_override=4
    save_dir=save_dir+'_no_of_neg_regions_'+str(parse_config.no_of_neg_regs_override)+'/'
else:
    save_dir=save_dir+'_no_of_neg_regions_'+str(parse_config.no_of_neg_local_regions)+'/'

save_dir=str(save_dir)+'pretrain_only_decoder_weights/'

save_dir=str(save_dir)+str(parse_config.no_of_tr_imgs)+'/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+'/enc_bbox_dim_'+str(parse_config.bbox_dim)+'_n_iter_'+str(parse_config.n_iter)+'_lr_reg_'+str(parse_config.lr_reg)+'/'

print('save dir ',save_dir)
######################################

######################################
# Load unlabeled training images only & no labels are loaded here
######################################
#load unlabeled volumes id numbers to pre-train the decoder
unl_list = data_list.train_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
print('load unlabeled images for pre-training')
#print(unl_list)

if(parse_config.local_loss_exp_no==0):
    unl_imgs=dt.load_cropped_img_labels(unl_list,label_present=0)
else:
    _,unl_imgs,_,_=load_val_imgs(unl_list,dt,orig_img_dt)

######################################

######################################
# Define checkpoint file to save CNN architecture and learnt hyperparameters
checkpoint_filename='train_decoder_wgts_'+str(parse_config.dataset)
logs_path = str(save_dir)+'tensorflow_logs/'
best_model_dir=str(save_dir)+'best_model/'
######################################
pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True)

######################################
# Define Encoder(e) + 'l' decoder blocks (d_l) + g_2 network graph for pre-training; l - no. of decoder blocks.
tf.reset_default_graph()
ae = model.decoder_pretrain_net(learn_rate_seg=parse_config.lr_reg,temp_fac=parse_config.temp_fac, \
                             no_of_local_regions=parse_config.no_of_local_regions,no_of_decoder_blocks=parse_config.no_of_decoder_blocks, \
                             local_loss_exp_no=parse_config.local_loss_exp_no,local_reg_size=parse_config.local_reg_size,\
                             wgt_en=parse_config.wgt_en,no_of_neg_local_regions=parse_config.no_of_neg_local_regions,\
                             no_of_neg_regs_override=parse_config.no_of_neg_regs_override)

if(parse_config.local_loss_exp_no==1):
    cfg.batch_size_ft=12

ae_rc = model.brit_cont_net(batch_size=cfg.batch_size_ft)
######################################

######################################
#writer for train summary
train_writer = tf.summary.FileWriter(logs_path)
#writer for dice score and val summary
#dsc_writer = tf.summary.FileWriter(logs_path)
val_sum_writer = tf.summary.FileWriter(logs_path)
######################################

######################################
# Define session and saver
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
#saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=2)
saver = tf.train.Saver(max_to_keep=2)
######################################

######################################
#  assign values to all trainable ops of network
assign_op=[]
print('Init of trainable vars')
for new_var in tf.trainable_variables():
    for var, var_val in zip(variables_names, var_values):
        if (str(var) == str(new_var.name) and ('reg_' not in str(new_var.name))):
            #print('match name',new_var.name,var)
            tmp_op=new_var.assign(var_val)
            assign_op.append(tmp_op)

sess.run(assign_op)
print('init done for all the encoder network weights and biases from pre-trained model')
######################################

######################################
# parameters values set for pre-training of Decoder
step_val=parse_config.n_iter
start_epoch=0
n_epochs=step_val

tr_loss_list=[]
# print_lst=[250,500,1000,2000,3000,4000,6000,8000]
######################################

######################################
# loop over all the epochs to pre-train the Decoder
for epoch_i in range(start_epoch,n_epochs):
    if (parse_config.local_loss_exp_no == 0):
        #########################
        # G^{R} -  Match corresponding local regions across two intensity transformed images (x_i_a1,x_i_a2) obtained from an original image (x_i)
        # x_i_a1 = t1(x_i) and x_i_a2 = t2(x_i) where t1, t2 are two different random intensity transformations.
        #########################
        # original images batch sampled from unlabeled images
        img_batch = shuffle_minibatch([unl_imgs], batch_size=cfg.batch_size_ft, labels_present=0)

        # make 2 different sets of images from this chosen batch.
        # Each set is applied with different intensity augmentation (aug) - brightness + distortion
        # Set 1 - random intensity aug
        color_batch1 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: img_batch})
        # Set 2 - different random intensity aug
        color_batch2 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: img_batch})

        # Stitch these 2 augmented sets into 1 batch for pre-training
        cat_batch = stitch_two_crop_batches([color_batch1, color_batch2], cfg, cfg.batch_size_ft)

    elif(parse_config.local_loss_exp_no==1):
        #########################
        # G^{D} -  Match corresponding local regions across two intensity transformed images (x_i_a1,x_j_a1) from 2 different volumes from same partition.
        # where x_i_a1, x_j_a1 are from volume i and j, respectively. (i not equal to j).
        #########################
        n_vols,n_parts=len(unl_list),parse_config.n_parts
        # original images batch sampled from unlabeled images
        img_batch=sample_minibatch_for_global_loss_opti(unl_imgs,cfg,3*n_parts,n_vols,n_parts)
        img_batch=img_batch[0:cfg.batch_size_ft]

        crop_batch1=img_batch
        # Set 1 - random intensity aug
        color_batch1=sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch1})
        # Set 2 - different random intensity aug
        color_batch2=sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch1})

        # stitch 2 batches into 1 batch for pre-training
        cat_batch = np.concatenate([color_batch1[0:cfg.batch_size_ft], color_batch2[0:cfg.batch_size_ft]], axis=0)


    #Run optimizer update on the training unlabeled data
    train_summary,tr_loss,_=sess.run([ae['train_summary'],ae['reg_cost'],ae['optimizer_unet_dec']],\
                                     feed_dict={ae['x']:cat_batch,ae['train_phase']:True})

    if(epoch_i%cfg.val_step_update==0):
        train_writer.add_summary(train_summary, epoch_i)
        train_writer.flush()

        tr_loss_list.append(np.mean(tr_loss))
        print('epoch_i,tr_loss', epoch_i, np.mean(tr_loss))

    if ((epoch_i==n_epochs-1)):
        # model saved at the last epoch of training
        mp = str(save_dir) + str(checkpoint_filename) + '_epochs_' + str(epoch_i) + ".ckpt"
        saver.save(sess, mp)
        try:
            mp_best
        except NameError:
            mp_best=mp
            
    # if(epoch_i in print_lst):
    #     f1_util.plt_seg_loss([tr_loss_list],save_dir,title_str='pretrain_decoder_net',plt_name='tr_seg_loss',ep_no=epoch_i)

######################################
# Plot the training loss of training images over all the epochs of training.
f1_util.plt_seg_loss([tr_loss_list],save_dir,title_str='pretrain_decoder_net',plt_name='tr_seg_loss',ep_no=epoch_i)
######################################

