import os

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

import matplotlib
matplotlib.use('Agg')

import numpy as np
# to make directories
import pathlib

import sys
sys.path.append('../')

from utils import *


import argparse
parser = argparse.ArgumentParser()
#dataset selection
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc','prostate_md','mmwhs'])
#no of training images
parser.add_argument('--no_of_tr_imgs', type=str, default='tr52', choices=['tr52','tr22','tr10'])
#combination of training images
parser.add_argument('--comb_tr_imgs', type=str, default='c1')
#learning rate of Enc net
parser.add_argument('--lr_reg', type=float, default=0.001)

#data aug - 0 - disabled, 1 - enabled
parser.add_argument('--data_aug', type=int, default=0, choices=[0,1])
#version of run
parser.add_argument('--ver', type=int, default=0)

# temperature_scaling factor
parser.add_argument('--temp_fac', type=float, default=0.1)
# bounding box dim - dimension of the cropped image. Ex. if bbox_dim=100, then 100 x 100 region is randomly cropped from original image of size W x W & then re-sized to W x W.
# Later, these re-sized images are used for pre-training using global contrastive loss.
parser.add_argument('--bbox_dim', type=int, default=100)

# type of global_loss_exp_no for global contrastive loss - used to pre-train the Encoder (e)
# 0 - G^{R}  - default loss formulation as in simCLR (sample images in a batch from all volumes)
# 1 - G^{D-} - prevent negatives to be contrasted for images coming from corresponding partitions from other volumes for a given positive image.
# 2 - G^{D}  - as in (1) + additionally match positive image to corresponding slice from similar partition in another volume
parser.add_argument('--global_loss_exp_no', type=int, default=2)
#no_of_partitions selected per volume
parser.add_argument('--n_parts', type=int, default=4)

#no of iterations to run
parser.add_argument('--n_iter', type=int, default=10001)

#batch_size value - if global_loss_exp_no = 1, bt_size = 12; if global_loss_exp_no = 2, bt_size = 8
parser.add_argument('--bt_size', type=int,default=12)

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

cfg.batch_size_ft=parse_config.bt_size

######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)

if parse_config.dataset == 'acdc' :
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
#define directory to save the pre-training model of encoder
save_dir=str(cfg.srt_dir)+'/models/'+str(parse_config.dataset)+'/trained_models/pretrain_encoder_with_global_contrastive_loss/'

save_dir=str(save_dir)+'/bt_size_'+str(parse_config.bt_size)+'/'

if(parse_config.data_aug==0):
    save_dir=str(save_dir)+'/no_data_aug/'
else:
    save_dir=str(save_dir)+'/with_data_aug/'

save_dir=str(save_dir)+'global_loss_exp_no_'+str(parse_config.global_loss_exp_no)+'_n_parts_'+str(parse_config.n_parts)+'/'

save_dir=str(save_dir)+'temp_fac_'+str(parse_config.temp_fac)+'/'

save_dir=str(save_dir)+str(parse_config.no_of_tr_imgs)+'/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+'/enc_bbox_dim_'+str(parse_config.bbox_dim)+'_n_iter_'+str(parse_config.n_iter)+'_lr_reg_'+str(parse_config.lr_reg)+'/'

print('save dir ',save_dir)
######################################

######################################
# Load unlabeled training images only & no labels are loaded here
######################################
# load unlabeled volumes id numbers to pre-train the encoder
unl_list = data_list.train_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
print('load unlabeled volumes for pre-training')

if(parse_config.global_loss_exp_no==0):
    unl_imgs=dt.load_cropped_img_labels(unl_list,label_present=0)
else:
    _,unl_imgs,_,_=load_val_imgs(unl_list,dt,orig_img_dt)
######################################

######################################
# Define checkpoint file to save CNN architecture and learnt hyperparameters
checkpoint_filename='train_encoder_wgts_'+str(parse_config.dataset)
logs_path = str(save_dir)+'tensorflow_logs/'
best_model_dir=str(save_dir)+'best_model/'
######################################
pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True)


######################################
# Define Encoder(e) + g_1 network graph for pre-training
tf.reset_default_graph()
ae = model.encoder_pretrain_net(learn_rate_seg=parse_config.lr_reg,temp_fac=parse_config.temp_fac,\
                        global_loss_exp_no=parse_config.global_loss_exp_no,n_parts=parse_config.n_parts)

# define network/graph to apply random contrast and brightness on input images
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
# parameters values set for pre-training of Encoder
step_val=parse_config.n_iter
start_epoch=0
n_epochs=step_val

tr_loss_list=[]
######################################

######################################
# loop over all the epochs to pre-train the Encoder Network.
for epoch_i in range(start_epoch,n_epochs):

    # if (parse_config.global_loss_exp_no == 0):
    #     #########################
    #     # G^{R} -  default loss formulation as in simCLR (sample images in a batch from all volumes)
    #     #########################
    #     # original images batch sampled from unlabeled images
    #     img_batch = shuffle_minibatch([unl_imgs], batch_size=cfg.batch_size_ft, labels_present=0)
    #
    #     # make 2 different sets of images from this chosen batch.
    #     # Each set is applied with different set of crop and intensity augmentation (aug) - brightness + distortion
    #
    #     # Set 1 - random crop followed by random intensity aug
    #     crop_batch1 = crop_batch([img_batch], cfg, cfg.batch_size_ft, parse_config.bbox_dim)
    #     color_batch1 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch1})
    #
    #     # Set 2 - different random crop followed by random intensity aug
    #     crop_batch2 = crop_batch([img_batch], cfg, cfg.batch_size_ft, parse_config.bbox_dim)
    #     color_batch2 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch2})
    #
    #     # Stitch these 2 augmented sets into 1 batch for pre-training
    #     cat_batch = stitch_two_crop_batches([color_batch1, color_batch2], cfg, cfg.batch_size_ft)

    if(parse_config.global_loss_exp_no==1):
        #########################
        # G^{D-} - prevent negatives to be contrasted for images coming from corresponding partitions from other volumes for a given positive image.
        #########################
        n_vols,n_parts=len(unl_list),parse_config.n_parts

        # original images batch sampled from unlabeled images
        # First, we randomly select 'm' volumes out of M. Then, we sample 1 image for each partition of the selected 'm' volumes. Overall we get m * n_parts images.
        img_batch=sample_minibatch_for_global_loss_opti(unl_imgs,cfg,2*cfg.batch_size_ft,n_vols,n_parts)
        img_batch=img_batch[0:cfg.batch_size_ft]

        # make 2 different sets of images from this chosen batch.
        # Each set is applied with different set of crop and intensity augmentation (aug) - brightness + distortion

        # Aug Set 1 - crop followed by intensity aug
        crop_batch1=crop_batch([img_batch],cfg,cfg.batch_size_ft,parse_config.bbox_dim)
        color_batch1=sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch1})

        # Aug Set 2 - crop followed by intensity aug
        crop_batch2=crop_batch([img_batch],cfg,cfg.batch_size_ft,parse_config.bbox_dim)
        color_batch2=sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch2})

        # Stitch 3 sets: original images batch, 2 different augmented version of original images batch into 1 batch for pre-training
        cat_batch = np.concatenate([img_batch,color_batch1,color_batch2],axis=0)

    elif(parse_config.global_loss_exp_no==2):
        #########################
        # G^{D} - as in (1) + additionally match positive image to corresponding slice from similar partition in another volume
        #########################
        if(parse_config.bt_size!=12):
            n_vols,n_parts=len(unl_list),parse_config.n_parts
        else:
            n_vols,n_parts=5,parse_config.n_parts

        # Set 1 of original images batch sampled from unlabeled images
        # First, we randomly select 'm' volumes out of M. Then, we sample 1 image for each partition of the selected 'm' volumes. Overall we get m * n_parts images.
        img_batch=sample_minibatch_for_global_loss_opti(unl_imgs,cfg,2*cfg.batch_size_ft,n_vols,n_parts)
        crop_batch1 = crop_batch([img_batch], cfg, 2 * cfg.batch_size_ft, parse_config.bbox_dim)

        # Augmented version 1 of Set 1 - augmentations applied are crop followed by intensity augmentation (aug).
        color_batch1_v1=sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch1[0:cfg.batch_size_ft]})
        # Augmented version 2 of Set 1 - augmentations applied are crop followed by intensity augmentation (aug).
        color_batch1_v2=sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch1[cfg.batch_size_ft:2*cfg.batch_size_ft]})
        aug_batch1 = np.concatenate([color_batch1_v1,color_batch1_v2],axis=0)

        # Set 2 of original images batch sampled from unlabeled images (Different to Set 1)
        # Again, we randomly select 'm' volumes out of M. Then, we sample 1 image for each partition of the selected 'm' volumes. Overall we get m * n_parts images.
        img_batch_t2=sample_minibatch_for_global_loss_opti(unl_imgs,cfg,2*cfg.batch_size_ft,n_vols,n_parts)
        crop_batch2=crop_batch([img_batch_t2],cfg,2*cfg.batch_size_ft,parse_config.bbox_dim)

        # Augmented version 1 of Set 2 - augmentations applied are crop followed by intensity augmentation (aug).
        color_batch2_v1=sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch2[0:cfg.batch_size_ft]})
        # Augmented version 2 of Set 2 - augmentations applied are crop followed by intensity augmentation (aug).
        color_batch2_v2=sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch2[cfg.batch_size_ft:2*cfg.batch_size_ft]})
        aug_batch2 = np.concatenate([color_batch2_v1,color_batch2_v2],axis=0)

        # Stitch 4 sets: (a) Set 1 of original images batch
        # (b) Set 1's 2 different augmented versions concatenated (aug_batch1)
        # (c) Set 2 of original images batch (different to set 1)
        # (d) Set 2's 2 different augmented versions concatenated (aug_batch2)
        cat_batch=stitch_batch_global_loss_gd(cfg,img_batch,aug_batch1,img_batch_t2,aug_batch2,n_parts)

    elif (parse_config.global_loss_exp_no == 4):
        #########################
        # G^{D} - as in (1) + additionally match positive image to corresponding slice from similar partition in another volume
        #########################
        if (parse_config.bt_size!=12):
            n_vols,n_parts=len(unl_list),parse_config.n_parts
        else:
            n_vols,n_parts=5,parse_config.n_parts

        n_vols,n_parts=len(unl_list),parse_config.n_parts

        # original images batch sampled from unlabeled images
        # First, we randomly select 'm' volumes out of M. Then, we sample 1 image for each partition of the selected 'm' volumes. Overall we get m * n_parts images.
        img_batch = sample_minibatch_for_global_loss_opti(unl_imgs, cfg, 2 * cfg.batch_size_ft, n_vols, n_parts)
        img_batch = img_batch[0:cfg.batch_size_ft]

        # make 2 different sets of images from this chosen batch.
        # Each set is applied with different set of crop and intensity augmentation (aug) - brightness + distortion

        # Aug Set 1 - crop followed by intensity aug
        crop_batch1 = crop_batch([img_batch], cfg, cfg.batch_size_ft, parse_config.bbox_dim)
        color_batch1 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch1})
        # Set 2
        crop_batch2 = crop_batch([img_batch], cfg, cfg.batch_size_ft, parse_config.bbox_dim)
        color_batch2 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch2})

        # Set 2 of original images batch sampled from unlabeled images (Different to Set 1)
        # Again, we randomly select 'm' volumes out of M. Then, we sample 1 image for each partition of the selected 'm' volumes. Overall we get m * n_parts images.
        img_batch_t2 = sample_minibatch_for_global_loss_opti(unl_imgs, cfg, 2 * cfg.batch_size_ft, n_vols, n_parts)
        img_batch_t2 = img_batch_t2[0:cfg.batch_size_ft]

        # Aug Set 2 - crop followed by intensity aug
        crop_batch3 = crop_batch([img_batch_t2], cfg, cfg.batch_size_ft, parse_config.bbox_dim)
        color_batch3 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch3})
        # Set 2
        crop_batch4 = crop_batch([img_batch_t2], cfg, cfg.batch_size_ft, parse_config.bbox_dim)
        color_batch4 = sess.run(ae_rc['rd_fin'], feed_dict={ae_rc['x_tmp']: crop_batch4})

        # Stitch 3 sets: original images batch, 2 different augmented version of original images batch into 1 batch for pre-training
        #cat_batch = np.concatenate([img_batch, color_batch1, color_batch2], axis=0)

        # Stitch 4 sets: (a) Set 1 of original images batch
        # (b) Set 1's 2 different augmented versions concatenated (aug_batch1)
        # (c) Set 2 of original images batch (different to set 1)
        # (d) Set 2's 2 different augmented versions concatenated (aug_batch2)
        cat_batch = stitch_batch_global_loss_gdnew(cfg, img_batch, color_batch1, color_batch2, img_batch_t2, color_batch3, color_batch4, n_parts)

    #Run optimizer update on the training unlabeled data
    train_summary,tr_loss,_=sess.run([ae['train_summary'],ae['reg_cost'],ae['optimizer_unet_reg']],\
                                     feed_dict={ae['x']:cat_batch,ae['train_phase']:True})

    if(epoch_i%cfg.val_step_update==0):
        train_writer.add_summary(train_summary, epoch_i)
        train_writer.flush()
        print('epoch_i,tr_loss,val_loss', epoch_i, np.mean(tr_loss))
        tr_loss_list.append(np.mean(tr_loss))


    if ((epoch_i==n_epochs-1)):
        # model saved at the last epoch of training
        mp = str(save_dir) + str(checkpoint_filename) + '_epochs_' + str(epoch_i) + ".ckpt"
        saver.save(sess, mp)
        try:
            mp_best
        except NameError:
            mp_best=mp
            
print('pre-training completed')
######################################
# Plot the training loss of training images over all the epochs of training.
f1_util.plt_seg_loss([tr_loss_list],save_dir,title_str='pretrain_encoder_net',plt_name='tr_seg_loss',ep_no=epoch_i)

######################################
