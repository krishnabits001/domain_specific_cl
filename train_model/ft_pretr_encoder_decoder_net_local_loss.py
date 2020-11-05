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
parser.add_argument('--no_of_tr_imgs', type=str, default='tr1', choices=['tr1', 'tr2', 'tr8','tr20'])
#combination of training images
parser.add_argument('--comb_tr_imgs', type=str, default='c1')
#learning rate of seg unet
parser.add_argument('--lr_seg', type=float, default=0.001)

#data aug - 0 - disabled, 1 - enabled
parser.add_argument('--data_aug', type=int, default=1, choices=[0,1])
#version of run
parser.add_argument('--ver', type=int, default=0)

# Pre-training configs
#no of training images
parser.add_argument('--pretr_no_of_tr_imgs', type=str, default='tr52', choices=['tr52','tr22','tr10'])
#combination of training images
parser.add_argument('--pretr_comb_tr_imgs', type=str, default='c1', choices=['c1'])
#version of run
parser.add_argument('--pretr_ver', type=int, default=0)
#no of iterations to run
parser.add_argument('--pretr_n_iter', type=int, default=10001)
#data augmentation used in pre-training
parser.add_argument('--pretr_data_aug', type=int, default=0)
# bounding box dim - dimension of the cropped image. Ex. if bbox_dim=100, then 100 x 100 region is randomly cropped from original image of size W x W & then re-sized to W x W.
# Later, these re-sized images are used for pre-training using global contrastive loss.
parser.add_argument('--pretr_cont_bbox_dim', type=int, default=100)
# temperature_scaling factor
parser.add_argument('--temp_fac', type=float, default=0.1)
#learning rate of seg unet
parser.add_argument('--lr_reg', type=float, default=0.001)

# type of global_loss_exp_no for global contrastive loss - used to pre-train the Encoder (e)
# 0 - G^{R}  - default loss formulation as in simCLR (sample images in a batch from all volumes)
# 1 - G^{D-} - prevent negatives to be contrasted for images coming from corresponding partitions from other volumes for a given positive image.
# 2 - G^{D}  - as in (1) + additionally match positive image to corresponding slice from similar partition in another volume
parser.add_argument('--global_loss_exp_no', type=int, default=0)
# no_of_partitions per volume
parser.add_argument('--n_parts', type=int, default=4)

# type of local_loss_exp_no for Local contrastive loss
# 0 - default loss formulation. Sample local regions from two images. these 2 images are intensity transformed version of same image.
# 1 - (0) + sample local regions to match from 2 differnt images that are from 2 different volumes but they belong to corresponding local regions of similar partitions.
parser.add_argument('--local_loss_exp_no', type=int, default=0)

# segmentation loss used for optimization
# 0 for weighted cross entropy, 1 for dice loss w/o background label, 2 for dice loss with background label (default)
parser.add_argument('--dsc_loss', type=int, default=2)

#random deformations - arguments
#enable random deformations
parser.add_argument('--rd_en', type=int, default=1)
#sigma of gaussian distribution used to sample random deformations 3x3 grid values
parser.add_argument('--sigma', type=float, default=5)
#enable random contrasts
parser.add_argument('--ri_en', type=int, default=1)
#enable 1-hot encoding of the labels 
parser.add_argument('--en_1hot', type=int, default=1)
#controls the ratio of deformed images to normal images used in each mini-batch of the training
parser.add_argument('--rd_ni', type=int, default=1)

# no. of local regions to consider in the feature map for local contrastive loss computation
parser.add_argument('--no_of_local_regions', type=int, default=13)

#no. of decoder blocks used. Here, 1 means 1 decoder block used, 2 is for 2 blocks,..., 5 is for all blocks aka full decoder.
parser.add_argument('--no_of_decoder_blocks', type=int, default=1)

#local_reg_size - 1 for 3x3 local region size in the feature map. <local_reg> -> flat -> w*flat -> 128 bit z vector matching;
#               - 0 for 1x1 local region size in the feature map
parser.add_argument('--local_reg_size', type=int, default=1)
#wgt_en - 1 for having extra weight layer on top of 'z' vector from local region.
#      - 0 for not having any weight layer.
parser.add_argument('--wgt_en', type=int, default=1)


#no. of neighbouring local regions sampled from the feature maps to act as negative samples in local contrastive loss
# for a given positive local region - currently 5 local regions are chosen from each feature map (due to memory issues).
parser.add_argument('--no_of_neg_local_regions', type=int, default=5)

#overide the no. of negative (-ve) local neighbouring regions chosen for local loss computation- 4 for L^{D} (local_loss_exp_no=1) - due to memory issues
parser.add_argument('--no_of_neg_regs_override', type=int, default=4)

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

if(parse_config.rd_en==1):
    parse_config.en_1hot=1
else:
    parse_config.en_1hot=0

struct_name=cfg.struct_name
val_step_update=cfg.val_step_update
######################################
#  load encoder + 'l' decoder blocks pre-trained weights from pre-trained model stage wise with global and local loss respectively.
######################################
#define directory where pre-trained encoder + decoder model was saved
save_dir=str(cfg.srt_dir)+'/models/'+str(parse_config.dataset)+'/trained_models/pretrain_decoder_with_local_contrastive_loss/'

save_dir=str(save_dir)+'/load_encoder_wgts_from_pretrained_model/'

if(parse_config.pretr_data_aug==0):
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

save_dir = str(save_dir) + 'no_of_decoder_blocks_' + str(parse_config.no_of_decoder_blocks) + '/'

save_dir=str(save_dir)+'no_of_local_regions_'+str(parse_config.no_of_local_regions)

if(parse_config.local_loss_exp_no==1):
    parse_config.no_of_neg_regs_override=4
    save_dir=save_dir+'_no_of_neg_regions_'+str(parse_config.no_of_neg_regs_override)+'/'
else:
    save_dir=save_dir+'_no_of_neg_regions_'+str(parse_config.no_of_neg_local_regions)+'/'

save_dir=str(save_dir)+'pretrain_only_decoder_weights/'

save_dir=str(save_dir)+str(parse_config.pretr_no_of_tr_imgs)+'/'+str(parse_config.pretr_comb_tr_imgs)+'_v'+str(parse_config.pretr_ver)+'/enc_bbox_dim_'+str(parse_config.pretr_cont_bbox_dim)+'_n_iter_'+str(parse_config.pretr_n_iter)+'_lr_reg_'+str(parse_config.lr_reg)+'/'

print('save dir ',save_dir)
######################################

######################################
# Define Encoder(e) + 'l' decoder blocks (d_l) + g_2 network graph used for pre-training. We load the pre-trained weights of encoder (e) and 'l' decoder blocks.
tf.reset_default_graph()
ae = model.decoder_pretrain_net(learn_rate_seg=parse_config.lr_reg,temp_fac=parse_config.temp_fac, \
                        no_of_local_regions=parse_config.no_of_local_regions, no_of_decoder_blocks=parse_config.no_of_decoder_blocks, \
                        local_loss_exp_no=parse_config.local_loss_exp_no, local_reg_size=parse_config.local_reg_size, \
                        wgt_en=parse_config.wgt_en, no_of_neg_local_regions=parse_config.no_of_neg_local_regions, \
                        no_of_neg_regs_override=parse_config.no_of_neg_regs_override, inf=1)

######################################
# Restore the model and initialize the encoder with the pre-trained weights from last epoch
######################################
mp_best=get_chkpt_file(save_dir)
print('load last step model from pre-training')
print('mp_best',mp_best)

saver_rnet = tf.train.Saver()
sess_rnet = tf.Session(config=config)
saver_rnet.restore(sess_rnet, mp_best)
print("Model restored")

#get all trainable variable names and their values
print('Loading trainable vars')
cont_variables_names = [v.name for v in tf.trainable_variables()]
#print('var names list',cont_variables_names)
cont_var_values = sess_rnet.run(cont_variables_names)
sess_rnet.close()
print('loaded encoder + l decoder blocks weight values from pre-trained model with global and local contrastive loss')
######################################

######################################
# Define final U-net model & directory to save - for segmentation task
#######################################
#define directory to save fine-tuned model
save_dir=str(cfg.srt_dir)+'/models/'+str(parse_config.dataset)+'/trained_models/fine_tune_on_pretrained_encoder_and_decoder_net/'

if(parse_config.data_aug==0):
    save_dir=str(save_dir)+'/no_data_aug/'
    parse_config.rd_en,parse_config.ri_en=0,0
    parse_config.rd_ni,parse_config.en_1hot=0,0
else:
    save_dir=str(save_dir)+'/with_data_aug/'

if(parse_config.rd_en==1 and parse_config.ri_en==1):
    save_dir=str(save_dir)+'rand_deforms_and_ints_en/'
elif(parse_config.rd_en==1):
    save_dir=str(save_dir)+'rand_deforms_en/'
elif(parse_config.ri_en==1):
    save_dir=str(save_dir)+'rand_ints_en/'

save_dir=str(save_dir)+'global_loss_exp_no_'+str(parse_config.global_loss_exp_no)+'_local_loss_exp_no_'+str(parse_config.local_loss_exp_no) \
             +'_n_parts_'+str(parse_config.n_parts)+'/'

save_dir=str(save_dir)+'temp_fac_'+str(parse_config.temp_fac)+'/'
save_dir=str(save_dir)+'enc_bbox_dim_'+str(parse_config.pretr_cont_bbox_dim)+'/'

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
    parse_config.no_of_neg_regs_override=4
    save_dir=save_dir+'_no_of_neg_regions_'+str(parse_config.no_of_neg_regs_override)+'/'
else:
    save_dir=save_dir+'_no_of_neg_regions_'+str(parse_config.no_of_neg_local_regions)+'/'

save_dir=str(save_dir)+'last_ep_model/'
    
save_dir=str(save_dir)+str(parse_config.no_of_tr_imgs)+'/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+'/unet_dsc_'+str(parse_config.dsc_loss)+'_n_iter_'+str(parse_config.n_iter)+'_lr_seg_'+str(parse_config.lr_seg)+'/'

print('save dir ',save_dir)
######################################

######################################
tf.reset_default_graph()
# Segmentation Network
ae = model.seg_unet(learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,en_1hot=parse_config.en_1hot,mtask_en=0)

# define network/graph to apply random deformations on input images
ae_rd = model.deform_net(batch_size=cfg.mtask_bs)

# define network/graph to apply random contrast and brightness on input images
ae_rc = model.contrast_net(batch_size=cfg.mtask_bs)

# define graph to compute 1-hot encoding of segmentation mask
ae_1hot = model.conv_1hot()
######################################

######################################
# Define checkpoint file to save CNN network architecture and learnt hyperparameters
checkpoint_filename='fine_tune_trained_encoder_and_decoder_net_'+str(parse_config.dataset)
logs_path = str(save_dir)+'tensorflow_logs/'
best_model_dir=str(save_dir)+'best_model/'
pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True)
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
saver = tf.train.Saver(max_to_keep=2)
######################################

######################################
# assign values to all trainable ops of network
assign_op=[]
print('Init of trainable vars')
for new_var in tf.trainable_variables():
    for var, var_val in zip(cont_variables_names, cont_var_values):
        if (str(var) == str(new_var.name) and ('reg_' not in str(new_var.name) and 'seg_' not in str(new_var.name))):
            #print('match name',new_var.name,var)
            tmp_op=new_var.assign(var_val)
            assign_op.append(tmp_op)

sess.run(assign_op)
print('init done for all the encoder + l decoder blocks weights and biases from pre-trained model')
#######################################

######################################
# Load training and validation images & labels
######################################
#load training volumes id numbers to train the unet
train_list = data_list.train_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
#load saved training data in cropped dimensions directly
print('load train volumes')
train_imgs, train_labels = dt.load_cropped_img_labels(train_list)
#print('train shape',train_imgs.shape,train_labels.shape)

#load validation volumes id numbers to save the best model during training
val_list = data_list.val_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
#load val data both in original dimensions and its cropped dimensions
print('load val volumes')
val_label_orig,val_img_crop,val_label_crop,pixel_val_list=load_val_imgs(val_list,dt,orig_img_dt)

# get test volumes id list
print('get test volumes list')
test_list = data_list.test_data()
######################################

######################################
# parameters values set for training of CNN
mean_f1_val_prev=0.0000001
threshold_f1=0.0000001
step_val=parse_config.n_iter
start_epoch=0
n_epochs=step_val

tr_loss_list,val_loss_list=[],[]
tr_dsc_list,val_dsc_list=[],[]
ep_no_list=[]
loss_least_val=1
f1_mean_least_val=0.0000000001
######################################

######################################
# Loop over all the epochs to train the CNN.
# Randomly sample a batch of images from all data per epoch. On the chosen batch, apply random augmentations and optimize the network.
for epoch_i in range(start_epoch,n_epochs):
    
    # Sample shuffled img, GT labels -- from labeled data
    ld_img_batch,ld_label_batch=shuffle_minibatch_mtask([train_imgs,train_labels],batch_size=cfg.mtask_bs)
    if(parse_config.data_aug==1):
        # Apply affine transformations
        ld_img_batch,ld_label_batch=augmentation_function([ld_img_batch,ld_label_batch],dt)
    if(parse_config.rd_en==1 or parse_config.ri_en==1):
        # Apply random augmentations - random deformations + random contrast & brightness values
        ld_img_batch,ld_label_batch=create_rand_augs(cfg,parse_config,sess,ae_rd,ae_rc,ld_img_batch,ld_label_batch)

    #Run optimizer update on the training data (labeled data)
    train_summary,loss,_=sess.run([ae['train_summary'],ae['seg_cost'],ae['optimizer_unet_all']],\
                                      feed_dict={ae['x']:ld_img_batch,ae['y_l']:ld_label_batch,ae['train_phase']:True})
    
    if(epoch_i%val_step_update==0):
        train_writer.add_summary(train_summary, epoch_i)
        train_writer.flush()

    if(epoch_i%cfg.val_step_update==0):
        # Measure validation volumes accuracy in Dice score (DSC) and evaluate validation loss
        # Save the model with the best DSC over validation volumes.
        mean_f1_val_prev,mp_best,mean_total_cost_val,mean_f1=f1_util.track_val_dsc(sess,ae,ae_1hot,saver,mean_f1_val_prev,threshold_f1,\
                                    best_model_dir,val_list,val_img_crop,val_label_crop,val_label_orig,pixel_val_list,\
                                    checkpoint_filename,epoch_i,en_1hot_val=parse_config.en_1hot)

        tr_y_pred=sess.run(ae['y_pred'],feed_dict={ae['x']:ld_img_batch,ae['y_l']:ld_label_batch,ae['train_phase']:False})

        if(parse_config.en_1hot==1):
            tr_accu=f1_util.calc_f1_score(np.argmax(tr_y_pred,axis=-1),np.argmax(ld_label_batch,-1))
        else:
            tr_accu=f1_util.calc_f1_score(np.argmax(tr_y_pred,axis=-1),ld_label_batch)
        tr_dsc_list.append(np.mean(tr_accu))
        val_dsc_list.append(mean_f1)
        tr_loss_list.append(np.mean(loss))
        val_loss_list.append(np.mean(mean_total_cost_val))

        print('epoch_i,loss,f1_val',epoch_i,np.mean(loss),mean_f1_val_prev,mean_f1)

        #Compute and save validation images dice & loss summary
        val_summary_msg = sess.run(ae['val_summary'], feed_dict={ae['mean_dice']: mean_f1, ae['val_totalc']:mean_total_cost_val})
        val_sum_writer.add_summary(val_summary_msg, epoch_i)
        val_sum_writer.flush()

        if(np.mean(mean_f1_val_prev)>f1_mean_least_val):
            f1_mean_least_val=mean_f1_val_prev
            ep_no_list.append(epoch_i)
        

    if ((epoch_i==n_epochs-1)):
        # model saved at the last epoch of training
        mp = str(save_dir) + str(checkpoint_filename) + '_epochs_' + str(epoch_i) + ".ckpt"
        saver.save(sess, mp)
        try:
            mp_best
        except NameError:
            mp_best=mp
            

######################################
# Plot the training, validation (val) loss and DSC score of training & val images over all the epochs of training.
f1_util.plt_seg_loss([tr_loss_list,val_loss_list],save_dir,title_str='ft_local_loss_model',plt_name='tr_seg_loss',ep_no=epoch_i)
f1_util.plt_seg_loss([tr_dsc_list,val_dsc_list],save_dir,title_str='ft_local_loss_model',plt_name='tr_dsc_score',ep_no=epoch_i)

sess.close()
######################################

######################################
# find best model checkpoint over all epochs and restore it
mp_best=get_max_chkpt_file(save_dir)
print('mp_best',mp_best)

saver = tf.train.Saver()
sess = tf.Session(config=config)
saver.restore(sess, mp_best)
print("Model restored")
#####################################

# infer predictions over test volumes from the best model saved during training
save_dir_tmp=save_dir+'/test_set_predictions/'
f1_util.test_set_predictions(test_list,sess,ae,dt,orig_img_dt,save_dir_tmp)

sess.close()
tf.reset_default_graph()
######################################
