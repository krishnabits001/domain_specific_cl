
cd cloned_code_directory

source activate tensorflow_env

cd train_model/

#Step 1
echo Step 1: start pre-training of Encoder

python pretr_encoder_global_contrastive_loss.py --dataset=acdc --no_of_tr_imgs=tr52 --global_loss_exp_no=2 --n_parts=4 --temp_fac=0.1 --bt_size=12

echo end of pre-training of Encoder

#Step 2
echo Step 2: start pre-training of Decoder

python pretr_decoder_local_contrastive_loss.py --dataset=acdc --no_of_tr_imgs=tr52 --pretr_no_of_tr_imgs=tr52 --local_reg_size=1 --no_of_local_regions=13 --temp_fac=0.1 --global_loss_exp_no=2 --local_loss_exp_no=0 --no_of_decoder_blocks=3 --no_of_neg_local_regions=5 --bt_size=12

echo end of pre-training of Decoder

#Step 3
echo Step 3: start fine-tuning with initialization from both Encoder and Decoder weights learned from pre-training 

python ft_pretr_encoder_decoder_net_local_loss.py --dataset=acdc --pretr_no_of_tr_imgs=tr52 --local_reg_size=1 --no_of_local_regions=13 --temp_fac=0.1 --global_loss_exp_no=2 --local_loss_exp_no=0 --no_of_decoder_blocks=3 --no_of_neg_local_regions=5 --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --ver=0 

echo end of fine-tuning

#Optional Step - Fine-tuning with only pre-trained Encoder weights
#echo start fine-tuning with initialization from only Encoder weights learned from pre-training 
#
#python ft_pretr_encoder_net_global_loss.py --dataset=acdc --data_aug=1 --pretr_no_of_tr_imgs=tr52 --ver=0 --global_loss_exp_no=2 --n_parts=4 --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --temp_fac=0.1 --bbox_dim=100  --bt_size=12
#
#echo end of fine-tuning
