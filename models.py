import tensorflow as tf
import numpy as np

# Load layers and losses
from layers_bn import layersObj
layers = layersObj()

from losses import lossObj
loss = lossObj()

class modelObj:
    def __init__(self,cfg,override_num_classes=0):
        self.img_size_x=cfg.img_size_x
        self.img_size_y=cfg.img_size_y
        self.num_classes=cfg.num_classes
        self.num_channels=cfg.num_channels

        self.interp_val = cfg.interp_val
        self.img_size_flat=cfg.img_size_flat
        self.batch_size=cfg.batch_size_ft

        self.mtask_bs=cfg.mtask_bs

        if(override_num_classes==1):
            self.num_classes=2

    def conv_1hot(self):
        # To compute the 1-hot encoding of input mask to number of classes
        # placeholders for the network
        y_tmp = tf.placeholder(tf.int32, shape=[None, self.img_size_x, self.img_size_y], name='y_tmp')

        y_tmp_1hot = tf.one_hot(y_tmp,depth=self.num_classes)
        return {'y_tmp':y_tmp,'y_tmp_1hot':y_tmp_1hot}

    def deform_net(self,batch_size):
        # To apply random deformations on the input image and segmentation mask

        # placeholders for the network
        x_tmp = tf.placeholder(tf.float32, shape=[batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')
        v_tmp = tf.placeholder(tf.float32, shape=[batch_size, self.img_size_x, self.img_size_y, 2], name='v_tmp')
        y_tmp = tf.placeholder(tf.int32, shape=[batch_size, self.img_size_x, self.img_size_y], name='y_tmp')

        y_tmp_1hot = tf.one_hot(y_tmp,depth=self.num_classes)
        w_tmp = tf.contrib.image.dense_image_warp(image=x_tmp,flow=v_tmp,name='dense_image_warp_tmp')
        w_tmp_1hot = tf.contrib.image.dense_image_warp(image=y_tmp_1hot,flow=v_tmp,name='dense_image_warp_tmp_1hot')

        return {'x_tmp':x_tmp,'flow_v':v_tmp,'deform_x':w_tmp,'y_tmp':y_tmp,'y_tmp_1hot':y_tmp_1hot,'deform_y_1hot':w_tmp_1hot}

    def contrast_net(self,batch_size):
        # To apply random contrast and brightness (random intensity transformations) on the input image (Fine-training stage)

        # placeholders for the network
        x_tmp = tf.placeholder(tf.float32, shape=[batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')

        rd_cont = tf.image.random_contrast(x_tmp,lower=0.8,upper=1.2,seed=1)
        rd_brit = tf.image.random_brightness(x_tmp,max_delta=0.1,seed=1)
        c_ind=np.arange(0,int(batch_size/2),dtype=np.int32)
        b_ind=np.arange(int(batch_size/2),int(batch_size),dtype=np.int32)

        rd_fin = tf.concat((tf.gather(rd_cont,c_ind),tf.gather(rd_brit,b_ind)),axis=0)
        return {'x_tmp':x_tmp,'rd_fin':rd_fin,'rd_cont':rd_cont,'rd_brit':rd_brit}

    def brit_cont_net(self,batch_size):
        # To apply random contrast and brightness (random intensity transformations) on the input image (Pre-training stages)

        # placeholders for the network
        x_tmp = tf.placeholder(tf.float32, shape=[batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')

        # brightness + contrast changes final image
        rd_brit = tf.image.random_brightness(x_tmp,max_delta=0.3,seed=1)
        rd_cont = tf.image.random_contrast(rd_brit,lower=0.7,upper=1.3,seed=1)
        rd_fin=tf.clip_by_value(rd_cont,0,1.5)

        return {'x_tmp':x_tmp,'rd_fin':rd_fin,'rd_cont':rd_cont,'rd_brit':rd_brit}


    def cos_sim(self,vec_a,vec_b,temp_fac):
        # To compute the cosine similarity score of the input 2 vectors scaled by temparature factor

        norm_vec_a = tf.nn.l2_normalize(vec_a,axis=-1)
        norm_vec_b = tf.nn.l2_normalize(vec_b,axis=-1)
        #cos_sim_val=tf.multiply(norm_vec_a,norm_vec_b)/scale_fac
        cos_sim_val=tf.linalg.matmul(norm_vec_a,norm_vec_b,transpose_b=True)/temp_fac
        return cos_sim_val

    def encoder_network(self,x,train_phase,no_filters,encoder_list_return=0):
        # Define the Encoder Network

        #layers list for skip connections
        enc_layers_list=[]
        ############################################
        # U-Net like Network
        ############################################
        # Encoder - Downsampling Path
        ############################################
        # 2x 3x3 conv and 1 maxpool
        # Level 1
        enc_c1_a = layers.conv2d_layer(ip_layer=x, name='enc_c1_a', num_filters=no_filters[1], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c1_b = layers.conv2d_layer(ip_layer=enc_c1_a, name='enc_c1_b', num_filters=no_filters[1], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c1_pool = layers.max_pool_layer2d(enc_c1_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c1_pool')
        enc_layers_list.append(enc_c1_b)

        # Level 2
        enc_c2_a = layers.conv2d_layer(ip_layer=enc_c1_pool, name='enc_c2_a', num_filters=no_filters[2], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c2_b = layers.conv2d_layer(ip_layer=enc_c2_a, name='enc_c2_b', num_filters=no_filters[2], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c2_pool = layers.max_pool_layer2d(enc_c2_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c2_pool')
        enc_layers_list.append(enc_c2_b)

        # Level 3
        enc_c3_a = layers.conv2d_layer(ip_layer=enc_c2_pool, name='enc_c3_a', num_filters=no_filters[3], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c3_b = layers.conv2d_layer(ip_layer=enc_c3_a, name='enc_c3_b', num_filters=no_filters[3], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c3_pool = layers.max_pool_layer2d(enc_c3_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c3_pool')
        enc_layers_list.append(enc_c3_b)

        # Level 4
        enc_c4_a = layers.conv2d_layer(ip_layer=enc_c3_pool, name='enc_c4_a', num_filters=no_filters[4], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c4_b = layers.conv2d_layer(ip_layer=enc_c4_a, name='enc_c4_b', num_filters=no_filters[4], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c4_pool = layers.max_pool_layer2d(enc_c4_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c4_pool')
        enc_layers_list.append(enc_c4_b)

        # Level 5 - 2x Conv
        enc_c5_a = layers.conv2d_layer(ip_layer=enc_c4_pool, name='enc_c5_a', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c5_b = layers.conv2d_layer(ip_layer=enc_c5_a, name='enc_c5_b', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c5_pool = layers.max_pool_layer2d(enc_c5_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c5_pool')
        enc_layers_list.append(enc_c5_b)

        # Level 6 - 2x Conv
        enc_c6_a = layers.conv2d_layer(ip_layer=enc_c5_pool, name='enc_c6_a', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c6_b = layers.conv2d_layer(ip_layer=enc_c6_a, name='enc_c6_b', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)

        if(encoder_list_return==1):
            return enc_c6_b,enc_layers_list
        else:
            return enc_c6_b

    def encoder_pretrain_net(self,learn_rate_seg=0.001,temp_fac=0.1,global_loss_exp_no=1,n_parts=4):
        # Define the Encoder Network with g_1 a small MLP network to pre-train the encoder

        # No of channels in each layer
        no_filters = [1, 16, 32, 64, 128, 128]
        num_channels=self.num_channels

        ###################################
        # placeholders for the network Inputs
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x')
        train_phase = tf.placeholder(tf.bool, name='train_phase')

        ###################################
        # Last layer from Encoder network (e)
        enc_c6_b = self.encoder_network(x, train_phase, no_filters,encoder_list_return=0)

        ###################################
        # Architecture of small network (g_1) on top of encoder (e) to match the representations
        # flat -> 3200 -> 1024 -> 128
        reg_flat = tf.layers.flatten(inputs=enc_c6_b)

        reg_NN_1 = tf.layers.dense(inputs=reg_flat,units=1024, name='reg_nn1', activation=tf.nn.relu, use_bias=False)
        reg_pred = tf.layers.dense(inputs=reg_NN_1, units=128, name='reg_pred', activation=None, use_bias=False)
        ###################################

        net_global_loss=0

        # if(global_loss_exp_no==0):
        #     ######################
        #     # G^{R} - Like in simCLR [12]
        #     ######################
        #     bs=2*self.batch_size
        #     #loop over each pair of positive images in the batch to calculate the Net global contrastive loss over the whole batch.
        #     for pos_index in range(0,bs,2):
        #         #indexes of positive pair of samples (x_1,x_2)
        #         num_i1=np.arange(pos_index,pos_index+1,dtype=np.int32)
        #         num_i2=np.arange(pos_index+1,pos_index+2,dtype=np.int32)
        #
        #         #indexes of corresponding negative samples as per positive pair of samples (x_1,x_2)
        #         den_index_i1=np.arange(0,bs,dtype=np.int32)
        #         den_index_i1 = np.delete(den_index_i1, pos_index)
        #         den_index_i2=np.arange(0,bs,dtype=np.int32)
        #         den_index_i2 = np.delete(den_index_i2, pos_index+1)
        #
        #         # gather required positive samples (x_1,x_2) for the numerator term
        #         x_num_i1=tf.gather(reg_pred,num_i1)
        #         x_num_i2=tf.gather(reg_pred,num_i2)
        #         # gather required corresponding negative samples for the denominator term
        #         x_den_i1=tf.gather(reg_pred,den_index_i1)
        #         x_den_i2=tf.gather(reg_pred,den_index_i2)
        #         #print('a1',x_num_i1,x_den_i1,x_num_i2,x_den_i2)
        #
        #         #calculate cosine similarity score as in simCLR[12] + global contrastive loss for the pair of positive images (x_1,x_2)
        #         # loss for positive image x_1 (num_i1_loss)
        #         # numerator of loss term (num_i1_ss), & denominator of loss term (den_i1_ss)
        #         num_i1_ss=self.cos_sim(x_num_i1,x_num_i2,temp_fac)
        #         den_i1_ss=self.cos_sim(x_num_i1,x_den_i1,temp_fac)
        #         num_i1_loss=-tf.log(tf.exp(num_i1_ss)/tf.math.reduce_sum(tf.exp(den_i1_ss)))
        #         net_global_loss = net_global_loss + num_i1_loss
        #         #print('a2',num_i1_ss,den_i1_ss,num_i1_loss)
        #
        #         # loss for positive image x_2 (num_i2_loss)
        #         # numerator of loss term (num_i2_ss), & denominator of loss term (den_i2_ss)
        #         num_i2_ss=self.cos_sim(x_num_i2,x_num_i1,temp_fac)
        #         den_i2_ss=self.cos_sim(x_num_i2,x_den_i2,temp_fac)
        #         num_i2_loss=-tf.log(tf.exp(num_i2_ss)/tf.math.reduce_sum(tf.exp(den_i2_ss)))
        #         net_global_loss = net_global_loss + num_i2_loss

        if(global_loss_exp_no==1):
            ######################
            # G^{D-} - Proposed variant
            # We split each volume into n_parts and select 1 image from each n_part of the volume
            # We select the negative samples that we want to contrast against for a given positive image.
            # Example: if positive image is from partition 1 of volume 1, then NO negative sample are taken from partition 1 of any other volume (including volume 1).
            ######################
            bs=3*self.batch_size
            # loop over each pair of positive images in the batch to calculate the Net global contrastive loss over the whole batch.
            for pos_index in range(0,self.batch_size,1):
                #indexes of positive pair of samples (x_1,x_2,x_3) - we can make 3 pairs: (x_1,x_2), (x_1,x_3), (x_2,x_3)
                num_i1=np.arange(pos_index,pos_index+1,dtype=np.int32)
                j=self.batch_size+pos_index
                num_i2=np.arange(j,j+1,dtype=np.int32)
                j=2*self.batch_size+pos_index
                num_i3=np.arange(j,j+1,dtype=np.int32)
                #print('n1,n2,n3',num_i1,num_i2,num_i3)

                # indexes of corresponding negative samples as per positive pair of samples: (x_1,x_2), (x_1,x_3), (x_2,x_3)
                den_index_net=np.arange(0,bs,dtype=np.int32)

                # Pruning the negative samples
                # Deleting the indexes of the samples in the batch used as negative samples for a given positive image. These indexes belong to identical partitions in other volumes in the batch.
                # Example: if positive image is from partition 1 of volume 1, then NO negative sample are taken from partition 1 of any other volume (including volume 1) in the batch
                ind_l=[]
                rem = int(num_i1) % n_parts
                for not_neg_index in range(rem, bs, 4):
                    ind_l.append(not_neg_index)

                #print('ind_l',ind_l)
                den_indexes = np.delete(den_index_net, ind_l)
                #print('d1',den_i1,len(den_i1))

                # gather required positive samples x_1,x_2,x_3 for the numerator term
                x_num_i1=tf.gather(reg_pred,num_i1)
                x_num_i2=tf.gather(reg_pred,num_i2)
                x_num_i3=tf.gather(reg_pred,num_i3)

                # gather required negative samples x_1,x_2,x_3 for the denominator term
                x_den=tf.gather(reg_pred,den_indexes)

                # calculate cosine similarity score + global contrastive loss for each pair of positive images

                #for positive pair (x_1,x_2);
                # numerator of loss term (num_i1_i2_ss) & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
                num_i1_i2_ss=self.cos_sim(x_num_i1,x_num_i2,temp_fac)
                den_i1_i2_ss=self.cos_sim(x_num_i1,x_den,temp_fac)
                num_i1_i2_loss=-tf.log(tf.exp(num_i1_i2_ss)/(tf.exp(num_i1_i2_ss)+tf.math.reduce_sum(tf.exp(den_i1_i2_ss))))
                net_global_loss = net_global_loss + num_i1_i2_loss
                # for positive pair (x_2,x_1);
                # numerator same & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
                den_i2_i1_ss=self.cos_sim(x_num_i2,x_den,temp_fac)
                num_i2_i1_loss=-tf.log(tf.exp(num_i1_i2_ss)/(tf.exp(num_i1_i2_ss)+tf.math.reduce_sum(tf.exp(den_i2_i1_ss))))
                net_global_loss = net_global_loss + num_i2_i1_loss

                # for positive pair (x_1,x_3);
                # numerator of loss term (num_i1_i3_ss) & denominator of loss term (den_i1_i3_ss) & loss (num_i1_i3_loss)
                num_i1_i3_ss=self.cos_sim(x_num_i1,x_num_i3,temp_fac)
                den_i1_i3_ss=self.cos_sim(x_num_i1,x_den,temp_fac)
                num_i1_i3_loss=-tf.log(tf.exp(num_i1_i3_ss)/(tf.exp(num_i1_i3_ss)+tf.math.reduce_sum(tf.exp(den_i1_i3_ss))))
                net_global_loss = net_global_loss + num_i1_i3_loss
                # for positive pair (x_3,x_1);
                # numerator same & denominator of loss term (den_i3_i1_ss) & loss (num_i3_i1_loss)
                den_i3_i1_ss=self.cos_sim(x_num_i3,x_den,temp_fac)
                num_i3_i1_loss=-tf.log(tf.exp(num_i1_i3_ss)/(tf.exp(num_i1_i3_ss)+tf.math.reduce_sum(tf.exp(den_i3_i1_ss))))
                net_global_loss = net_global_loss + num_i3_i1_loss

                # for positive pair (x_2,x_3);
                # numerator of loss term (num_i2_i3_ss) & denominator of loss term (den_i2_i3_ss) & loss (num_i2_i3_loss)
                num_i2_i3_ss=self.cos_sim(x_num_i2,x_num_i3,temp_fac)
                den_i2_i3_ss=self.cos_sim(x_num_i2,x_den,temp_fac)
                num_i2_i3_loss=-tf.log(tf.exp(num_i2_i3_ss)/(tf.exp(num_i2_i3_ss)+tf.math.reduce_sum(tf.exp(den_i2_i3_ss))))
                net_global_loss = net_global_loss + num_i2_i3_loss
                # for positive pair (x_3,x_2):
                # numerator same & denominator of loss term (den_i3_i2_ss) & loss (num_i3_i2_loss)
                den_i3_i2_ss=self.cos_sim(x_num_i3,x_den,temp_fac)
                num_i3_i2_loss=-tf.log(tf.exp(num_i2_i3_ss)/(tf.exp(num_i2_i3_ss)+tf.math.reduce_sum(tf.exp(den_i3_i2_ss))))
                net_global_loss = net_global_loss + num_i3_i2_loss

        elif(global_loss_exp_no==2):
            ######################
            # G^{D} - Proposed variant
            # We split each volume into n_parts and select 1 image from each n_part of the volume
            # the Negative image selection is done as in G^{D-} (global_loss_exp_no=1)
            # Additionally, we match images across volumes belonging to identical partition numbers of the volumes along with matching the positive image with its augmented version.
            # Example: if positive image (x_i1) is from partition 1 of volume 1, then the paired positive image (x_j1) to match is taken from partition 1 of any other volume (excluding volume 1).
            ######################
            if(n_parts==4):
                bs=4*self.batch_size
                if(self.batch_size!=12):
                    factor=10*n_parts
                else:
                    factor=n_parts
            elif(n_parts==3):
                bs=4*self.batch_size+5
                factor=n_parts+2
            elif(n_parts==6):
                bs=5*self.batch_size+4
                factor=2

            # loop over each pair of positive images in the batch to calculate the Net global contrastive loss over the whole batch.
            for pos_index in range(0,bs,1):

                # indexes of positive pair of samples (x_i1,x_a_i1, x_j1,x_a_j1) - we can make 4 pairs: (x_i1,x_a_i1), (x_i1,x_j1), (x_j1,x_a_j1), (x_a_i1,x_a_j1)
                # x_a_i1, x_a_j1 are augmented versions of x_i1 and x_j1, respectively.
                num_i1=np.arange(pos_index,pos_index+1,dtype=np.int32)
                if(pos_index+n_parts>=bs):
                    j=(pos_index+n_parts)%bs
                    num_i2=np.arange(j,j+1,dtype=np.int32)
                else:
                    num_i2=np.arange(pos_index+n_parts,pos_index+n_parts+1,dtype=np.int32)
                if(pos_index+2*n_parts>=bs):
                    j=(pos_index+2*n_parts)%bs
                    num_i3=np.arange(j,j+1,dtype=np.int32)
                else:
                    num_i3=np.arange(pos_index+2*n_parts,pos_index+2*n_parts+1,dtype=np.int32)
                if(pos_index+3*n_parts>=bs):
                    j=(pos_index+3*n_parts)%bs
                    num_i4=np.arange(j,j+1,dtype=np.int32)
                else:
                    num_i4=np.arange(pos_index+3*n_parts,pos_index+3*n_parts+1,dtype=np.int32)

                #print('n1,n2,n3,n4',num_i1,num_i2,num_i3,num_i4)

                # indexes of corresponding negative samples as per positive pair of samples.
                den_index_net=np.arange(0,bs,dtype=np.int32)

                ind_l=[]
                for not_neg_index in range(0,factor*n_parts):
                    if(num_i1+not_neg_index*n_parts>=bs):
                        j=(num_i1+not_neg_index*n_parts)%bs
                        #print('j1',j)
                        ind_l.append(j)
                    else:
                        #print('j0',num_i1+k*n_parts)
                        ind_l.append(num_i1+not_neg_index*n_parts)
                #print('ind_l',ind_l)
                den_indexes = np.delete(den_index_net, ind_l)
                #print('d1',den_i1,len(den_i1))

                # gather required positive samples x_1,x_2,x_3,x_4 for the numerator term
                x_num_i1=tf.gather(reg_pred,num_i1)
                x_num_i2=tf.gather(reg_pred,num_i2)
                x_num_i3=tf.gather(reg_pred,num_i3)
                x_num_i4=tf.gather(reg_pred,num_i4)

                # gather required negative samples x_1,x_2,x_3 for the denominator term
                x_den = tf.gather(reg_pred, den_indexes)

                # calculate cosine similarity score + global contrastive loss for each pair of positive images
                #if(i%8<4):
                if(pos_index%(2*n_parts)<n_parts):
                    # for positive pair (x_i1, x_a_i1): (i1,i2)
                    # numerator of loss term (num_i1_i2_ss) & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
                    num_i1_i2_ss=self.cos_sim(x_num_i1,x_num_i2,temp_fac)
                    den_i1_i2_ss=self.cos_sim(x_num_i1,x_den,temp_fac)
                    num_i1_i2_loss=-tf.log(tf.exp(num_i1_i2_ss)/(tf.exp(num_i1_i2_ss)+tf.math.reduce_sum(tf.exp(den_i1_i2_ss))))
                    net_global_loss = net_global_loss + num_i1_i2_loss
                    # for positive pair (x_a_i1,x_i1);
                    # numerator same & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
                    den_i2_i1_ss=self.cos_sim(x_num_i2,x_den,temp_fac)
                    num_i2_i1_loss=-tf.log(tf.exp(num_i1_i2_ss)/(tf.exp(num_i1_i2_ss)+tf.math.reduce_sum(tf.exp(den_i2_i1_ss))))
                    net_global_loss = net_global_loss + num_i2_i1_loss

                    # for positive pair (x_i1, x_j1): (i1,i3)
                    # numerator of loss term (num_i1_i3_ss) & denominator of loss term (den_i1_i3_ss) & loss (num_i1_i3_loss)
                    num_i1_i3_ss=self.cos_sim(x_num_i1,x_num_i3,temp_fac)
                    den_i1_i3_ss=self.cos_sim(x_num_i1,x_den,temp_fac)
                    num_i1_i3_loss=-tf.log(tf.exp(num_i1_i3_ss)/(tf.exp(num_i1_i3_ss)+tf.math.reduce_sum(tf.exp(den_i1_i3_ss))))
                    net_global_loss = net_global_loss + num_i1_i3_loss
                    # for positive pair (x_j1, x_i1);
                    # numerator same & denominator of loss term (den_i3_i1_ss) & loss (num_i3_i1_loss)
                    den_i3_i1_ss=self.cos_sim(x_num_i3,x_den,temp_fac)
                    num_i3_i1_loss=-tf.log(tf.exp(num_i1_i3_ss)/(tf.exp(num_i1_i3_ss)+tf.math.reduce_sum(tf.exp(den_i3_i1_ss))))
                    net_global_loss = net_global_loss + num_i3_i1_loss

                    # for positive pair (x_j1, x_a_j1): (i3,i4)
                    # numerator of loss term (num_i2_i3_ss) & denominator of loss term (den_i2_i3_ss) & loss (num_i2_i3_loss)
                    num_i3_i4_ss=self.cos_sim(x_num_i3,x_num_i4,temp_fac)
                    den_i3_i4_ss=self.cos_sim(x_num_i3,x_den,temp_fac)
                    num_i3_i4_loss=-tf.log(tf.exp(num_i3_i4_ss)/(tf.exp(num_i3_i4_ss)+tf.math.reduce_sum(tf.exp(den_i3_i4_ss))))
                    net_global_loss = net_global_loss + num_i3_i4_loss
                    # for positive pair (x_a_j1, x_j1)
                    # numerator same & denominator of loss term (den_i3_i2_ss) & loss (num_i3_i2_loss)
                    den_i4_i3_ss=self.cos_sim(x_num_i4,x_den,temp_fac)
                    num_i4_i3_loss=-tf.log(tf.exp(num_i3_i4_ss)/(tf.exp(num_i3_i4_ss)+tf.math.reduce_sum(tf.exp(den_i4_i3_ss))))
                    net_global_loss = net_global_loss + num_i4_i3_loss

                    # for positive pair (x_a_i1, x_a_j1): (i2,i4)
                    # numerator of loss term (num_i2_i4_ss) & denominator of loss term (den_i2_i4_ss) & loss (num_i2_i4_loss)
                    num_i2_i4_ss=self.cos_sim(x_num_i2, x_num_i4, temp_fac)
                    den_i2_i4_ss=self.cos_sim(x_num_i2, x_den, temp_fac)
                    num_i2_i4_loss=-tf.log(tf.exp(num_i2_i4_ss)/(tf.exp(num_i2_i4_ss)+tf.math.reduce_sum(tf.exp(den_i2_i4_ss))))
                    net_global_loss = net_global_loss + num_i2_i4_loss
                    # for positive pair (x_a_j1, x_a_i1)
                    # numerator same & denominator of loss term (den_i4_i2_ss) & loss (num_i4_i2_loss)
                    den_i4_i2_ss=self.cos_sim(x_num_i4, x_den, temp_fac)
                    num_i4_i2_loss=-tf.log(tf.exp(num_i2_i4_ss)/(tf.exp(num_i2_i4_ss)+tf.math.reduce_sum(tf.exp(den_i4_i2_ss))))
                    net_global_loss = net_global_loss + num_i4_i2_loss
        elif(global_loss_exp_no==4):
            ######################
            # G^{D} - Proposed variant
            # We split each volume into n_parts and select 1 image from each n_part of the volume
            # the Negative image selection is done as in G^{D-} (global_loss_exp_no=1)
            # Additionally, we match images across volumes belonging to identical partition numbers of the volumes along with matching the positive image with its augmented version.
            # Example: if positive image (x_i1) is from partition 1 of volume 1, then the paired positive image (x_j1) to match is taken from partition 1 of any other volume (excluding volume 1).
            ######################
            if(n_parts==4):
                bs=4*self.batch_size
                if(self.batch_size!=12):
                    factor=10*n_parts
                else:
                    factor=n_parts
            elif(n_parts==3):
                bs=4*self.batch_size+5
                factor=n_parts+2
            elif(n_parts==6):
                bs=5*self.batch_size+4
                factor=2

            # loop over each pair of positive images in the batch to calculate the Net global contrastive loss over the whole batch.
            for pos_index in range(0,bs,1):

                # indexes of positive pair of samples (x_i1,x_a_i1, x_j1,x_a_j1) - we can make 4 pairs: (x_i1,x_a_i1), (x_i1,x_j1), (x_j1,x_a_j1), (x_a_i1,x_a_j1)
                # x_a_i1, x_a_j1 are augmented versions of x_i1 and x_j1, respectively.
                num_i1=np.arange(pos_index,pos_index+1,dtype=np.int32)
                if(pos_index+n_parts>=bs):
                    j=(pos_index+n_parts)%bs
                    num_i2=np.arange(j,j+1,dtype=np.int32)
                else:
                    num_i2=np.arange(pos_index+n_parts,pos_index+n_parts+1,dtype=np.int32)
                if(pos_index+2*n_parts>=bs):
                    j=(pos_index+2*n_parts)%bs
                    num_i3=np.arange(j,j+1,dtype=np.int32)
                else:
                    num_i3=np.arange(pos_index+2*n_parts,pos_index+2*n_parts+1,dtype=np.int32)
                if(pos_index+3*n_parts>=bs):
                    j=(pos_index+3*n_parts)%bs
                    num_i4=np.arange(j,j+1,dtype=np.int32)
                else:
                    num_i4=np.arange(pos_index+3*n_parts,pos_index+3*n_parts+1,dtype=np.int32)

                if(pos_index+4*n_parts>=bs):
                    j=(pos_index+4*n_parts)%bs
                    num_i5=np.arange(j,j+1,dtype=np.int32)
                else:
                    num_i5=np.arange(pos_index+4*n_parts,pos_index+4*n_parts+1,dtype=np.int32)
            
                if(pos_index+5*n_parts>=bs):
                    j=(pos_index+5*n_parts)%bs
                    num_i6=np.arange(j,j+1,dtype=np.int32)
                else:
                    num_i6=np.arange(pos_index+5*n_parts,pos_index+5*n_parts+1,dtype=np.int32)
                
                #print('n1,n2,n3,n4',num_i1,num_i2,num_i3,num_i4,num_i5,num_i6)

                # indexes of corresponding negative samples as per positive pair of samples.
                den_index_net=np.arange(0,bs,dtype=np.int32)

                ind_l=[]
                for not_neg_index in range(0,factor*n_parts):
                    if(num_i1+not_neg_index*n_parts>=bs):
                        j=(num_i1+not_neg_index*n_parts)%bs
                        #print('j1',j)
                        ind_l.append(j)
                    else:
                        #print('j0',num_i1+k*n_parts)
                        ind_l.append(num_i1+not_neg_index*n_parts)
                #print('ind_l',ind_l)
                den_indexes = np.delete(den_index_net, ind_l)
                #print('d1',den_i1,len(den_i1))

                # gather required positive samples x_1,x_2,x_3,x_4 for the numerator term
                x_num_i1=tf.gather(reg_pred,num_i1)
                x_num_i2=tf.gather(reg_pred,num_i2)
                x_num_i3=tf.gather(reg_pred,num_i3)
                x_num_i4=tf.gather(reg_pred,num_i4)
                x_num_i5=tf.gather(reg_pred,num_i5)
                x_num_i6=tf.gather(reg_pred,num_i6)

                # gather required negative samples x_1,x_2,x_3 for the denominator term
                x_den = tf.gather(reg_pred, den_indexes)

                # calculate cosine similarity score + global contrastive loss for each pair of positive images
                #if(i%8<4):
                if(pos_index%(3*n_parts)<n_parts):
                    # for positive pair (x_i1, x_a_i1): (i1,i2)
                    # numerator of loss term (num_i1_i2_ss) & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
                    num_i1_i2_ss=self.cos_sim(x_num_i1,x_num_i2,temp_fac)
                    den_i1_i2_ss=self.cos_sim(x_num_i1,x_den,temp_fac)
                    num_i1_i2_loss=-tf.log(tf.exp(num_i1_i2_ss)/(tf.exp(num_i1_i2_ss)+tf.math.reduce_sum(tf.exp(den_i1_i2_ss))))
                    net_global_loss = net_global_loss + num_i1_i2_loss
                    # for positive pair (x_a_i1,x_i1);
                    # numerator same & denominator of loss term (den_i1_i2_ss) & loss (num_i1_i2_loss)
                    den_i2_i1_ss=self.cos_sim(x_num_i2,x_den,temp_fac)
                    num_i2_i1_loss=-tf.log(tf.exp(num_i1_i2_ss)/(tf.exp(num_i1_i2_ss)+tf.math.reduce_sum(tf.exp(den_i2_i1_ss))))
                    net_global_loss = net_global_loss + num_i2_i1_loss

                    # for positive pair (x_a_i1, x_a_i2): (i2,i3)
                    # numerator of loss term (num_i1_i3_ss) & denominator of loss term (den_i1_i3_ss) & loss (num_i1_i3_loss)
                    num_i2_i3_ss=self.cos_sim(x_num_i2,x_num_i3,temp_fac)
                    den_i2_i3_ss=self.cos_sim(x_num_i2,x_den,temp_fac)
                    num_i2_i3_loss=-tf.log(tf.exp(num_i2_i3_ss)/(tf.exp(num_i2_i3_ss)+tf.math.reduce_sum(tf.exp(den_i2_i3_ss))))
                    net_global_loss = net_global_loss + num_i2_i3_loss
                    # for positive pair (x_a_i2, x_a_i1);
                    # numerator same & denominator of loss term (den_i3_i1_ss) & loss (num_i3_i1_loss)
                    den_i3_i2_ss=self.cos_sim(x_num_i3,x_den,temp_fac)
                    num_i3_i2_loss=-tf.log(tf.exp(num_i2_i3_ss)/(tf.exp(num_i2_i3_ss)+tf.math.reduce_sum(tf.exp(den_i3_i2_ss))))
                    net_global_loss = net_global_loss + num_i3_i2_loss

                    # for positive pair (x_i1, x_j1): (i1,i4)
                    # numerator of loss term (num_i2_i3_ss) & denominator of loss term (den_i2_i3_ss) & loss (num_i2_i3_loss)
                    num_i1_i4_ss=self.cos_sim(x_num_i1,x_num_i4,temp_fac)
                    den_i1_i4_ss=self.cos_sim(x_num_i1,x_den,temp_fac)
                    num_i1_i4_loss=-tf.log(tf.exp(num_i1_i4_ss)/(tf.exp(num_i1_i4_ss)+tf.math.reduce_sum(tf.exp(den_i1_i4_ss))))
                    net_global_loss = net_global_loss + num_i1_i4_loss
                    # for positive pair (x_j1, x_i1)
                    # numerator same & denominator of loss term (den_i3_i2_ss) & loss (num_i3_i2_loss)
                    den_i4_i1_ss=self.cos_sim(x_num_i4,x_den,temp_fac)
                    num_i4_i1_loss=-tf.log(tf.exp(num_i1_i4_ss)/(tf.exp(num_i1_i4_ss)+tf.math.reduce_sum(tf.exp(den_i4_i1_ss))))
                    net_global_loss = net_global_loss + num_i4_i1_loss

                    # for positive pair (x_j1, x_a_j1): (i4,i5)
                    # numerator of loss term (num_i2_i4_ss) & denominator of loss term (den_i2_i4_ss) & loss (num_i2_i4_loss)
                    num_i4_i5_ss=self.cos_sim(x_num_i4, x_num_i5, temp_fac)
                    den_i4_i5_ss=self.cos_sim(x_num_i4, x_den, temp_fac)
                    num_i4_i5_loss=-tf.log(tf.exp(num_i4_i5_ss)/(tf.exp(num_i4_i5_ss)+tf.math.reduce_sum(tf.exp(den_i4_i5_ss))))
                    net_global_loss = net_global_loss + num_i4_i5_loss
                    # for positive pair (x_a_j1, x_j1)
                    # numerator same & denominator of loss term (den_i4_i2_ss) & loss (num_i4_i2_loss)
                    den_i5_i4_ss=self.cos_sim(x_num_i5, x_den, temp_fac)
                    num_i5_i4_loss=-tf.log(tf.exp(num_i4_i5_ss)/(tf.exp(num_i4_i5_ss)+tf.math.reduce_sum(tf.exp(den_i5_i4_ss))))
                    net_global_loss = net_global_loss + num_i5_i4_loss
                    
                    # for positive pair (x_j1, x_a_j2): (i5,i6)
                    # numerator of loss term (num_i2_i4_ss) & denominator of loss term (den_i2_i4_ss) & loss (num_i2_i4_loss)
                    num_i5_i6_ss=self.cos_sim(x_num_i5, x_num_i6, temp_fac)
                    den_i5_i6_ss=self.cos_sim(x_num_i5, x_den, temp_fac)
                    num_i5_i6_loss=-tf.log(tf.exp(num_i5_i6_ss)/(tf.exp(num_i5_i6_ss)+tf.math.reduce_sum(tf.exp(den_i5_i6_ss))))
                    net_global_loss = net_global_loss + num_i5_i6_loss
                    # for positive pair (x_a_j2, x_j1)
                    # numerator same & denominator of loss term (den_i4_i2_ss) & loss (num_i4_i2_loss)
                    den_i6_i5_ss=self.cos_sim(x_num_i6, x_den, temp_fac)
                    num_i6_i5_loss=-tf.log(tf.exp(num_i5_i6_ss)/(tf.exp(num_i5_i6_ss)+tf.math.reduce_sum(tf.exp(den_i6_i5_ss))))
                    net_global_loss = net_global_loss + num_i6_i5_loss

                    # for positive pair (x_a_i1, x_a_j2): (i2,i5)
                    # numerator of loss term (num_i2_i4_ss) & denominator of loss term (den_i2_i4_ss) & loss (num_i2_i4_loss)
                    num_i2_i5_ss=self.cos_sim(x_num_i2, x_num_i5, temp_fac)
                    den_i2_i5_ss=self.cos_sim(x_num_i2, x_den, temp_fac)
                    num_i2_i5_loss=-tf.log(tf.exp(num_i2_i5_ss)/(tf.exp(num_i2_i5_ss)+tf.math.reduce_sum(tf.exp(den_i2_i5_ss))))
                    net_global_loss = net_global_loss + num_i2_i5_loss
                    # for positive pair (x_a_j1, x_a_i1)
                    # numerator same & denominator of loss term (den_i4_i2_ss) & loss (num_i4_i2_loss)
                    den_i5_i2_ss=self.cos_sim(x_num_i5, x_den, temp_fac)
                    num_i5_i2_loss=-tf.log(tf.exp(num_i2_i5_ss)/(tf.exp(num_i2_i5_ss)+tf.math.reduce_sum(tf.exp(den_i5_i2_ss))))
                    net_global_loss = net_global_loss + num_i5_i2_loss

                    # for positive pair (x_a_i2, x_a_j2): (i3,i6)
                    # numerator of loss term (num_i2_i4_ss) & denominator of loss term (den_i2_i4_ss) & loss (num_i2_i4_loss)
                    num_i3_i6_ss=self.cos_sim(x_num_i3, x_num_i6, temp_fac)
                    den_i3_i6_ss=self.cos_sim(x_num_i3, x_den, temp_fac)
                    num_i3_i6_loss=-tf.log(tf.exp(num_i3_i6_ss)/(tf.exp(num_i3_i6_ss)+tf.math.reduce_sum(tf.exp(den_i3_i6_ss))))
                    net_global_loss = net_global_loss + num_i3_i6_loss
                    # for positive pair (x_a_j2, x_a_i2)
                    # numerator same & denominator of loss term (den_i4_i2_ss) & loss (num_i4_i2_loss)
                    den_i6_i3_ss=self.cos_sim(x_num_i6, x_den, temp_fac)
                    num_i6_i3_loss=-tf.log(tf.exp(num_i3_i6_ss)/(tf.exp(num_i3_i6_ss)+tf.math.reduce_sum(tf.exp(den_i6_i3_ss))))
                    net_global_loss = net_global_loss + num_i6_i3_loss



        if(global_loss_exp_no==0):
            bs=2*self.batch_size
            reg_cost=net_global_loss/bs
        elif(global_loss_exp_no==1):
            bs=3*self.batch_size
            reg_cost=net_global_loss/bs
        elif(global_loss_exp_no==2):
            bs=4*self.batch_size
            reg_cost=net_global_loss/bs
        elif(global_loss_exp_no==4):
            bs=6*self.batch_size
            reg_cost=net_global_loss/bs
        else:
            bs=3*self.batch_size
            reg_cost=net_global_loss/bs

        # var list of u-net (segmentation net)
        reg_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'enc_' in var_name: reg_net_vars.append(v)
            elif 'dec_' in var_name: reg_net_vars.append(v)
            elif 'reg_' in var_name: reg_net_vars.append(v)
        #print('var_list',reg_net_vars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            cost_reg=tf.reduce_mean(reg_cost)
            optimizer_unet_reg = tf.train.AdamOptimizer(learn_rate_seg).minimize(cost_reg, var_list=reg_net_vars)

        #accu= tf.metrics.accuracy(labels=y_l_onehot,predictions=seg_fin_layer)

        seg_summary = tf.summary.scalar('reg_cost', tf.reduce_mean(reg_cost))
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        train_summary = tf.summary.merge([seg_summary])

        val_totalc = tf.placeholder(tf.float32, shape=[], name='val_totalc')
        val_totalc_sum= tf.summary.scalar('val_totalc_', val_totalc)
        val_summary = tf.summary.merge([val_totalc_sum])
        #val_summary = tf.summary.merge([mean_dice_summary,val_totalc_sum])

        return {'x':x, 'train_phase':train_phase, 'reg_cost':cost_reg, \
                'optimizer_unet_reg':optimizer_unet_reg, 'train_summary':train_summary, 'reg_pred':reg_pred,\
                'val_totalc':val_totalc, 'val_summary':val_summary}

    def seg_unet(self,learn_rate_seg=0.001,dsc_loss=2,en_1hot=0,mtask_en=1,fs_de=2):
        # Define the U-Net (Encoder & Decoder Network) to segment the input image

        # No of channels in each layer
        no_filters=[1, 16, 32, 64, 128, 128]

        if(self.num_classes==2):
            class_weights = tf.constant([[0.1, 0.9]],name='class_weights')
        elif(self.num_classes==3):
            class_weights = tf.constant([[0.1, 0.45, 0.45]],name='class_weights')
        elif(self.num_classes==4):
            class_weights = tf.constant([[0.1, 0.3, 0.3, 0.3]],name='class_weights')
        elif (self.num_classes==8):
            class_weights = tf.constant([[0.09, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13]], name='class_weights')

        num_channels=self.num_channels

        # placeholders for the network
        # Inputs
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x')
        if(en_1hot==1):
            y_l = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y,self.num_classes], name='y_l')
        else:
            y_l = tf.placeholder(tf.int32, shape=[None, self.img_size_x, self.img_size_y], name='y_l')
        train_phase = tf.placeholder(tf.bool, name='train_phase')

        if(en_1hot==0):
            y_l_onehot=tf.one_hot(y_l,depth=self.num_classes)
        else:
            y_l_onehot=y_l

        #print('x,y_l_onehot',x,y_l_onehot)

        ###################################
        # Encoder network
        ###################################
        # Last layer from Encoder network (e)
        enc_c6_b,enc_layers_list = self.encoder_network(x, train_phase, no_filters,encoder_list_return=1)

        ###################################
        # skip-connection layers from encoder
        enc_c1_b,enc_c2_b,enc_c3_b,enc_c4_b,enc_c5_b = enc_layers_list[0],enc_layers_list[1],enc_layers_list[2],enc_layers_list[3],enc_layers_list[4]

        ###################################
        # Decoder network - Upsampling Path
        ###################################
        scale_fac=2
        dec_c6_up = layers.upsample_layer(ip_layer=enc_c6_b, method=self.interp_val, scale_factor=int(scale_fac))
        #print('dec 2 large up',dec_c6_up)
        dec_dc6 = layers.conv2d_layer(ip_layer=dec_c6_up,name='dec_dc6', kernel_size=(fs_de,fs_de),num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c6 = tf.concat((dec_dc6,enc_c5_b),axis=3,name='dec_cat_c6')
        dec_c5_a = layers.conv2d_layer(ip_layer=dec_cat_c6,name='dec_c5_a', num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c5_b = layers.conv2d_layer(ip_layer=dec_c5_a,name='dec_c5_b', num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        dec_c5_up = layers.upsample_layer(ip_layer=dec_c5_b, method=self.interp_val, scale_factor=int(scale_fac))
        #print('dec large up',dec_c6_up,dec_c5_up)
        dec_dc5 = layers.conv2d_layer(ip_layer=dec_c5_up,name='dec_dc5', kernel_size=(fs_de,fs_de),num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c5 = tf.concat((dec_dc5,enc_c4_b),axis=3,name='dec_cat_c5')
        dec_c4_a = layers.conv2d_layer(ip_layer=dec_cat_c5,name='dec_c4_a', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c4_b = layers.conv2d_layer(ip_layer=dec_c4_a,name='dec_c4_b', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 4
        dec_up4 = layers.upsample_layer(ip_layer=dec_c4_b, method=self.interp_val, scale_factor=scale_fac)
        dec_dc4 = layers.conv2d_layer(ip_layer=dec_up4,name='dec_dc4', kernel_size=(fs_de,fs_de),num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c4 = tf.concat((dec_dc4,enc_c3_b),axis=3,name='dec_cat_c4')
        dec_c3_a = layers.conv2d_layer(ip_layer=dec_cat_c4,name='dec_c3_a', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c3_b = layers.conv2d_layer(ip_layer=dec_c3_a,name='dec_c3_b', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 3
        dec_up3 = layers.upsample_layer(ip_layer=dec_c3_b, method=self.interp_val, scale_factor=scale_fac)
        dec_dc3 = layers.conv2d_layer(ip_layer=dec_up3,name='dec_dc3', kernel_size=(fs_de,fs_de),num_filters=no_filters[2],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c3 = tf.concat((dec_dc3,enc_c2_b),axis=3,name='dec_cat_c3')
        dec_c2_a = layers.conv2d_layer(ip_layer=dec_cat_c3,name='dec_c2_a', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c2_b = layers.conv2d_layer(ip_layer=dec_c2_a,name='dec_c2_b', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 2
        dec_up2 = layers.upsample_layer(ip_layer=dec_c2_b, method=self.interp_val, scale_factor=scale_fac)
        dec_dc2 = layers.conv2d_layer(ip_layer=dec_up2,name='dec_dc2', kernel_size=(fs_de,fs_de),num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c2 = tf.concat((dec_dc2,enc_c1_b),axis=3,name='dec_cat_c2')
        dec_c1_a = layers.conv2d_layer(ip_layer=dec_cat_c2,name='dec_c1_a', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 1
        seg_c1_a = layers.conv2d_layer(ip_layer=dec_c1_a,name='seg_c1_a',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_b = layers.conv2d_layer(ip_layer=seg_c1_a,name='seg_c1_b', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        #Final output layer - Logits before softmax
        seg_fin_layer = layers.conv2d_layer(ip_layer=seg_c1_b,name='seg_fin_layer', num_filters=self.num_classes,use_bias=False, use_relu=False, use_batch_norm=False, training_phase=train_phase)
        actual_cost = loss.dice_loss_with_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)

        # Predict Class
        y_pred = tf.nn.softmax(seg_fin_layer)
        y_pred_cls = tf.argmax(y_pred,axis=3)

        ########################
        # Simple Cross Entropy (CE) between predicted labels and true labels
        if(dsc_loss==1):
            # For dice score loss function
            #without background
            seg_cost = loss.dice_loss_without_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
        elif(dsc_loss==2):
            #with background
            seg_cost = loss.dice_loss_with_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
        else:
            # For Weighted Cross Entropy loss function with background
            seg_cost = loss.pixel_wise_cross_entropy_loss_weighted(logits=seg_fin_layer, labels=y_l_onehot, class_weights=class_weights)

        # var list of u-net (segmentation net)
        all_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'enc_' in var_name: all_net_vars.append(v)
            elif 'dec_' in var_name: all_net_vars.append(v)
            elif 'seg_' in var_name: all_net_vars.append(v)

        dec_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'dec_' in var_name: dec_net_vars.append(v)
            if 'seg_' in var_name: dec_net_vars.append(v)

        seg_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'seg_' in var_name: seg_net_vars.append(v)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            cost_seg=tf.reduce_mean(seg_cost)

            optimizer_unet_seg = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(cost_seg,var_list=seg_net_vars)
            optimizer_unet_dec = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(cost_seg,var_list=dec_net_vars)
            optimizer_unet_all = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(cost_seg,var_list=all_net_vars)

        seg_summary = tf.summary.scalar('seg_cost', tf.reduce_mean(seg_cost))
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        train_summary = tf.summary.merge([seg_summary])
        # For dice score summary

        mean_dice = tf.placeholder(tf.float32, shape=[], name='mean_dice')
        mean_dice_summary = tf.summary.scalar('mean_val_dice', mean_dice)

        val_totalc = tf.placeholder(tf.float32, shape=[], name='val_totalc')
        val_totalc_sum= tf.summary.scalar('val_totalc_', val_totalc)
        val_summary = tf.summary.merge([mean_dice_summary,val_totalc_sum])

        if(mtask_en==1):
            return {'x': x, 'y_l':y_l, 'train_phase':train_phase, 'seg_cost': cost_seg,'optimizer_unet_seg':optimizer_unet_seg, \
                'y_pred' : y_pred, 'y_pred_cls': y_pred_cls, 'optimizer_unet_dec':optimizer_unet_dec,'actual_cost':actual_cost,\
                'train_summary':train_summary,'seg_fin_layer':seg_fin_layer,'optimizer_unet_all':optimizer_unet_all, \
                'mean_dice':mean_dice,'val_totalc':val_totalc,'val_summary':val_summary}
        else:
            return {'x': x, 'y_l':y_l, 'train_phase':train_phase,'seg_cost': cost_seg,'optimizer_unet_seg':optimizer_unet_seg, \
                'y_pred' : y_pred, 'y_pred_cls': y_pred_cls, 'optimizer_unet_dec':optimizer_unet_dec,\
                'train_summary':train_summary,'seg_fin_layer':seg_fin_layer,'optimizer_unet_all':optimizer_unet_all,\
                'actual_cost':actual_cost,'mean_dice':mean_dice,'val_totalc':val_totalc,'val_summary':val_summary}


    def decoder_pretrain_net(self,learn_rate_seg=0.001,temp_fac=1,no_of_local_regions=5,fs_de=2,no_of_decoder_blocks=1,local_reg_size=0,\
                          wgt_en=0,no_of_neg_local_regions=5,local_loss_exp_no=0,no_of_neg_regs_override=5,inf=0):
        # Define the Encoder + 'L' Decoder blocks Network with g_2 a small 1x1 network to pre-train the decoder layers
        # L = no_of_decoder_blocks

        # No of channels in each layer
        no_filters=[1, 16, 32, 64, 128, 128]

        num_channels=self.num_channels
        # placeholders for the network
        # Inputs
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x')
        train_phase = tf.placeholder(tf.bool, name='train_phase')

        ###################################
        # Encoder network
        ########################
        # Last layer from Encoder network (e)
        enc_c6_b, enc_layers_list = self.encoder_network(x, train_phase, no_filters, encoder_list_return=1)

        ###################################
        # skip-connection layers from encoder
        enc_c1_b, enc_c2_b, enc_c3_b, enc_c4_b, enc_c5_b = enc_layers_list[0], enc_layers_list[1], enc_layers_list[2], enc_layers_list[3], enc_layers_list[4]

        ########################
        # Decoder Network + g_2 small network to compute the local regions
        # For decoder, can use 'l' blocks based on input
        ########################
        # each decoder level - one upsampling layer + one 2x2 conv op. + skip connection from encoder level + two 3x3 conv op.

        scale_fac=2
        dec_c6_up = layers.upsample_layer(ip_layer=enc_c6_b, method=self.interp_val, scale_factor=int(scale_fac))
        #print('dec 2 large up',dec_c6_up)
        dec_dc6 = layers.conv2d_layer(ip_layer=dec_c6_up,name='dec_dc6', kernel_size=(fs_de,fs_de),num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c6 = tf.concat((dec_dc6,enc_c5_b),axis=3,name='dec_cat_c6')
        dec_c5_a = layers.conv2d_layer(ip_layer=dec_cat_c6,name='dec_c5_a', num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c5_b = layers.conv2d_layer(ip_layer=dec_c5_a,name='dec_c5_b', num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        if(no_of_decoder_blocks>=1):
            # No of decoder blocks = 1
            tmp_dec_layer=dec_c5_b
            tmp_no_filters=no_filters[4]
            #print('decoder level I', no_of_decoder_blocks, tmp_no_filters, tmp_dec_layer)

            if(no_of_decoder_blocks>=2):
                # No of decoder blocks =  2
                dec_c5_up = layers.upsample_layer(ip_layer=dec_c5_b, method=self.interp_val,scale_factor=int(scale_fac))
                dec_dc5 = layers.conv2d_layer(ip_layer=dec_c5_up, name='dec_dc5', kernel_size=(fs_de, fs_de),num_filters=no_filters[4], use_relu=True, use_batch_norm=True,training_phase=train_phase)
                dec_cat_c5 = tf.concat((dec_dc5, enc_c4_b), axis=3, name='dec_cat_c5')
                dec_c4_a = layers.conv2d_layer(ip_layer=dec_cat_c5, name='dec_c4_a', num_filters=no_filters[4],use_relu=True, use_batch_norm=True, training_phase=train_phase)
                dec_c4_b = layers.conv2d_layer(ip_layer=dec_c4_a, name='dec_c4_b', num_filters=no_filters[4],use_relu=True, use_batch_norm=True, training_phase=train_phase)
                tmp_dec_layer = dec_c4_b
                tmp_no_filters=no_filters[4]
                #print('decoder level II', no_of_decoder_blocks, tmp_no_filters, tmp_dec_layer)

                if(no_of_decoder_blocks>=3):
                    # No of decoder blocks = 3
                    dec_up4 = layers.upsample_layer(ip_layer=dec_c4_b, method=self.interp_val, scale_factor=scale_fac)
                    dec_dc4 = layers.conv2d_layer(ip_layer=dec_up4, name='dec_dc4', kernel_size=(fs_de, fs_de),num_filters=no_filters[3], use_relu=True, use_batch_norm=True,training_phase=train_phase)
                    dec_cat_c4 = tf.concat((dec_dc4, enc_c3_b), axis=3, name='dec_cat_c4')
                    dec_c3_a = layers.conv2d_layer(ip_layer=dec_cat_c4, name='dec_c3_a', num_filters=no_filters[3],use_relu=True, use_batch_norm=True, training_phase=train_phase)
                    dec_c3_b = layers.conv2d_layer(ip_layer=dec_c3_a, name='dec_c3_b', num_filters=no_filters[3],use_relu=True, use_batch_norm=True, training_phase=train_phase)
                    tmp_dec_layer = dec_c3_b
                    tmp_no_filters=no_filters[3]
                    #print('decoder level III', no_of_decoder_blocks, tmp_no_filters, tmp_dec_layer)

                    if(no_of_decoder_blocks>=4):
                        # No of decoder blocks = 4
                        dec_up3 = layers.upsample_layer(ip_layer=dec_c3_b, method=self.interp_val,scale_factor=scale_fac)
                        dec_dc3 = layers.conv2d_layer(ip_layer=dec_up3, name='dec_dc3', kernel_size=(fs_de, fs_de),num_filters=no_filters[2], use_relu=True, use_batch_norm=True,training_phase=train_phase)
                        dec_cat_c3 = tf.concat((dec_dc3, enc_c2_b), axis=3, name='dec_cat_c3')
                        dec_c2_a = layers.conv2d_layer(ip_layer=dec_cat_c3, name='dec_c2_a', num_filters=no_filters[2],use_relu=True, use_batch_norm=True, training_phase=train_phase)
                        dec_c2_b = layers.conv2d_layer(ip_layer=dec_c2_a, name='dec_c2_b', num_filters=no_filters[2],use_relu=True, use_batch_norm=True, training_phase=train_phase)
                        tmp_dec_layer = dec_c2_b
                        tmp_no_filters=no_filters[2]
                        #print('decoder level IV', no_of_decoder_blocks, tmp_no_filters, tmp_dec_layer)

                        if(no_of_decoder_blocks>=5):
                            # No of decoder blocks = 5 (full decoder)
                            dec_up2 = layers.upsample_layer(ip_layer=dec_c2_b, method=self.interp_val,scale_factor=scale_fac)
                            dec_dc2 = layers.conv2d_layer(ip_layer=dec_up2, name='dec_dc2', kernel_size=(fs_de, fs_de),num_filters=no_filters[1], use_relu=True, use_batch_norm=True,training_phase=train_phase)
                            dec_cat_c2 = tf.concat((dec_dc2, enc_c1_b), axis=3, name='dec_cat_c2')
                            dec_c1_a = layers.conv2d_layer(ip_layer=dec_cat_c2, name='dec_c1_a',num_filters=no_filters[1], use_relu=True,use_batch_norm=True, training_phase=train_phase)
                            tmp_dec_layer = dec_c1_a
                            tmp_no_filters=no_filters[1]
                            #print('decoder level V', no_of_decoder_blocks, tmp_no_filters, tmp_dec_layer)

        #g_2 small network with two 1x1 convolutions
        seg_c1_a = layers.conv2d_layer(ip_layer=tmp_dec_layer,name='seg_c1_a', kernel_size=(1,1), num_filters=tmp_no_filters,use_bias=False, use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_b = layers.conv2d_layer(ip_layer=seg_c1_a, name='seg_c1_b', kernel_size=(1, 1),num_filters=tmp_no_filters, use_bias=False, use_relu=False,use_batch_norm=False, training_phase=train_phase)

        y_fin_tmp=seg_c1_b
        #print('y_fin',y_fin_tmp)

        bs=2*self.batch_size

        #define local loss term

        # dimension of feature map in x and y directions (im_x,im_y) defined based on the no. of decoder blocks used.
        # if local_reg_size=1 then local region size is 3x3, local_reg_size=0 then local region size is 1x1.
        if(no_of_decoder_blocks==1):
            if(local_reg_size==1):
                im_x,im_y=int(self.img_size_x/16)-4,int(self.img_size_y/16)-4
            else:
                im_x,im_y=int(self.img_size_x/16)-1,int(self.img_size_y/16)-1
        elif(no_of_decoder_blocks == 2):
            if (local_reg_size == 1):
                im_x,im_y=int(self.img_size_x/8)-4,int(self.img_size_y/8)-4
            else:
                im_x,im_y=int(self.img_size_x/8)-1,int(self.img_size_y/8)-1
        elif(no_of_decoder_blocks == 3):
            if(local_reg_size==1):
                im_x,im_y=int(self.img_size_x/4)-4,int(self.img_size_y/4)-4
            else:
                im_x,im_y=int(self.img_size_x/4)-1,int(self.img_size_y/4)-1
        elif(no_of_decoder_blocks == 4):
            if(local_reg_size==1):
                im_x,im_y=int(self.img_size_x/2)-4,int(self.img_size_y/2)-4
            else:
                im_x,im_y=int(self.img_size_x/2)-1,int(self.img_size_y/2)-1
        else:
            if(local_reg_size==1):
                im_x,im_y=int(self.img_size_x)-4,int(self.img_size_y)-4
            else:
                im_x,im_y=int(self.img_size_x)-1,int(self.img_size_y)-1


        if(no_of_local_regions==9):
            #Indexes for the local regions to be selected for computing local contrastive loss
            # All the local regions for positive samples from images (x_a1_i,x_a2_i), where x_a1_i,x_a2_i are two augmented versions of x_i.
            pos_sample_indexes=np.zeros((no_of_local_regions,2),dtype=np.int32)
            pos_sample_indexes[0],pos_sample_indexes[1],pos_sample_indexes[2]=[0,0],[0,int(im_y/2)],[0,im_y]
            pos_sample_indexes[3],pos_sample_indexes[4],pos_sample_indexes[5]=[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y]
            pos_sample_indexes[6],pos_sample_indexes[7],pos_sample_indexes[8]=[im_x,0],[im_x,int(im_y/2)],[im_x,im_y]

            #Indexes for negative samples w,r.t a positive sample.
            neg_sample_indexes=np.zeros((no_of_local_regions,no_of_neg_local_regions,2),dtype=np.int32)
            # Each positive local region will have corresponding regions that act as negative samples to be contrasted.
            # For each positive sample, we pick the nearby no_of_neg_local_regions (5) local regions as negative samples from both the images (x_a1_i, x_a2_i)
            # for local region at (0,0), define the negative samples co-ordinates accordingly
            neg_sample_indexes[0,:,:]=[[0,int(im_y/2)],[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[0,im_y],[im_x,0]]
            # similarly, define negative samples co-ordinates according to positive sample
            neg_sample_indexes[1,:,:]=[[0,0],[0,im_y],[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y]]
            neg_sample_indexes[2,:,:]=[[0,0],[0,int(im_y/2)],[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y],[im_x,im_y]]
            neg_sample_indexes[3,:,:]=[[0,0],[0,int(im_y/2)],[int(im_x/2),int(im_y/2)],[im_x,0],[im_x,int(im_y/2)]]
            neg_sample_indexes[4,:,:]=[[0,0],[0,int(im_y/2)],[int(im_x/2),0],[int(im_x/2),im_y],[im_x,int(im_y/2)]]
            neg_sample_indexes[5,:,:]=[[0,int(im_y/2)],[0,im_y],[int(im_x/2),int(im_y/2)],[im_x,int(im_y/2)],[im_x,im_y]]
            neg_sample_indexes[6,:,:]=[[0,0],[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[im_x,int(im_y/2)],[im_x,im_y]]
            neg_sample_indexes[7,:,:]=[[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y],[im_x,0],[im_x,im_y]]
            neg_sample_indexes[8,:,:]=[[0,im_y],[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y],[im_x,0],[im_x,int(im_y/2)]]

        elif(no_of_local_regions==13 and no_of_neg_local_regions==5):
            # Indexes for the local regions to be selected for computing local contrastive loss
            # All the local regions for positive samples from images (x_a1_i,x_a2_i), where x_a1_i,x_a2_i are two augmented versions of x_i.
            pos_sample_indexes=np.zeros((no_of_local_regions,2),dtype=np.int32)
            pos_sample_indexes[0],pos_sample_indexes[1],pos_sample_indexes[2]=[0,0],[0,int(im_y/2)],[0,im_y]
            pos_sample_indexes[3],pos_sample_indexes[4],pos_sample_indexes[5]=[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y]
            pos_sample_indexes[6],pos_sample_indexes[7],pos_sample_indexes[8]=[im_x,0],[im_x,int(im_y/2)],[im_x,im_y]
            pos_sample_indexes[9],pos_sample_indexes[10]=[int(im_x/4),int(im_y/4)],[int(im_x/4),int(3*im_y/4)]
            pos_sample_indexes[11],pos_sample_indexes[12]=[int(3*im_x/4),int(im_y/4)],[int(3*im_x/4),int(3*im_y/4)]

            #Indexes for negative samples w,r.t a positive sample.
            neg_sample_indexes=np.zeros((no_of_local_regions,no_of_neg_local_regions,2),dtype=np.int32)
            # Each positive local region will have corresponding regions that act as negative samples to be contrasted.
            # For each positive sample, we pick the nearby no_of_neg_local_regions (5) local regions as negative samples from both the images (x_a1_i, x_a2_i)

            if(local_reg_size==1):
                # local region size = 3x3
                # for local region at (0,0), define the negative samples co-ordinates accordingly
                neg_sample_indexes[0,:,:]=[[0,int(im_y/2)],[int(im_x/4),int(im_y/4)],[int(im_x/4),int(im_y/2)],[int(im_x/2),0],[int(im_x/2),int(im_y/4)]]
                # similarly, define negative samples co-ordinates according to positive sample
                neg_sample_indexes[1,:,:]=[[0,0],[0,im_y],[int(im_x/4),int(im_y/4)],[int(im_x/4),int(3*im_y/4)],[int(im_x/2),int(im_y/2)]]
                neg_sample_indexes[2,:,:]=[[0,int(im_y/2)],[int(im_x/4),int(im_y/2)],[int(im_x/4),int(3*im_y/4)],[int(im_x/2),int(3*im_y/4)],[int(im_x/2),im_y]]
                neg_sample_indexes[3,:,:]=[[0,0],[int(im_x/4),int(im_y/4)],[int(im_x/2),int(im_y/2)],[im_x,0],[int(3*im_x/4),int(im_y/4)]]
                neg_sample_indexes[4,:,:]=[[int(im_x/4),int(im_y/4)],[int(im_x/4),int(3*im_y/4)],[int(3*im_x/4),int(im_y/4)],[int(3*im_x/4),int(3*im_y/4)],[int(im_x/2),0]]
                neg_sample_indexes[5,:,:]=[[0,im_y],[int(im_x/2),int(im_y/2)],[im_x,im_y],[int(im_x/4),int(3*im_y/4)],[int(3*im_x/4),int(3*im_y/4)]]
                neg_sample_indexes[6,:,:]=[[int(im_x/2),0],[int(im_x/2),int(im_y/4)],[im_x,int(im_y/2)],[int(3*im_x/4),int(im_y/4)],[int(3*im_x/4),int(im_y/2)]]
                neg_sample_indexes[7,:,:]=[[int(im_x/2),int(im_y/2)],[im_x,0],[im_x,im_y],[int(3*im_x/4),int(im_y/4)],[int(3*im_x/4),int(3*im_y/4)]]
                neg_sample_indexes[8,:,:]=[[int(im_x/2),int(3*im_y/4)],[int(im_x/2),im_y],[im_x,int(im_y/2)],[int(3*im_x/4),int(3*im_y/4)],[int(3*im_x/4),int(im_y/2)]]
                neg_sample_indexes[9,:,:]=[[0,0],[0,int(im_y/2)],[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[int(im_x/4),int(3*im_y/4)]]
                neg_sample_indexes[10,:,:]=[[0,int(im_y/2)],[0,im_y],[int(im_x/4),int(im_y/4)],[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y]]
                neg_sample_indexes[11,:,:]=[[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[im_x,0],[im_x,int(im_y/2)],[int(3*im_x/4),int(3*im_y/4)]]
                neg_sample_indexes[12,:,:]=[[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y],[im_x,int(im_y/2)],[im_x,im_y],[int(3*im_x/4),int(im_y/4)]]
            else:
                # local region size = 1x1
                # for local region at (0,0), define the negative samples co-ordinates accordingly
                neg_sample_indexes[0,:,:]=[[0,int(im_y/2)],[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[int(im_x/4),int(im_y/4)],[int(im_x/4),int(3*im_y/4)]]
                # similarly, define negative samples co-ordinates according to positive sample
                neg_sample_indexes[1,:,:]=[[0,0],[0,im_y],[int(im_x/2),int(im_y/2)],[int(im_x/4),int(im_y/4)],[int(im_x/4),int(3*im_y/4)]]
                neg_sample_indexes[2,:,:]=[[0,int(im_y/2)],[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y],[int(3*im_x/4),int(im_y/4)],[int(im_x/4),int(3*im_y/4)]]
                neg_sample_indexes[3,:,:]=[[0,0],[int(im_x/2),int(im_y/2)],[im_x,0],[int(im_x/4),int(im_y/4)],[int(3*im_x/4),int(im_y/4)]]
                neg_sample_indexes[4,:,:]=[[int(im_x/4),int(im_y/4)],[int(im_x/4),int(3*im_y/4)],[int(3*im_x/4),int(im_y/4)],[int(3*im_x/4),int(3*im_y/4)],[int(im_x/2),0]]
                neg_sample_indexes[5,:,:]=[[0,im_y],[int(im_x/2),int(im_y/2)],[im_x,im_y],[int(im_x/4),int(3*im_y/4)],[int(3*im_x/4),int(3*im_y/4)]]
                neg_sample_indexes[6,:,:]=[[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[im_x,int(im_y/2)],[int(3*im_x/4),int(im_y/4)],[int(im_x/4),int(im_y/4)]]
                neg_sample_indexes[7,:,:]=[[int(im_x/2),int(im_y/2)],[im_x,0],[im_x,im_y],[int(3*im_x/4),int(im_y/4)],[int(3*im_x/4),int(3*im_y/4)]]
                neg_sample_indexes[8,:,:]=[[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y],[im_x,int(im_y/2)],[int(3*im_x/4),int(3*im_y/4)],[int(3*im_x/4),int(im_y/4)]]
                neg_sample_indexes[9,:,:]=[[0,0],[0,int(im_y/2)],[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[int(im_x/4),int(3*im_y/4)]]
                neg_sample_indexes[10,:,:]=[[0,int(im_y/2)],[0,im_y],[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y],[int(3*im_x/4),int(3*im_y/4)]]
                neg_sample_indexes[11,:,:]=[[int(im_x/2),0],[int(im_x/2),int(im_y/2)],[im_x,0],[0,int(im_y/2)],[int(im_x/4),int(im_y/4)]]
                neg_sample_indexes[12,:,:]=[[int(im_x/2),int(im_y/2)],[int(im_x/2),im_y],[im_x,int(im_y/2)],[im_x,im_y],[int(im_x/4),int(3*im_y/4)]]


        local_loss=0
        if(inf==1):
            y_fin=y_fin_tmp
            local_loss=1
            bs,tmp_batch_size=24,12
            #bs=2*self.batch_size
        elif(local_loss_exp_no==0):
            y_fin=y_fin_tmp
            #print('y_fin_local',y_fin)

            #loop over each image pair to iterate over all positive local regions within a feature map to calculate the local contrastive loss
            for pos_index in range(0,bs,2):

                #indexes of positive pair of samples (f_a1_i,f_a2_i) of input images (x_a1_i,x_a2_i) from the batch of feature maps.
                num_i1=np.arange(pos_index,pos_index+1,dtype=np.int32)
                num_i2 = np.arange(pos_index+1, pos_index+2, dtype=np.int32)

                # gather required positive samples (f_a1_i,f_a2_i) of (x_a1_i,x_a2_i) for the numerator term
                x_num_i1=tf.gather(y_fin,num_i1)
                x_num_i2=tf.gather(y_fin,num_i2)
                #print('x_num_i1,x_num_i2',x_num_i1,x_num_i2)

                # if local region size is 3x3
                if(local_reg_size==1):
                    # loop over all defined local regions within a feature map
                    for local_pos_index in range(0,no_of_local_regions,1):
                        # 'pos_index_num' is the positive local region index in feature map f_a1_i of image x_a1_i that contributes to the numerator term.
                        #fetch x and y coordinates
                        x_num_tmp_i1=tf.gather(x_num_i1,[pos_sample_indexes[local_pos_index,0],pos_sample_indexes[local_pos_index,0]+1,pos_sample_indexes[local_pos_index,0]+2],axis=1)
                        x_num_tmp_i1=tf.gather(x_num_tmp_i1,[pos_sample_indexes[local_pos_index,1],pos_sample_indexes[local_pos_index,1]+1,pos_sample_indexes[local_pos_index,1]+2],axis=2)
                        x_n_i1_flat = tf.layers.flatten(inputs=x_num_tmp_i1)
                        if(wgt_en==1):
                            x_w3_n_i1=tf.layers.dense(inputs=x_n_i1_flat, units=128, name='seg_pred', activation=None, use_bias=False,reuse=tf.AUTO_REUSE)
                        else:
                            x_w3_n_i1=x_n_i1_flat

                        # corresponding positive local region index in feature map f_a2_i of image x_a2_i that contributes to the numerator term.
                        #fetch x and y coordinates
                        x_num_tmp_i2=tf.gather(x_num_i2,[pos_sample_indexes[local_pos_index,0],pos_sample_indexes[local_pos_index,0]+1,pos_sample_indexes[local_pos_index,0]+2],axis=1)
                        x_num_tmp_i2=tf.gather(x_num_tmp_i2,[pos_sample_indexes[local_pos_index,1],pos_sample_indexes[local_pos_index,1]+1,pos_sample_indexes[local_pos_index,1]+2],axis=2)
                        x_n_i2_flat = tf.layers.flatten(inputs=x_num_tmp_i2)
                        if(wgt_en==1):
                            x_w3_n_i2=tf.layers.dense(inputs=x_n_i2_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE)
                        else:
                            x_w3_n_i2=x_n_i2_flat

                        # calculate cosine similarity score for the pair of positive local regions with index 'pos_index_den' within the feature maps from images (x_a1_i,x_a2_i)
                        # loss for positive pairs of local regions in feature maps  (f_a1_i,f_a2_i) & (f_a2_i,f_a1_i) in (num_i1_loss,num_i2_loss)

                        # Numerator loss terms of local loss
                        num_i1_ss=self.cos_sim(x_w3_n_i1,x_w3_n_i2,temp_fac)
                        num_i2_ss=self.cos_sim(x_w3_n_i2,x_w3_n_i1,temp_fac)

                        # Negative local regions as per the chosen positive local region at index 'pos_index_den'
                        neg_samples_index_list = np.squeeze(neg_sample_indexes[local_pos_index])
                        no_of_neg_pts=len(neg_samples_index_list)

                        # Denominator loss terms of local loss
                        den_i1_ss,den_i2_ss=0,0

                        for local_neg_index in range(0,no_of_neg_pts,1):
                            #negative local regions in feature map (f_a1_i) from image (x_a1_i)
                            x_den_tmp_i1=tf.gather(x_num_i1,[neg_samples_index_list[local_neg_index,0],neg_samples_index_list[local_neg_index,0]+1,neg_samples_index_list[local_neg_index,0]+2],axis=1)
                            x_den_tmp_i1=tf.gather(x_den_tmp_i1,[neg_samples_index_list[local_neg_index,1],neg_samples_index_list[local_neg_index,1]+1,neg_samples_index_list[local_neg_index,1]+2],axis=2)
                            x_d_i1_flat = tf.layers.flatten(inputs=x_den_tmp_i1)
                            if(wgt_en==1):
                                x_w3_d_i1=tf.layers.dense(inputs=x_d_i1_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE )
                            else:
                                x_w3_d_i1=x_d_i1_flat

                            # negative local regions in feature map (f_a2_i) from image (x_a2_i)
                            x_den_tmp_i2=tf.gather(x_num_i2,[neg_samples_index_list[local_neg_index,0],neg_samples_index_list[local_neg_index,0]+1,neg_samples_index_list[local_neg_index,0]+2],axis=1)
                            x_den_tmp_i2=tf.gather(x_den_tmp_i2,[neg_samples_index_list[local_neg_index,1],neg_samples_index_list[local_neg_index,1]+1,neg_samples_index_list[local_neg_index,1]+2],axis=2)
                            x_d_i2_flat = tf.layers.flatten(inputs=x_den_tmp_i2)
                            if(wgt_en==1):
                                x_w3_d_i2=tf.layers.dense(inputs=x_d_i2_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE )
                            else:
                                x_w3_d_i2=x_d_i2_flat


                            # cosine score b/w local region of feature map (f_a1_i) vs other local regions within the same feature map (f_a1_i)
                            den_i1_ss=den_i1_ss+tf.exp(self.cos_sim(x_w3_n_i1,x_w3_d_i1,temp_fac))
                            # cosine score b/w local region of feature map (f_a1_i) vs other local regions from another feature map (f_a2_i)
                            den_i1_ss=den_i1_ss+tf.exp(self.cos_sim(x_w3_n_i1,x_w3_d_i2,temp_fac))

                            # cosine score b/w local region of feature map (f_a2_i) vs other local regions within the same feature map (f_a2_i)
                            den_i2_ss=den_i2_ss+tf.exp(self.cos_sim(x_w3_n_i2,x_w3_d_i2,temp_fac))
                            # cosine score b/w local region of feature map (f_a2_i) vs other local regions from another feature map (f_a1_i)
                            den_i2_ss=den_i2_ss+tf.exp(self.cos_sim(x_w3_n_i2,x_w3_d_i1,temp_fac))

                        #local loss from feature map f_a1_i
                        num_i1_loss=-tf.log(tf.math.reduce_sum(tf.exp(num_i1_ss))/(tf.math.reduce_sum(tf.exp(num_i1_ss))+tf.math.reduce_sum(den_i1_ss)))
                        #num_i1_loss=-tf.log(tf.exp(num_i1_ss)/(tf.exp(num_i1_ss)+tf.math.reduce_sum(den_i1_ss)))
                        local_loss = local_loss + num_i1_loss

                        # local loss from feature map f_a2_i
                        num_i2_loss=-tf.log(tf.math.reduce_sum(tf.exp(num_i2_ss))/(tf.math.reduce_sum(tf.exp(num_i2_ss))+tf.math.reduce_sum(den_i2_ss)))
                        #num_i2_loss=-tf.log(tf.exp(num_i2_ss)/(tf.exp(num_i2_ss)+tf.math.reduce_sum(den_i2_ss)))
                        local_loss = local_loss + num_i2_loss

                # if local region size is 1x1
                else:
                    # loop over all defined local regions within a feature map
                    for local_pos_index in range(0,no_of_local_regions,1):
                        # positive local region 'pos_index_den' in feature map from image x_a1_i
                        # fetch x and y coordinates
                        x_num_tmp_i1=tf.gather(x_num_i1,pos_sample_indexes[local_pos_index,0],axis=1)
                        x_num_tmp_i1=tf.gather(x_num_tmp_i1,pos_sample_indexes[local_pos_index,1],axis=1)

                        # corresponding positive local region 'pos_index_den' in feature map from image x_a2_i
                        # fetch x and y coordinates
                        x_num_tmp_i2=tf.gather(x_num_i2,pos_sample_indexes[local_pos_index,0],axis=1)
                        x_num_tmp_i2=tf.gather(x_num_tmp_i2,pos_sample_indexes[local_pos_index,1],axis=1)

                        # calculate cosine similarity score for the pair of positive local regions with index 'pos_index_den' within the feature maps from images (x_a1_i,x_a2_i)
                        # loss for positive pairs of local regions in feature maps  (f_a1_i,f_a2_i) & (f_a2_i,f_a1_i) in (num_i1_loss,num_i2_loss)

                        # Numerator loss terms of local loss
                        num_i1_ss=self.cos_sim(x_num_tmp_i1,x_num_tmp_i2,temp_fac)
                        num_i2_ss=self.cos_sim(x_num_tmp_i2,x_num_tmp_i1,temp_fac)

                        # Negative local regions as per the chosen positive local region at index 'pos_index_den'
                        neg_samples_index_list = np.squeeze(neg_sample_indexes[local_pos_index])
                        no_of_neg_pts=neg_samples_index_list.shape[0]

                        # Denominator loss terms of local loss
                        den_i1_ss,den_i2_ss=0,0

                        for local_neg_index in range(0,no_of_neg_pts,1):
                            # negative local regions in feature map (f_a1_i) from image (x_a1_i)
                            x_den_tmp_i1=tf.gather(x_num_i1,neg_samples_index_list[local_neg_index,0],axis=1)
                            x_den_tmp_i1=tf.gather(x_den_tmp_i1,neg_samples_index_list[local_neg_index,1],axis=1)

                            # negative local regions in feature map (f_a2_i) from image (x_a2_i)
                            x_den_tmp_i2=tf.gather(x_num_i2,neg_samples_index_list[local_neg_index,0],axis=1)
                            x_den_tmp_i2=tf.gather(x_den_tmp_i2,neg_samples_index_list[local_neg_index,1],axis=1)

                            # cosine score b/w local region of feature map (f_a1_i) vs other local regions within the same feature map (f_a1_i)
                            den_i1_ss=den_i1_ss+tf.exp(self.cos_sim(x_num_tmp_i1,x_den_tmp_i1,temp_fac))
                            # cosine score b/w local region of feature map (f_a1_i) vs other local regions from another feature map (f_a2_i)
                            den_i1_ss=den_i1_ss+tf.exp(self.cos_sim(x_num_tmp_i1,x_den_tmp_i2,temp_fac))

                            # cosine score b/w local region of feature map (f_a2_i) vs other local regions within the same feature map (f_a2_i)
                            den_i2_ss = den_i2_ss + tf.exp(self.cos_sim(x_num_tmp_i2, x_den_tmp_i2, temp_fac))
                            # cosine score b/w local region of feature map (f_a2_i) vs other local regions from another feature map (f_a1_i)
                            den_i2_ss=den_i2_ss+tf.exp(self.cos_sim(x_num_tmp_i2,x_den_tmp_i1,temp_fac))

                        #local loss from feature map f_a1_i
                        num_i1_loss=-tf.log(tf.math.reduce_sum(tf.exp(num_i1_ss))/(tf.math.reduce_sum(tf.exp(num_i1_ss))+tf.math.reduce_sum(den_i1_ss)))
                        local_loss = local_loss + num_i1_loss

                        #local loss from feature map f_a2_i
                        num_i2_loss=-tf.log(tf.math.reduce_sum(tf.exp(num_i2_ss))/(tf.math.reduce_sum(tf.exp(num_i2_ss))+tf.math.reduce_sum(den_i2_ss)))
                        local_loss = local_loss + num_i2_loss

            local_loss=local_loss/no_of_local_regions

        elif(local_loss_exp_no==1):
            bs,tmp_batch_size=24,12
            n_parts=4
            y_fin=y_fin_tmp

            #loop over each image pair to iterate over all positive local regions within a feature map to calculate the local contrastive loss
            for pos_index in range(0,tmp_batch_size,1):
                #indexes of positive pairs of samples (x_a1_i,x_a2_i), (x_a1_i,x_a1_j), (x_a1_i,x_a1_k).
                # x_a1_i,x_a2_i are two different augmented versions of original image x_i.
                # while x_a1_j and x_a1_k are two images from two differnt volumes compared to 'i' but belong to corresponding partition as 'i'th volume.
                # index of x_a1_i
                num_i1=np.arange(pos_index,pos_index+1,dtype=np.int32)
                # index of x_a2_i
                pos_index_j=tmp_batch_size+pos_index
                num_i2=np.arange(pos_index_j,pos_index_j+1,dtype=np.int32)
                # index of x_a1_j
                pos_index_j=(n_parts+pos_index)%tmp_batch_size
                num_i3=np.arange(pos_index_j,pos_index_j+1,dtype=np.int32)
                # index of x_a1_k
                pos_index_j=(2*n_parts+pos_index)%tmp_batch_size
                num_i4=np.arange(pos_index_j,pos_index_j+1,dtype=np.int32)

                # gather required positive samples (x_a1_i,x_a2_i), (x_a1_i,x_a1_j), (x_a1_i,x_a1_k) for the numerator term
                x_num_i1=tf.gather(y_fin,num_i1)
                x_num_i2=tf.gather(y_fin,num_i2)
                x_num_i3=tf.gather(y_fin,num_i3)
                x_num_i4=tf.gather(y_fin,num_i4)

                #print('x_num_i1,i2,i3,i4',x_num_i1,x_num_i2,x_num_i3,x_num_i4,pos_sample_indexes[0,0],pos_sample_indexes[0,1])

                # loop over all defined local regions within a feature map
                for local_pos_index in range(0,no_of_local_regions,1):
                    # if local region size is 3x3
                    if(local_reg_size==1):
                        # positive local region 'j' in feature map from image x_a1_i
                        # fetch x and y coordinates of local region
                        x_n_tmp_i1=tf.gather(x_num_i1,[pos_sample_indexes[local_pos_index,0],pos_sample_indexes[local_pos_index,0]+1,pos_sample_indexes[local_pos_index,0]+2],axis=1)
                        x_n_tmp_i1=tf.gather(x_n_tmp_i1,[pos_sample_indexes[local_pos_index,1],pos_sample_indexes[local_pos_index,1]+1,pos_sample_indexes[local_pos_index,1]+2],axis=2)
                        x_n_i1_flat = tf.layers.flatten(inputs=x_n_tmp_i1)
                        if(wgt_en==1):
                            x_num_tmp_i1=tf.layers.dense(inputs=x_n_i1_flat, units=128, name='seg_pred', activation=None, use_bias=False,reuse=tf.AUTO_REUSE)
                        else:
                            x_num_tmp_i1=x_n_i1_flat

                        # positive local region 'j' in feature map from image x_a2_i
                        # fetch x and y coordinates of local region
                        x_n_tmp_i2=tf.gather(x_num_i2,[pos_sample_indexes[local_pos_index,0],pos_sample_indexes[local_pos_index,0]+1,pos_sample_indexes[local_pos_index,0]+2],axis=1)
                        x_n_tmp_i2=tf.gather(x_n_tmp_i2,[pos_sample_indexes[local_pos_index,1],pos_sample_indexes[local_pos_index,1]+1,pos_sample_indexes[local_pos_index,1]+2],axis=2)
                        x_n_i2_flat = tf.layers.flatten(inputs=x_n_tmp_i2)
                        if(wgt_en==1):
                            x_num_tmp_i2=tf.layers.dense(inputs=x_n_i2_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE)
                        else:
                            x_num_tmp_i2=x_n_i2_flat

                        # positive local region 'j' in feature map from image x_a1_j (j is a different volume to i)
                        # fetch x and y coordinates of local region
                        x_n_tmp_i3=tf.gather(x_num_i3,[pos_sample_indexes[local_pos_index,0],pos_sample_indexes[local_pos_index,0]+1,pos_sample_indexes[local_pos_index,0]+2],axis=1)
                        x_n_tmp_i3=tf.gather(x_n_tmp_i3,[pos_sample_indexes[local_pos_index,1],pos_sample_indexes[local_pos_index,1]+1,pos_sample_indexes[local_pos_index,1]+2],axis=2)
                        x_n_i3_flat = tf.layers.flatten(inputs=x_n_tmp_i3)
                        if(wgt_en==1):
                            x_num_tmp_i3=tf.layers.dense(inputs=x_n_i3_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE)
                        else:
                            x_num_tmp_i3=x_n_i3_flat

                        # positive local region 'j' in feature map from image x_a1_k (k is a different volume to i)
                        # fetch x and y coordinates of local region
                        x_n_tmp_i4=tf.gather(x_num_i4,[pos_sample_indexes[local_pos_index,0],pos_sample_indexes[local_pos_index,0]+1,pos_sample_indexes[local_pos_index,0]+2],axis=1)
                        x_n_tmp_i4=tf.gather(x_n_tmp_i4,[pos_sample_indexes[local_pos_index,1],pos_sample_indexes[local_pos_index,1]+1,pos_sample_indexes[local_pos_index,1]+2],axis=2)
                        x_n_i4_flat = tf.layers.flatten(inputs=x_n_tmp_i4)
                        if(wgt_en==1):
                            x_num_tmp_i4=tf.layers.dense(inputs=x_n_i4_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE)
                        else:
                            x_num_tmp_i4=x_n_i4_flat
                    # if local region size is 1x1
                    else:
                        # positive local region 'j' in feature map from image x_a1_i
                        # fetch x and y coordinates of local region
                        x_num_tmp_i1=tf.gather(x_num_i1,pos_sample_indexes[local_pos_index,0],axis=1)
                        x_num_tmp_i1=tf.gather(x_num_tmp_i1,pos_sample_indexes[local_pos_index,1],axis=1)

                        # positive local region 'j' in feature map from image x_a2_i
                        # fetch x and y coordinates of local region
                        x_num_tmp_i2=tf.gather(x_num_i2,pos_sample_indexes[local_pos_index,0],axis=1)
                        x_num_tmp_i2=tf.gather(x_num_tmp_i2,pos_sample_indexes[local_pos_index,1],axis=1)

                        # positive local region 'j' in feature map from image x_a1_j (j is a different volume to i)
                        # fetch x and y coordinates of local region
                        x_num_tmp_i3=tf.gather(x_num_i3,pos_sample_indexes[local_pos_index,0],axis=1)
                        x_num_tmp_i3=tf.gather(x_num_tmp_i3,pos_sample_indexes[local_pos_index,1],axis=1)

                        # positive local region 'j' in feature map from image x_a1_k (k is a different volume to i)
                        # fetch x and y coordinates of local region
                        x_num_tmp_i4=tf.gather(x_num_i4,pos_sample_indexes[local_pos_index,0],axis=1)
                        x_num_tmp_i4=tf.gather(x_num_tmp_i4,pos_sample_indexes[local_pos_index,1],axis=1)

                    # calculate cosine similarity score for the pair of positive local regions with index 'j' within the feature maps from images (x_a1_i,x_a2_i), (x_a1_i,x_a1_j) & (x_a1_i,x_a1_k).
                    # loss for positive pairs (x_a1_i,x_a2_i), (x_a1_i,x_a1_j) & (x_a1_i,x_a1_k) in (num_i1_i2_ss,num_i1_i3_ss,num_i1_i4_ss)
                    # Numerator loss terms of local loss
                    num_i1_i2_ss=self.cos_sim(x_num_tmp_i1,x_num_tmp_i2,temp_fac)
                    num_i1_i3_ss=self.cos_sim(x_num_tmp_i1,x_num_tmp_i3,temp_fac)
                    num_i1_i4_ss=self.cos_sim(x_num_tmp_i1,x_num_tmp_i4,temp_fac)

                    # Negative local regions as per the chosen positive local region at index 'j'
                    neg_samples_index_list = np.squeeze(neg_sample_indexes[local_pos_index])
                    #len_k_pts=len(den_i1_l)
                    #len_k_pts=den_i1_l.shape[0]
                    if(no_of_neg_regs_override==4):
                        no_of_neg_pts=neg_samples_index_list.shape[0] - 1
                    elif(no_of_neg_regs_override==3):
                        no_of_neg_pts=neg_samples_index_list.shape[0] - 2
                    else:
                        no_of_neg_pts=neg_samples_index_list.shape[0]
                    #print('fin den_i1',den_i1_l.shape,den_i1_l[2,0],den_i1_l[2,1])

                    # Denominator loss terms of local loss
                    den_i1_ss,den_i2_ss,den_i3_ss=0,0,0
                    #sample -ve patches from same image and its augmented versions
                    for local_neg_index in range(0,no_of_neg_pts,1):
                        if(local_reg_size==1):
                            #negative local regions in feature map (f_a1_i) from image (x_a1_i)
                            x_d_tmp_i1=tf.gather(x_num_i1,[neg_samples_index_list[local_neg_index,0],neg_samples_index_list[local_neg_index,0]+1,neg_samples_index_list[local_neg_index,0]+2],axis=1)
                            x_d_tmp_i1=tf.gather(x_d_tmp_i1,[neg_samples_index_list[local_neg_index,1],neg_samples_index_list[local_neg_index,1]+1,neg_samples_index_list[local_neg_index,1]+2],axis=2)
                            x_d_i1_flat = tf.layers.flatten(inputs=x_d_tmp_i1)
                            if(wgt_en==1):
                                x_den_tmp_i1=tf.layers.dense(inputs=x_d_i1_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE )
                            else:
                                x_den_tmp_i1=x_d_i1_flat

                            #negative local regions in feature map (f_a2_i) from image (x_a2_i)
                            x_d_tmp_i2=tf.gather(x_num_i2,[neg_samples_index_list[local_neg_index,0],neg_samples_index_list[local_neg_index,0]+1,neg_samples_index_list[local_neg_index,0]+2],axis=1)
                            x_d_tmp_i2=tf.gather(x_d_tmp_i2,[neg_samples_index_list[local_neg_index,1],neg_samples_index_list[local_neg_index,1]+1,neg_samples_index_list[local_neg_index,1]+2],axis=2)
                            x_d_i2_flat = tf.layers.flatten(inputs=x_d_tmp_i2)
                            if(wgt_en==1):
                                x_den_tmp_i2=tf.layers.dense(inputs=x_d_i2_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE )
                            else:
                                x_den_tmp_i2=x_d_i2_flat

                            #negative local regions in feature map (f_a1_j) from image (x_a1_j)
                            x_d_tmp_i3=tf.gather(x_num_i3,[neg_samples_index_list[local_neg_index,0],neg_samples_index_list[local_neg_index,0]+1,neg_samples_index_list[local_neg_index,0]+2],axis=1)
                            x_d_tmp_i3=tf.gather(x_d_tmp_i3,[neg_samples_index_list[local_neg_index,1],neg_samples_index_list[local_neg_index,1]+1,neg_samples_index_list[local_neg_index,1]+2],axis=2)
                            x_d_i3_flat = tf.layers.flatten(inputs=x_d_tmp_i3)
                            if(wgt_en==1):
                                x_den_tmp_i3=tf.layers.dense(inputs=x_d_i3_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE )
                            else:
                                x_den_tmp_i3=x_d_i3_flat

                            #negative local regions in feature map (f_a1_k) from image (x_a1_k)
                            x_d_tmp_i4=tf.gather(x_num_i4,[neg_samples_index_list[local_neg_index,0],neg_samples_index_list[local_neg_index,0]+1,neg_samples_index_list[local_neg_index,0]+2],axis=1)
                            x_d_tmp_i4=tf.gather(x_d_tmp_i4,[neg_samples_index_list[local_neg_index,1],neg_samples_index_list[local_neg_index,1]+1,neg_samples_index_list[local_neg_index,1]+2],axis=2)
                            x_d_i4_flat = tf.layers.flatten(inputs=x_d_tmp_i4)
                            if(wgt_en==1):
                                x_den_tmp_i4=tf.layers.dense(inputs=x_d_i4_flat, units=128, name='seg_pred', activation=None, use_bias=False, reuse=tf.AUTO_REUSE )
                            else:
                                x_den_tmp_i4=x_d_i4_flat
                        else:
                            #negative local regions in feature map (f_a1_i) from image (x_a1_i)
                            x_den_tmp_i1=tf.gather(x_num_i1,neg_samples_index_list[local_neg_index,0],axis=1)
                            x_den_tmp_i1=tf.gather(x_den_tmp_i1,neg_samples_index_list[local_neg_index,1],axis=1)

                            #negative local regions in feature map (f_a2_i) from image (x_a2_i)
                            x_den_tmp_i2=tf.gather(x_num_i2,neg_samples_index_list[local_neg_index,0],axis=1)
                            x_den_tmp_i2=tf.gather(x_den_tmp_i2,neg_samples_index_list[local_neg_index,1],axis=1)

                            #negative local regions in feature map (f_a1_j) from image (x_a1_j)
                            x_den_tmp_i3=tf.gather(x_num_i3,neg_samples_index_list[local_neg_index,0],axis=1)
                            x_den_tmp_i3=tf.gather(x_den_tmp_i3,neg_samples_index_list[local_neg_index,1],axis=1)

                            #negative local regions in feature map (f_a1_k) from image (x_a1_k)
                            x_den_tmp_i4=tf.gather(x_num_i4,neg_samples_index_list[local_neg_index,0],axis=1)
                            x_den_tmp_i4=tf.gather(x_den_tmp_i4,neg_samples_index_list[local_neg_index,1],axis=1)

                        # cosine score b/w local region of feature map (f_a1_i) vs other local regions within the same feature map (f_a1_i)
                        #cosine score b/w patch1 of img1a vs other patches from img1a
                        den_i1_ss=den_i1_ss+tf.exp(self.cos_sim(x_num_tmp_i1,x_den_tmp_i1,temp_fac))
                        # cosine score b/w local region of feature map (f_a1_i) vs other local regions from another feature map (f_a2_i) which is augmented version of x_i
                        #cosine score b/w patch1 of img1a vs other patches from img1b
                        den_i1_ss=den_i1_ss+tf.exp(self.cos_sim(x_num_tmp_i1,x_den_tmp_i2,temp_fac))

                        # cosine score b/w local region of feature map (f_a1_i) vs other local regions within the same feature map (f_a1_i)
                        #cosine score b/w patch1 of img1a vs other patches from img1a
                        den_i2_ss=den_i2_ss+tf.exp(self.cos_sim(x_num_tmp_i1,x_den_tmp_i1,temp_fac))
                        # cosine score b/w local region of feature map (f_a1_i) vs other local regions in a feature map (f_a1_j) from a diffrent volume 'j' compared to i.
                        #cosine score b/w patch1 of img1a vs other patches from img2a
                        den_i2_ss=den_i2_ss+tf.exp(self.cos_sim(x_num_tmp_i1,x_den_tmp_i3,temp_fac))

                        # cosine score b/w local region of feature map (f_a1_i) vs other local regions within the same feature map (f_a1_i)
                        #cosine score b/w patch1 of img1a vs other patches from img1a
                        den_i3_ss=den_i3_ss+tf.exp(self.cos_sim(x_num_tmp_i1,x_den_tmp_i1,temp_fac))
                        # cosine score b/w local region of feature map (f_a1_i) vs other local regions in a feature map (f_a1_k) from a diffrent volume 'k' compared to i.
                        #cosine score b/w patch1 of img1a vs other patches from img3a
                        den_i3_ss=den_i3_ss+tf.exp(self.cos_sim(x_num_tmp_i1,x_den_tmp_i4,temp_fac))

                    #local loss from feature map f_a1_i and f_a2_i
                    #loss from img-i vs i+12 (1a vs 1b)
                    local_loss=local_loss-tf.log(tf.math.reduce_sum(tf.exp(num_i1_i2_ss))/(tf.math.reduce_sum(tf.exp(num_i1_i2_ss))+tf.math.reduce_sum(den_i1_ss)))
                    # local loss from feature map f_a1_i and f_a1_j
                    #loss from img-i vs i+4 (1a vs 2a)
                    local_loss=local_loss-tf.log(tf.math.reduce_sum(tf.exp(num_i1_i3_ss))/(tf.math.reduce_sum(tf.exp(num_i1_i3_ss))+tf.math.reduce_sum(den_i2_ss)))
                    # local loss from feature map f_a1_i and f_a1_k
                    #loss from img-i vs i+8 (1a vs 3a)
                    local_loss=local_loss-tf.log(tf.math.reduce_sum(tf.exp(num_i1_i4_ss))/(tf.math.reduce_sum(tf.exp(num_i1_i4_ss))+tf.math.reduce_sum(den_i3_ss)))

            local_loss=local_loss/no_of_local_regions

        net_local_loss=local_loss/bs

        # var list of decoder network (including g_2 - small network)
        dec_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'dec_' in var_name: dec_net_vars.append(v)
            if 'seg_' in var_name: dec_net_vars.append(v)
            if 'reg_' in var_name: dec_net_vars.append(v)
        #print('dec_net_vars',dec_net_vars)

        if(inf==0):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                cost_net=tf.reduce_mean(net_local_loss)

                optimizer_unet_dec = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(cost_net,var_list=dec_net_vars)
                #optimizer_unet_all = tf.train.AdamOptimizer(learning_rate=learn_rate_seg).minimize(cost_net,var_list=all_net_vars)
                #'optimizer_unet_all':optimizer_unet_all,

            # Merge all the summaries and write them out to logs
            seg_summary = tf.summary.scalar('cost_net', tf.reduce_mean(cost_net))
            train_summary = tf.summary.merge([seg_summary])

            val_totalc = tf.placeholder(tf.float32, shape=[], name='val_totalc')
            val_totalc_sum= tf.summary.scalar('val_totalc_', val_totalc)
            val_summary = tf.summary.merge([val_totalc_sum])

        if(inf==1):
            return {'x':x, 'train_phase':train_phase,'y_pred':y_fin}
        else:
            return {'x':x, 'train_phase':train_phase, 'reg_cost':cost_net, 'optimizer_unet_dec':optimizer_unet_dec,\
                    'train_summary':train_summary, 'y_pred':y_fin,'val_totalc':val_totalc, 'val_summary':val_summary}

