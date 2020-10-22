import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import nibabel as nib

#to make directories
import pathlib
from skimage import transform

class f1_utilsObj:
    def __init__(self,cfg,dt,override_num_classes=0):
        #print('f1 utils init')
        self.img_size_x=cfg.img_size_x
        self.img_size_y=cfg.img_size_y
        self.batch_size=cfg.batch_size_ft
        self.num_classes=cfg.num_classes
        self.num_channels=cfg.num_channels
        self.interp_val = cfg.interp_val
        self.target_resolution=cfg.target_resolution
        self.data_path_tr=cfg.data_path_tr
        self.val_step_update=cfg.val_step_update
        self.dt=dt
        self.mtask_bs=cfg.mtask_bs
        if(override_num_classes==1):
            self.num_classes=2

    def calc_pred_mask_batchwise(self, sess, ae, labeled_data_imgs,axis_no=0):
        '''
        To compute the predicted segmentation for an input 3D volume by inferring a batch of 2D slices at a time (faster than inferring 1 2D slice at a time)
        input params:
            sess: current session
            ae: graph name
            labeled_data_imgs: input 3D volume
        returns:
            seg_pred: predicted segmentation mask of 3D volume
        '''
        total_slices = labeled_data_imgs.shape[axis_no]
        img_size_x=labeled_data_imgs.shape[1]
        img_size_y=labeled_data_imgs.shape[2]
        #print('label data',labeled_data_imgs.shape)
        bs=self.mtask_bs
        for slice_no in range(0,total_slices,bs):
            if((slice_no+bs)>total_slices):
                n_slices=total_slices-slice_no
                img_test_batch = np.reshape(labeled_data_imgs[slice_no:slice_no+n_slices], (n_slices, img_size_x, img_size_y, self.num_channels))
            else:
                img_test_batch = np.reshape(labeled_data_imgs[slice_no:slice_no+bs], (bs, img_size_x, img_size_y, self.num_channels))

            seg_pred = sess.run(ae['y_pred'], feed_dict={ae['x']: img_test_batch, ae['train_phase']: False})

            if((slice_no+bs)>total_slices):
                seg_pred=seg_pred[0:n_slices]
            else:
                seg_pred=seg_pred[0:self.mtask_bs]

            # Merging predicted labels of slices(2D) of test image into one volume(3D) of predicted labels
            if (slice_no == 0):
                mergedlist_y_pred=seg_pred
            else:
                mergedlist_y_pred = np.concatenate((mergedlist_y_pred, seg_pred), axis=0)

        #print('merge',seg_pred.shape, mergedlist_y_pred.shape)

        return mergedlist_y_pred

    def calc_val_loss_batchwise(self, sess, ae, labeled_data_imgs,labels,en_1hot,axis_no=0):
        '''
        To compute the net loss for an input 3D volume using batch-wise inference of 2D slices
        input params:
            sess: current session
            ae: graph name
            labeled_data_imgs: input 3D volume
            labels: 3D mask
            en_1hot: to indicate if labels are in 1-hot encoding form / not
        returns:
            total_cost_val: total loss of input 3D volume
        '''
        total_slices = labeled_data_imgs.shape[axis_no]
        img_size_x=labeled_data_imgs.shape[1]
        img_size_y=labeled_data_imgs.shape[2]
        bs=self.mtask_bs
        total_cost_val=0
        for slice_no in range(0,total_slices,bs):
            if((slice_no+bs)>total_slices):
                n_slices=total_slices-slice_no
                img_test_batch = np.reshape(labeled_data_imgs[slice_no:slice_no+n_slices], (n_slices, img_size_x, img_size_y, self.num_channels))
                if(en_1hot==1):
                    label_test_batch = np.reshape(labels[slice_no:slice_no+n_slices], (n_slices, img_size_x, img_size_y, self.num_classes))
                else:
                    label_test_batch = np.reshape(labels[slice_no:slice_no+n_slices], (n_slices, img_size_x, img_size_y))
            else:
                img_test_batch = np.reshape(labeled_data_imgs[slice_no:slice_no+bs], (bs, img_size_x, img_size_y, self.num_channels))
                if(en_1hot==1):
                    label_test_batch = np.reshape(labels[slice_no:slice_no+bs], (bs, img_size_x, img_size_y, self.num_classes))
                else:
                    label_test_batch = np.reshape(labels[slice_no:slice_no+bs], (bs, img_size_x, img_size_y))

            cost_val=sess.run(ae['actual_cost'], feed_dict={ae['x']: img_test_batch, ae['y_l']: label_test_batch,ae['train_phase']: False})
            total_cost_val=total_cost_val+np.mean(cost_val)

        return total_cost_val

    def reshape_img_and_f1_score(self, predicted_img_arr, gt_mask, pixel_size):
        '''
        To reshape image into the target resolution and then compute the f1 score w.r.t ground truth mask
        input params:
            predicted_img_arr: predicted segmentation mask that is computed over the re-sampled and cropped input image
            gt_mask: ground truth mask in native image resolution
            pixel_size: native image resolution
        returns:
            predictions_mask: predictions mask in native resolution (re-sampled and cropped/zeros append as per size requirements)
            f1_val: f1 score over predicted segmentation masks vs ground truth
        '''
        nx,ny= self.img_size_x,self.img_size_y

        scale_vector = (pixel_size[0] / self.target_resolution[0], pixel_size[1] / self.target_resolution[1])
        mask_rescaled = transform.rescale(gt_mask[:, :, 0], scale_vector, order=0, preserve_range=True, mode='constant')
        x, y = mask_rescaled.shape[0],mask_rescaled.shape[1]

        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2

        total_slices = predicted_img_arr.shape[0]
        predictions_mask = np.zeros((gt_mask.shape[0],gt_mask.shape[1],total_slices))

        for slice_no in range(total_slices):
            # ASSEMBLE BACK THE SLICES
            slice_predictions = np.zeros((x,y,self.num_classes))
            predicted_img=predicted_img_arr[slice_no,:,:,:]
            # insert cropped region into original image again
            if x > nx and y > ny:
                slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = predicted_img
            else:
                if x <= nx and y > ny:
                    slice_predictions[:, y_s:y_s+ny,:] = predicted_img[x_c:x_c+ x, :,:]
                elif x > nx and y <= ny:
                    slice_predictions[x_s:x_s + nx, :,:] = predicted_img[:, y_c:y_c + y,:]
                else:
                    slice_predictions[:, :,:] = predicted_img[x_c:x_c+ x, y_c:y_c + y,:]

            # RESCALING ON THE LOGITS
            prediction = transform.resize(slice_predictions,
                                              (gt_mask.shape[0], gt_mask.shape[1], self.num_classes),
                                              order=1,
                                              preserve_range=True,
                                              mode='constant')
            #print("b",prediction.shape)
            prediction = np.uint16(np.argmax(prediction, axis=-1))

            predictions_mask[:,:,slice_no]=prediction

        #Calculate DSC / F1 score
        f1_val = self.calc_f1_score(predictions_mask,gt_mask)

        return predictions_mask,f1_val

    def calc_f1_score(self,predictions_mask,gt_mask):
        '''
        to compute f1/dice score
        input params:
            predictions_arr: predicted segmentation mask
            mask: ground truth mask
        returns:
            f1_val: f1/dice score
        '''
        y_pred= predictions_mask.flatten()
        y_true= gt_mask.flatten()

        f1_val= f1_score(y_true, y_pred, average=None)

        return f1_val

    def plot_predicted_seg_ss(self, test_data_img,test_data_labels,predicted_labels,save_dir,test_id):
        '''
        To plot the original image, ground truth mask and predicted mask
        input params:
            test_data_img: test image to be plotted
            test_data_labels: test image GT mask to be plotted
            predicted_labels: predicted mask of the test image
            save_dir: directory where to save the plot
            test_id: patient id number of the dataset
        returns:
            None
        '''
        n_examples=4
        factor=int(test_data_img.shape[2]/n_examples)-1
        fig, axs = plt.subplots(3, n_examples, figsize=(10, 10))
        fig.suptitle('Predicted Seg',fontsize=10)
        for example_i in range(n_examples):
            if(example_i==0):
                axs[0][0].set_title('test image')
                axs[1][0].set_title('ground truth mask')
                axs[2][0].set_title('predicted mask')

            axs[0][example_i].imshow(test_data_img[:,:,(example_i+1)*factor],cmap='gray')
            axs[1][example_i].imshow(test_data_labels[:,:,(example_i+1)*factor],cmap='Paired',clim=(0,self.num_classes))
            axs[2][example_i].imshow(np.squeeze(predicted_labels[:,:,(example_i+1)*factor]),cmap='Paired',clim=(0,self.num_classes))
            axs[0][example_i].axis('off')
            axs[1][example_i].axis('off')
            axs[2][example_i].axis('off')

        savefile_name=str(save_dir)+'tst'+str(test_id)+'_predicted_segmentation_masks.png'
        fig.savefig(savefile_name)
        plt.close('all')

    def plt_seg_loss(self,seg_loss_list,save_dir_tmp,title_str='_',plt_name='seg_loss',ep_no=10000):
        '''
        Plot figures for the segmentation loss and F1/Dsc score lists over all epochs
        input params:
            seg_loss_list: input list with segmentation loss and Dsc score values over all epochs
            save_dir_tmp: directory to save the figure
        returns:
            None
        '''
        plt.figure()
        if('seg_loss' in plt_name):
            x=np.arange(0,len(seg_loss_list[0])*self.val_step_update,self.val_step_update)
            plt.title('seg.loss - '+str(title_str))
            #plt.ylabel('dice score loss')
            plt.ylabel('loss val')
            plt.xlabel('# epochs')
            if(len(seg_loss_list)==2):
                plt.plot(x,seg_loss_list[0],'g-.',label='tr loss')
                plt.plot(x,seg_loss_list[1],'b-',label='val loss')
            else:
                plt.plot(x, seg_loss_list[0], 'g-.', label='tr loss')
            plt.legend(loc=1)
        elif('dsc_score' in plt_name):
            x=np.arange(0,len(seg_loss_list[0])*self.val_step_update,self.val_step_update)
            plt.title('DSC score - '+str(title_str))
            plt.ylabel('mean dsc')
            plt.xlabel('# epochs')
            #plt.plot(x,seg_loss_list)
            if (len(seg_loss_list) == 2):
                plt.plot(x,seg_loss_list[0],'g-.',label='tr dsc/f1')
                plt.plot(x,seg_loss_list[1],'b-',label='val dsc/f1')
            else:
                plt.plot(x, seg_loss_list[0], 'g-.', label='tr dsc/f1')
            plt.legend(loc=1)
            #plt.show()
        plt.savefig(save_dir_tmp+'/'+str(plt_name)+'_ep_'+str(ep_no)+'.png')
        plt.close('all')

    def track_val_dsc(self,sess,ae,ae_1hot,saver,mean_f1_val_prev,threshold_f1,best_model_dir,val_list,val_img_crop,\
                  val_label_crop,val_label_orig,pixel_val_list,checkpoint_filename,epoch_i,en_1hot_val=1):
        '''
            Save the model with best DSC of Validation Volumes compared to previous all iterations' best DSC value.
        inputs params:
            sess:current session
            ae: graph name
            ae_1hot: graph to compute 1-hot encoding of labels
            saver: to save the checkpoint file with best DSC over val images
            mean_f1_val_prev: best DSC value until this iteration from iteration 0.
            threshold_f1: small threshold value to check if DSC improvement in current iteration is significant/not
            best_model_dir: directory to save the best DSC model
            val_list: validation volumes ID list
            val_img_crop: validation images list in target resolution and cropped to defined fixed size
            val_label_crop: validation images' masks list in target resolution and cropped dimensions
            val_label_orig: validation images' masks list in native resolution (used to compute DSC)
            pixel_val_list: list of pixel resolution values of all validation images
            checkpoint_filename: checkpoint filename used to save the model
            epoch_i: iteration number
            en_1hot_val: to indicate if labels are in 1-hot encoding format / not.
        return:
            mean_f1_val_prev: updated best DSC value till this iteration.
                              If current DSC is higher than earlier DSC's then its set to this value else use the previous best DSC until this iteration.
            mp_best: best DSC model checkpoint file
            mean_total_cost_val: mean segmentation loss over validation volumes
            mean_f1: mean DSC/F1 value over validation volumes
        '''

        mean_f1_arr=[]
        f1_arr=[]
        mean_total_cost_val=0
        # Compute segmentation mask and dice_score for each validation volume
        for val_id_no in range(0,len(val_list)):
            val_img_crop_tmp=val_img_crop[val_id_no]
            val_label_crop_tmp=val_label_crop[val_id_no]
            val_label_orig_tmp=val_label_orig[val_id_no]
            pixel_size_val=pixel_val_list[val_id_no]

            # infer segmentation mask for each validation volume
            pred_sf_mask = self.calc_pred_mask_batchwise(sess, ae, val_img_crop_tmp)

            if(en_1hot_val==1):
                ld_label_batch_1hot = sess.run(ae_1hot['y_tmp_1hot'],feed_dict={ae_1hot['y_tmp']:val_label_crop_tmp})
                total_cost_val=self.calc_val_loss_batchwise(sess, ae, val_img_crop_tmp,ld_label_batch_1hot,en_1hot_val)
            else:
                total_cost_val=self.calc_val_loss_batchwise(sess, ae, val_img_crop_tmp,val_label_crop_tmp,en_1hot_val)

            mean_total_cost_val=mean_total_cost_val+np.mean(total_cost_val)
            #print('mask shapes,',pred_sf_mask.shape,val_label_orig_tmp.shape)

            # resize segmentation mask to original dimensions as ground truth to calcluate the DSC/F1 - score.
            re_pred_mask_sys,f1_val = self.reshape_img_and_f1_score(pred_sf_mask, val_label_orig_tmp, pixel_size_val)

            #concatenate dice scores of each val image
            mean_f1_arr.append(np.mean(f1_val[1:self.num_classes]))
            f1_arr.append(f1_val[1:self.num_classes])

        #avg mean over 2 val subjects
        mean_f1_arr=np.asarray(mean_f1_arr)
        mean_f1=np.mean(mean_f1_arr)
        mean_total_cost_val=mean_total_cost_val/len(val_list)

        mp = str(best_model_dir) + str(checkpoint_filename) + '_epochs_' + str(epoch_i) + ".ckpt"

        if (mean_f1-mean_f1_val_prev>threshold_f1):
            print("prev f1_val; present_f1_val", mean_f1_val_prev, mean_f1, mean_f1_arr)
            mean_f1_val_prev = mean_f1
            # to save the best model with maximum dice score over the entire n_epochs
            print("best model saved at epoch no. ", epoch_i)
            mp_best = str(best_model_dir) + str(checkpoint_filename) + '_best_model_epoch_' + str(epoch_i) + ".ckpt"
            saver.save(sess, mp_best)
        try:
            mp_best
        except NameError:
            mp_best=mp

        return mean_f1_val_prev,mp_best,mean_total_cost_val,mean_f1


    def test_set_predictions(self, val_list,sess,ae,dt,orig_img_dt,save_dir_tmp):
        '''
        To estimate the predicted segmentation masks of test images and compute their Dice scores (DSC) and plot the predicted segmentations.
        input params:
            val_list: list of patient test ids
            sess: current session
            ae: current model graph
            dt : dataloader class
            orig_img_dt: dataloader of <specific_dataset>
            save_dir_tmp: save directory for the predictions of test images
            struct_name: list of structures to segment. For example for cardiac dataset ACDC: its Right ventricle (RV), myocardium (Myo), left ventricle (LV).
        returns:
            None
        '''
        count=0
        mean_20subjs_dsc=[]

        # Load each test image and infer the predicted segmentation mask and compute Dice scores
        for test_id in val_list:
            test_id_l=[test_id]

            #load image,label pairs and process it to chosen resolution and dimensions
            img_sys,label_sys,pixel_size,affine_tst= orig_img_dt(test_id_l,ret_affine=1)
            cropped_img_sys,cropped_mask_sys = dt.preprocess_data(img_sys, label_sys, pixel_size)
            img_crop_re=np.swapaxes(cropped_img_sys,1,2)
            img_crop_re=np.swapaxes(img_crop_re,0,1)

            # Calc dice score / F1 score and predicted segmentation
            pred_sf_mask = self.calc_pred_mask_batchwise(sess, ae, img_crop_re)
            re_pred_mask_sys,dsc_val = self.reshape_img_and_f1_score(pred_sf_mask, label_sys, pixel_size)
            print('test id, mean DSC', test_id, dsc_val)

            # Make directory for the test image with id number
            seg_model_dir=str(save_dir_tmp)+str(test_id)+'/'
            pathlib.Path(seg_model_dir).mkdir(parents=True, exist_ok=True)

            # Save the computed Dice score value
            savefile_name = str(seg_model_dir)+'dsc_score_test_id_'+str(test_id)+'.txt'
            np.savetxt(savefile_name, dsc_val, fmt='%s')

            # Plot some slices of the image, GT mask & predicted mask for test image
            #self.plot_predicted_seg_ss(img_sys,label_sys,re_pred_mask_sys,seg_model_dir,test_id)

            # Save the predicted segmentation mask in nifti format
            array_img = nib.Nifti1Image(re_pred_mask_sys.astype(np.int16), affine_tst)
            pred_filename = str(seg_model_dir)+'pred_seg_id_'+str(test_id)+'.nii.gz'
            nib.save(array_img, pred_filename)

            dsc_tmp=np.reshape(dsc_val[1:self.num_classes], (1, self.num_classes - 1))
            mean_20subjs_dsc.append(np.mean(dsc_tmp))

            if(count==0):
                dsc_all=dsc_tmp
                count=1
            else:
                dsc_all=np.concatenate((dsc_all, dsc_tmp))

        #Mean Dice of all structures of each subject
        filename_save=str(save_dir_tmp)+'mean_dsc_of_each_subj.txt'
        np.savetxt(filename_save,mean_20subjs_dsc,fmt='%s')

        # Net Mean Dice over all test subjects
        filename_save = str(save_dir_tmp) + 'net_mean_dsc_over_all_subjs.txt'
        print('mean dsc', np.mean(mean_20subjs_dsc))
        tmp = []
        tmp.append(np.mean(mean_20subjs_dsc))
        np.savetxt(filename_save, tmp, fmt='%s')

        #for DSC of each structure
        # val_list_mean=[]
        #
        # for i in range(0,self.num_classes-1):
        #     dsc=dsc_all[:,i]
        #     #DSC
        #     val_list_mean.append(round(np.mean(dsc), 3))
        #     filename_save=str(save_dir_tmp)+str(struct_name[i])+'_dsc_of_each_subj.txt'
        #     np.savetxt(filename_save,dsc,fmt='%s')

        #filename_save = str(save_dir_tmp) + 'per_structure_mean_dsc_over_all_subjs.txt'
        #np.savetxt(filename_save, val_list_mean, fmt='%s')
