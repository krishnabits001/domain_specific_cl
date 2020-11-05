################################################################
# Definitions required for CNN graph
################################################################
#Filter size at different depth level of CNN in order
fs=3
#Interpolation type for upsampling layers in decoder
interp_val=1 # 0 - bilinear interpolation; 1- nearest neighbour interpolation
################################################################

################################################################
# data dimensions, num of classes and resolution
################################################################
#Name of dataset
dataset_name='prostate_md'
#Image Dimensions
img_size_x = 192
img_size_y = 192
# Images dimensions in one-dimensional array
img_size_flat = img_size_x * img_size_y
# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1
# Number of label classes : # 0-background, 1-cz, 2-pz
num_classes=3
#Image dimensions in x and y directions
size=(img_size_x,img_size_y)
#target image resolution
target_resolution=(0.625,0.625)
#label class name
class_name='cz'
#class_name='pz'
################################################################
#data paths
################################################################
#validation_update_step to save values
val_step_update=50
#base directory of the code
base_dir='/usr/bmicnas01/data-biwi-01/krishnch/projects/self_tr/contrastive_lr/git_may24/'
srt_dir='/usr/bmicnas01/data-biwi-01/krishnch/projects/self_tr/contrastive_lr/git_may24/'

#Path to data in original dimensions in default resolution
data_path_tr='/usr/bmicnas01/data-biwi-01/krishnch/datasets/prostate_md/orig/'

#Path to data in cropped dimensions in target resolution (saved apriori)
data_path_tr_cropped='/usr/bmicnas01/data-biwi-01/krishnch/datasets/prostate_md/orig_cropped/'
################################################################

################################################################
#training hyper-parameters
################################################################
#learning rate for segmentation net
lr=0.001
#pre-training batch size
mtask_bs=20
#batch_size for fine-tuning on segmentation task
batch_size_ft=10
#foreground structures names to segment
struct_name=['pz','cz']
