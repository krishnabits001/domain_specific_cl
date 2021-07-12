**Contrastive learning of global and local features for medical image segmentation with limited annotations** <br/>

The code is for the article "Contrastive learning of global and local features for medical image segmentation with limited annotations" which got accepted as an Oral presentation at NeurIPS 2020 (33rd international conference on Neural Information Processing Systems). With the proposed pre-training method using Contrastive learning, we get competitive segmentation performance with just 2 labeled training volumes compared to a benchmark that is trained with many labeled volumes.<br/>
https://arxiv.org/abs/2006.10511 <br/>

**Observations / Conclusions:** <br/>
1) For medical image segmentation, the proposed contrastive pre-training strategy incorporating domain knowledge present naturally across medical volumes yields better performance than baseline, other pre-training methods, semi-supervised, and data augmentation methods.
2) Proposed local contrastive loss, an extension of global loss, provides an additional boost in performance by learning distinctive local-level representation to distinguish between neighbouring regions.
3) The proposed pre-training strategy is complementary to semi-supervised and data augmentation methods. Combining them yields a further boost in accuracy.

**Authors:** <br/>
Krishna Chaitanya ([email](mailto:krishna.chaitanya@vision.ee.ethz.ch)),<br/>
Ertunc Erdil,<br/>
Neerav Karani,<br/>
Ender Konukoglu.<br/>

**Requirements:** <br/>
Python 3.6.1,<br/>
Tensorflow 1.12.0,<br/>
rest of the requirements are mentioned in the "requirements.txt" file. <br/>

I)  To clone the git repository.<br/>
git clone https://github.com/krishnabits001/domain_specific_dl.git <br/>

II) Install python, required packages and tensorflow.<br/>
Then, install python packages required using below command or the packages mentioned in the file.<br/>
pip install -r requirements.txt <br/>

To install tensorflow <br/>
pip install tensorflow-gpu=1.12.0 <br/>

III) Dataset download.<br/>
To download the ACDC Cardiac dataset, check the website :<br/>
https://www.creatis.insa-lyon.fr/Challenge/acdc. <br/>

To download the Medical Decathlon Prostate dataset, check the website :<br/>
http://medicaldecathlon.com/

To download the MMWHS Cardiac dataset, check the website :<br/>
http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/
 
All the images were bias corrected using N4 algorithm with a threshold value of 0.001. For more details, refer to the "N4_bias_correction.py" file in scripts.<br/>
Image and label pairs are re-sampled (to chosen target resolution) and cropped/zero-padded to a fixed size using "create_cropped_imgs.py" file. <br/>

IV) Train the models.<br/>
Below commands are an example for ACDC dataset.<br/> 
The models need to be trained sequentially as follows (check "train_model/pretrain_and_fine_tune_script.sh" script for commands)<br/>
Steps :<br/>
1) Step 1: To pre-train the encoder with global loss by incorporating proposed domain knowledge when defining positive and negative pairs.<br/>
cd train_model/ <br/>
python pretr_encoder_global_contrastive_loss.py --dataset=acdc --no_of_tr_imgs=tr52 --global_loss_exp_no=2 --n_parts=4 --temp_fac=0.1 --bt_size=12

2) Step 2: After step 1, we pre-train the decoder with proposed local loss to aid segmentation task by learning distinctive local-level representations.<br/>
python pretr_decoder_local_contrastive_loss.py --dataset=acdc --no_of_tr_imgs=tr52 --pretr_no_of_tr_imgs=tr52 --local_reg_size=1 --no_of_local_regions=13 --temp_fac=0.1 --global_loss_exp_no=2 --local_loss_exp_no=0 --no_of_decoder_blocks=3 --no_of_neg_local_regions=5 --bt_size=12

3) Step 3: We use the pre-trained encoder and decoder weights as initialization and fine-tune to segmentation task using limited annotations.<br/>
python ft_pretr_encoder_decoder_net_local_loss.py --dataset=acdc --pretr_no_of_tr_imgs=tr52 --local_reg_size=1 --no_of_local_regions=13 --temp_fac=0.1 --global_loss_exp_no=2 --local_loss_exp_no=0 --no_of_decoder_blocks=3 --no_of_neg_local_regions=5 --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --ver=0 

To train the baseline with affine and random deformations & intensity transformations for comparison, use the below code file.<br/>
cd train_model/ <br/>
python tr_baseline.py --dataset=acdc --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --ver=0

V) Config files contents.<br/>
One can modify the contents of the below 2 config files to run the required experiments.<br/>
experiment_init directory contains 2 files.<br/>
Example for ACDC dataset:<br/>
1) init_acdc.py <br/>
--> contains the config details like target resolution, image dimensions, data path where the dataset is stored and path to save the trained models.<br/>
2) data_cfg_acdc.py <br/>
--> contains an example of data config details where one can set the patient ids which they want to use as train, validation and test images.<br/>


**Bibtex citation:** 

	@article{chaitanya2020contrastive,
	  title={Contrastive learning of global and local features for medical image segmentation with limited annotations},
	  author={Chaitanya, Krishna and Erdil, Ertunc and Karani, Neerav and Konukoglu, Ender},
	  journal={Advances in Neural Information Processing Systems},
	  volume={33},
	  year={2020}
	}
