# To perform N4 Bias Correction on input image 

import numpy as np
import SimpleITK as sitk
import sys 
import os

# parameters for ACDC
threshold_value = 0.001
n_fitting_levels = 4
n_iters = 50

# Input and output image path 
in_file_name='<input_path>/img.nii.gz'
out_file_name='<output_path>/img_bias_corr.nii.gz'

# Read the image
inputImage = sitk.ReadImage(in_file_name)
inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

# Apply N4 bias correction
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrector.SetConvergenceThreshold(threshold_value)
corrector.SetMaximumNumberOfIterations([int(n_iters)] * n_fitting_levels)

#Save the bias corrected output file
output = corrector.Execute(inputImage)
sitk.WriteImage(output, out_file_name)

