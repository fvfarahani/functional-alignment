#!pip install h5py nibabel pprocess pymvpa2 nilearn duecredit scikit-network
#!pip install networkx 
#!pip install git+https://github.com/FIU-Neuro/brainconn#egg=brainconn
#!pip install netneurotools tqdm

#%matplotlib inline
import nibabel as nib
import nilearn.plotting as plotting
import numpy as np
import matplotlib.pyplot as plt
import hcp_utils as hcp
import os
from scipy.spatial.distance import pdist, cdist
from scipy.stats.stats import pearsonr
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.zscore import zscore
from mvpa2.misc.surfing.queryengine import SurfaceQueryEngine
from mvpa2.algorithms.searchlight_hyperalignment import SearchlightHyperalignment
from mvpa2.algorithms.hyperalignment import Hyperalignment
from mvpa2.base.hdf5 import h5save, h5load
# Alternatively, all those above can be imported using
# from mvpa2.suite import *
from mvpa2.support.nibabel.surf import read as read_surface
from mvpa2.base.dataset import vstack, hstack
import timeit

from mvpa2.datasets.mri import fmri_dataset 
from mvpa2.misc.neighborhood import IndexQueryEngine, Sphere

# pip install numpy==1.16.4   -> 'numpy.testing.decorators' error
# pip install 'h5py==2.10.0' --force-reinstall  -> AttributeError: 'str' object has no attribute 'decode' -> this happens for searchlight_hyperalignemnt

#%% Loading HCP data (subjects, atlas, and targets)
# -------------------

data_path = '/dcl01/smart/data/hpc900/' 
os.chdir('/dcs05/ciprian/smart/farahani/SL-CHA') # '/dcl01/smart/data/fvfarahani/searchlight/'
"""
os.chdir('/Volumes/Elements/Hyperalignment/HCP/HCP900/')
data_path = '/Volumes/Elements/Hyperalignment/HCP/HCP900/'
!rm .DS_Store
os.chdir('/Volumes/Elements/Hyperalignment/HCP/')
"""
disks = ['disk1/', 'disk2/', 'disk3/']

subjects1 = sorted(os.listdir(data_path + disks[0]))
subjects2 = sorted(os.listdir(data_path + disks[1]))
subjects3 = sorted(os.listdir(data_path + disks[2]))
subjects = set(subjects1 + subjects2 + subjects3)

# finding the twins (both NotMZ and MZ)
import pandas as pd
#HCP = pd.read_csv('/Volumes/Elements/HCP.csv')
HCP = pd.read_csv('/dcs05/ciprian/smart/farahani/SL-CHA/HCP.csv')
NotMZ = []
MZ = []
for ID in subjects:
    var = HCP.loc[HCP['Subject'] == int(ID)]['ZygositySR'].iloc[0]
    if var == 'NotMZ':
        Father_ID = HCP.loc[HCP['Subject'] == int(ID)]['Father_ID'].iloc[0]
        Twin = HCP.loc[HCP['Father_ID'] == Father_ID].loc[HCP['ZygositySR'] == 'NotMZ']
        if len(Twin) == 2:
            if str(Twin['Subject'].iloc[0]) in subjects and str(Twin['Subject'].iloc[1]) in subjects:
                NotMZ.append(str(Twin['Subject'].iloc[1]))
    elif var == 'MZ':
        Father_ID = HCP.loc[HCP['Subject'] == int(ID)]['Father_ID'].iloc[0]
        Twin = HCP.loc[HCP['Father_ID'] == Father_ID].loc[HCP['ZygositySR'] == 'MZ']
        if len(Twin) == 2:
            if str(Twin['Subject'].iloc[0]) in subjects and str(Twin['Subject'].iloc[1]) in subjects:
                MZ.append(str(Twin['Subject'].iloc[1]))

NotMZ = set(NotMZ)
MZ = set(MZ)

# subtract from incomplete data (timeseries) and from twins
missing = set.union(set(['116120', '121820', '126931', '129432', '131621', '143527']), # REST1
                       set(['101410', '107220', '112819', '113821', '121315', '129432', '131621', '143527']), # REST2
                       set(['150019', '179548', '190132', '197449', '203721', '207628']), # REST1_LR
                       set(['128329', '129533', '142424', '145531', '146634', '150524', '179952', '193441', '197651', '200210', '201717', '208428']), # REST2_LR
                       set(['116221', '129937']), # REST2_RL 
                       set(['119833', '140420']), # REST1_LR (incomplete tps)
                       set(['119732', '183337']), # REST2 (incomplete tps)
                       set(['150423']), # REST2_LR (incomplete tps)
                       NotMZ, # non-monozygotic twins (NotMZ)
                       MZ) # monozygotic twins (MZ)

subjects = sorted(set(subjects) - missing)

#% Configurations: atlas, targets
atlas = hcp.mmp # {‘mmp’, ‘ca_parcels’, ‘ca_network’, ‘yeo7’, ‘yeo17’}
target_nmbr = len(atlas['nontrivial_ids'])-19 # 718, 379-19 
grayordinate = 91282 # L(29696) + R(29716) vertices; whole brain has 91282 grayordinates (go)
#hemi = 'left' # 'left' or 'right'

#%% Editing/Saving CIFTI

img = nib.load('/Users/Farzad/Desktop/help/My_Cifti.dtseries.nii')

data_a = img.get_fdata()
data_b = data_a * 2

new_img = nib.Cifti2Image(data_b, header=img.header,
                         nifti_header=img.nifti_header)

new_img.to_filename('/Users/Farzad/Desktop/help/My_New_Cifti.dtseries.nii')
# or -> nib.cifti2.cifti2.save(new_img, '/Users/Farzad/Desktop/help/My_New_Cifti.dtseries.nii')

#%% HCP parcellation using workbench (wb_command)
# https://github.com/ColeLab/ColeAnticevicNetPartition/blob/master/LoadParcellatedDataInPython_Example_cortexonly.py

# Script requires workbench (wb_command), in addition to the below python packages
# -----------------------------------
# Load dependencies
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from scipy import stats

"""
# make subjects directories for disc 2 (ONE TIME)
root_path = '/Volumes/Elements/Hyperalignment/HCP/HCP900/disk2/'
# disk2_list = sorted([128127,  129533,  130922,  132118,  134425,  136732,  138534,  140824,  143426,  145531,  147030,  149236,  150625,  151829,  154229,  128329,  129634,  131217,  133019,  134728,  136833,  138837,  140925,  143527,  145834,  147737,  149337,  150726,  152831,  128632,  129937,  131419,  133625,  134829,  137027,  139233,  141119,  144125,  146129,  148032,  149539,  150928,  153025,  128935,  130013,  131621,  133827,  135225,  137128,  139637,  141422,  144226,  146331,  148133,  149741,  151223,  153227,  129028,  130316,  131722,  133928,  135528,  137229,  139839,  141826,  144428,  146432,  148335,  149842,  151425,  153429,  129129,  130417,  131823,  134021,  135730,  137633,  140117,  142424,  144731,  146533,  148436,  150019,  151526,  153631,  129331,  130619,  131924,  134223,  135932,  137936,  140319,  142828,  144832,  146634,  148840,  150423,  151627,  153732,  129432,  130821,  132017,  134324,  136227,  138231,  140420,  143325,  145127,  146937,  148941,  150524,  151728,  153833])
disk2_list = [128127, 128329, 128632, 128935, 129028, 129129, 129331, 129432, 129533, 129634, 129937, 130013, 130316, 130417, 130619, 130821, 130922, 131217, 131419, 131621, 131722, 131823, 131924, 132017, 132118, 133019, 133625, 133827, 133928, 134021, 134223, 134324, 134425, 134728, 134829, 135225, 135528, 135730, 135932, 136227, 136732, 136833, 137027, 137128, 137229, 137633, 137936, 138231, 138534, 138837, 139233, 139637, 139839, 140117, 140319, 140420, 140824, 140925, 141119, 141422, 141826, 142424, 142828, 143325, 143426, 143527, 144125, 144226, 144428, 144731, 144832, 145127, 145531, 145834, 146129, 146331, 146432, 146533, 146634, 146937, 147030, 147737, 148032, 148133, 148335, 148436, 148840, 148941, 149236, 149337, 149539, 149741, 149842, 150019, 150423, 150524, 150625, 150726, 150928, 151223, 151425, 151526, 151627, 151728, 151829, 152831, 153025, 153227, 153429, 153631, 153732, 153833, 154229]
# create folders and subfolders
for folder in disk2_list:
    os.mkdir(os.path.join(root_path,str(folder)))
    new_path = root_path + str(folder) + '/'
    os.mkdir(os.path.join(new_path,'MNINonLinear'))
    new_path = new_path + 'MNINonLinear/'
    os.mkdir(os.path.join(new_path,'Results'))
    new_path = new_path + 'Results/'
    os.mkdir(os.path.join(new_path,'rfMRI_REST1_LR'))
    os.mkdir(os.path.join(new_path,'rfMRI_REST1_RL'))
    os.mkdir(os.path.join(new_path,'rfMRI_REST2_LR'))
    os.mkdir(os.path.join(new_path,'rfMRI_REST2_RL'))

# make subjects directories for disc 3 (ONE TIME)
root_path = '/Volumes/Elements/Hyperalignment/HCP/HCP900/disk3/'
#disk3_list = sorted([179245, 181232, 185442, 188448, 191841, 194140, 196346, 198855, 200614, 203721, 205826, 209127, 179346, 181636, 185846, 188549, 191942, 194645, 196750, 199150, 200917, 203923, 206222, 209228, 179548, 182032, 185947, 188751, 192035, 194746, 197348, 199251, 201111, 204016, 207123, 209329, 179952, 182436, 186141, 189349, 192136, 194847, 197449, 199453, 201414, 204319, 207426, 180129, 182739, 186444, 189450, 192439, 195041, 197550, 199655, 201515, 204420, 207628, 180432, 182840, 187143, 190031, 192540, 195445, 197651, 199958, 201717, 204521, 208024, 180735, 183034, 187345, 190132, 192641, 195647, 198249, 200008, 201818, 204622, 208125, 180836, 183337, 187547, 191033, 192843, 195849, 198350, 200109, 202113, 205119, 208226, 180937, 185139, 187850, 191336, 193239, 195950, 198451, 200210, 202719, 205220, 208327, 181131, 185341, 188347, 191437, 193441, 196144, 198653, 200311, 203418, 205725, 208428])
disk3_list = [179245, 179346, 179548, 179952, 180129, 180432, 180735, 180836, 180937, 181131, 181232, 181636, 182032, 182436, 182739, 182840, 183034, 183337, 185139, 185341, 185442, 185846, 185947, 186141, 186444, 187143, 187345, 187547, 187850, 188347, 188448, 188549, 188751, 189349, 189450, 190031, 190132, 191033, 191336, 191437, 191841, 191942, 192035, 192136, 192439, 192540, 192641, 192843, 193239, 193441, 194140, 194645, 194746, 194847, 195041, 195445, 195647, 195849, 195950, 196144, 196346, 196750, 197348, 197449, 197550, 197651, 198249, 198350, 198451, 198653, 198855, 199150, 199251, 199453, 199655, 199958, 200008, 200109, 200210, 200311, 200614, 200917, 201111, 201414, 201515, 201717, 201818, 202113, 202719, 203418, 203721, 203923, 204016, 204319, 204420, 204521, 204622, 205119, 205220, 205725, 205826, 206222, 207123, 207426, 207628, 208024, 208125, 208226, 208327, 208428, 209127, 209228, 209329]
# create folders and subfolders
for folder in disk3_list:
    os.mkdir(os.path.join(root_path,str(folder)))
    new_path = root_path + str(folder) + '/'
    os.mkdir(os.path.join(new_path,'MNINonLinear'))
    new_path = new_path + 'MNINonLinear/'
    os.mkdir(os.path.join(new_path,'Results'))
    new_path = new_path + 'Results/'
    os.mkdir(os.path.join(new_path,'rfMRI_REST1_LR'))
    os.mkdir(os.path.join(new_path,'rfMRI_REST1_RL'))
    os.mkdir(os.path.join(new_path,'rfMRI_REST2_LR'))
    os.mkdir(os.path.join(new_path,'rfMRI_REST2_RL'))
"""

"""
# remove files in the computer
import os

disk = ['sub-disk1/', 'sub-disk2/', 'sub-disk3/']
os.chdir('/Volumes/Elements/Hyperalignment/HCP/HCP900/')
data_path = '/Volumes/Elements/Hyperalignment/HCP/HCP900/'
!rm .DS_Store
subjects1 = sorted(os.listdir(data_path + disk[0]))

for subj in subjects1:
    filePath = data_path + disk[0] + subj + '/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll.dtseries.nii'.format(subj=subj)
    if os.path.exists(filePath):
        os.remove(filePath)
"""

disk = 'disk3/' # 'disk1/', 'disk2/', 'disk3/'
os.chdir('/Volumes/Elements/Hyperalignment/HCP/HCP900/')
data_path = '/Volumes/Elements/Hyperalignment/HCP/HCP900/'
!rm .DS_Store
os.chdir('/Volumes/Elements/Hyperalignment/HCP/')

#subjects1 = sorted(os.listdir(data_path + disk))
#subjects2 = sorted(os.listdir(data_path + disk))
subjects3 = sorted(os.listdir(data_path + disk))

# sub-disk1 outliers
#subjects1.remove('116120'); subjects1.remove('121820'); # only Rest1_RL
#subjects1.remove('126931'); # no data (Results)
#subjects1.remove('101410'); subjects1.remove('107220'); subjects1.remove('112819'); subjects1.remove('113821'); subjects1.remove('121315'); # no rest2 data
#subjects1.remove('116221'); # no REST2_RL
#subjects1.remove('119833'); # not similar timepoints as others [rest1_lr]
#subjects1.remove('119732'); # not similar timepoints as others [rest2_lr,rl]

# sub-disk2 outliers
#subjects2.remove('129432'); subjects2.remove('131621'); subjects2.remove('143527'); # nothing
#subjects2.remove('150019'); # REST1_LR
#subjects2.remove('128329'); subjects2.remove('129533'); subjects2.remove('142424'); subjects2.remove('145531'); subjects2.remove('146634'); subjects2.remove('150524'); # REST2_LR
#subjects2.remove('129937'); # REST2_RL
#subjects2.remove('140420'); # not similar timepoints as othres [rest1_lr]
#subjects2.remove('150423'); # not similar timepoints as othres [rest2_lr]

# sub-disk3 outliers
subjects3.remove('179548'); subjects3.remove('197449'); subjects3.remove('203721'); subjects3.remove('207628'); # no resting data
subjects3.remove('190132'); # REST1_LR
subjects3.remove('179952'); subjects3.remove('193441'); subjects3.remove('197651'); subjects3.remove('200210'); subjects3.remove('201717'); subjects3.remove('208428'); # REST2_LR
subjects3.remove('183337'); # not similar timepoints as others [rest2_lr,rl]

#Setting the parcel files to be the Yeo17_200 cortical parcels (or anything else)
parcelCIFTIFile='/Volumes/Elements/Hyperalignment/Parcellations/HCP/fslr32k/cifti/Schaefer2018_200Parcels_17Networks_order.dlabel.nii'

for subj in subjects3:
    #Set input CIFTI file and output FILE NAME
    inputFile = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj)
    parcelTSFilename = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_Yeo17_200.ptseries.nii'.format(subj=subj)
    # Load in dense array time series using nibabel
    dtseries = np.squeeze(nib.load(inputFile).get_data())
    # Parcellate dense time series using wb_command
    os.chdir('/Applications/workbench/bin_macosx64/')
    os.system('./wb_command -cifti-parcellate ' + inputFile + ' ' + parcelCIFTIFile + ' COLUMN ' + parcelTSFilename + ' -method MEAN')
    print('Rest1_LR: ' + subj)
    
for subj in subjects3:
    #Set input CIFTI file and output FILE NAME
    inputFile = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj)
    parcelTSFilename = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_Yeo17_200.ptseries.nii'.format(subj=subj)
    # Load in dense array time series using nibabel
    dtseries = np.squeeze(nib.load(inputFile).get_data())
    # Parcellate dense time series using wb_command
    os.chdir('/Applications/workbench/bin_macosx64/')
    os.system('./wb_command -cifti-parcellate ' + inputFile + ' ' + parcelCIFTIFile + ' COLUMN ' + parcelTSFilename + ' -method MEAN')
    print('Rest1_RL: ' + subj) 
    
for subj in subjects3:
    #Set input CIFTI file and output FILE NAME
    inputFile = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj)
    parcelTSFilename = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_Yeo17_200.ptseries.nii'.format(subj=subj)
    # Load in dense array time series using nibabel
    dtseries = np.squeeze(nib.load(inputFile).get_data())
    # Parcellate dense time series using wb_command
    os.chdir('/Applications/workbench/bin_macosx64/')
    os.system('./wb_command -cifti-parcellate ' + inputFile + ' ' + parcelCIFTIFile + ' COLUMN ' + parcelTSFilename + ' -method MEAN')
    print('Rest2_LR: ' + subj)
    
for subj in subjects3:
    #Set input CIFTI file and output FILE NAME
    inputFile = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj)
    parcelTSFilename = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_Yeo17_200.ptseries.nii'.format(subj=subj)
    # Load in dense array time series using nibabel
    dtseries = np.squeeze(nib.load(inputFile).get_data())
    # Parcellate dense time series using wb_command
    os.chdir('/Applications/workbench/bin_macosx64/')
    os.system('./wb_command -cifti-parcellate ' + inputFile + ' ' + parcelCIFTIFile + ' COLUMN ' + parcelTSFilename + ' -method MEAN')
    print('Rest2_RL: ' + subj) 

"""# Load in parcellated data using nibabel
parcellated = np.squeeze(nib.load(parcelTSFilename).get_data())
normalized = stats.zscore(parcellated, axis=0)

# Loading community ordering files
# netassignments = np.loadtxt('cortex_parcel_network_assignments.txt')
##need to subtract one to make it compatible for python indices
#indsort = np.loadtxt('cortex_community_order.txt',dtype=int) - 1 
#indsort.shape = (len(indsort),1)
##netorder = np.loadtxt('network_labelfile.txt')

# Computing functional connectivity and visualizing the data (assuming preprocessing has already been done)
FCmat = np.corrcoef(normalized.T)
#FCmat_sorted = FCmat[indsort,indsort.T]
plt.figure(figsize=(7,7))
plt.imshow(FCmat, origin='upper', cmap='gray_r')
plt.colorbar(fraction=0.046)
plt.title('FC Matrix',fontsize=24)
plt.show()
"""

# Save parcellated fMRI time series to a numpy array
# -----------------------------------
rest1_lr = []
rest1_rl = []
rest2_lr = []
rest2_rl = []

for subj in subjects3:
    # REST1_LR
    path = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_Yeo17_200.ptseries.nii'.format(subj=subj)
    ts_p = nib.load(path).get_data()
    ts_n = stats.zscore(ts_p, axis=0)
    rest1_lr.append(ts_n) 
    # REST1_RL
    path = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_Yeo17_200.ptseries.nii'.format(subj=subj)
    ts_p = nib.load(path).get_data()
    ts_n = stats.zscore(ts_p, axis=0)
    rest1_rl.append(ts_n) 
    # REST2_LR  
    path = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_Yeo17_200.ptseries.nii'.format(subj=subj)
    ts_p = nib.load(path).get_data()
    ts_n = stats.zscore(ts_p, axis=0)
    rest2_lr.append(ts_n) 
    # REST2_RL
    path = data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_Yeo17_200.ptseries.nii'.format(subj=subj)
    ts_p = nib.load(path).get_data()
    ts_n = stats.zscore(ts_p, axis=0)
    rest2_rl.append(ts_n)
    
    print('Subject ' + subj)
    
np.save('/Volumes/Elements/Chris/yeo17_200/disk3/ts/rest1_lr', rest1_lr)
np.save('/Volumes/Elements/Chris/yeo17_200/disk3/ts/rest1_rl', rest1_rl) 
np.save('/Volumes/Elements/Chris/yeo17_200/disk3/ts/rest2_lr', rest2_lr)
np.save('/Volumes/Elements/Chris/yeo17_200/disk3/ts/rest2_rl', rest2_rl)

np.save('/Volumes/Elements/Chris/yeo17_200/disk3/ID', subjects3)

# ROI-to-ROI correlations of resting-state fMRI data
# -----------------------------------
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional

fc_rest1_lr = correlation_measure.fit_transform(rest1_lr)
fc_rest1_rl = correlation_measure.fit_transform(rest1_rl)
fc_rest2_lr = correlation_measure.fit_transform(rest2_lr)
fc_rest2_rl = correlation_measure.fit_transform(rest2_rl)

np.save('/Volumes/Elements/Chris/yeo17_200/disk3/fc/corr_rest1_lr', fc_rest1_lr)
np.save('/Volumes/Elements/Chris/yeo17_200/disk3/fc/corr_rest1_rl', fc_rest1_rl) 
np.save('/Volumes/Elements/Chris/yeo17_200/disk3/fc/corr_rest2_lr', fc_rest2_lr)
np.save('/Volumes/Elements/Chris/yeo17_200/disk3/fc/corr_rest2_rl', fc_rest2_rl) 

# as well as the average correlation across all fitted subjects.
mean_correlation_matrix = correlation_measure.mean_
print('Mean correlation has shape {0}.'.format(mean_correlation_matrix.shape))  

#%% Connectivity profiles (CPs): use nibabel to load a CIFTI file with the fMRI time series  
# then extract the fMRI time series to a numpy array, then create CPs

# idx = int(os.getenv("SGE_TASK_ID"))-1
# idx = 0

n_sbj = 200
N = 1200

cp_rest1LR = []
cp_rest1RL = []
cp_rest2LR = []
cp_rest2RL = []

for i, subj in enumerate(subjects[:n_sbj]): # subjects[idx:idx+1]:
    
    sub_id = int(subj)
    # there are 3 disks we considered in HCP dataset
    if sub_id >= 100206 and sub_id <= 128026: # disk1
        disk = disks[1-1]
    elif sub_id >= 128127 and sub_id <= 154229: # disk2
        disk = disks[2-1]
    elif sub_id >= 179245 and sub_id <= 209329: # disk3
        disk = disks[3-1]
        
    img = nib.load(data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj))
    data = img.get_fdata()
    A = hcp.normalize(data) # fine-scale data
    B = hcp.parcellate(A, atlas)[:, :360] # targets: coarse-scale data
    # Store columnw-wise sum in A and B, as they would be used at few places
    sA, sB = A.sum(0), B.sum(0)
    # Basically there are four parts in the main formula, compute them one-by-one
    p1 = N*np.einsum('ij,ik->kj',A,B)
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)
    # Finally compute Pearson Correlation Coefficient as 2D array (connectivty profiles)
    cp = (p1 - p2)/np.sqrt(p4*p3[:,None])
    cp_rest1LR.append(cp) 
    print(i, subj)
    
    img = nib.load(data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj))
    data = img.get_fdata()  
    A = hcp.normalize(data) # fine-scale data
    B = hcp.parcellate(A, atlas)[:, :360] # targets: coarse-scale data
    # Store columnw-wise sum in A and B, as they would be used at few places
    sA, sB = A.sum(0), B.sum(0)
    # Basically there are four parts in the main formula, compute them one-by-one
    p1 = N*np.einsum('ij,ik->kj',A,B)
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)
    # Finally compute Pearson Correlation Coefficient as 2D array (connectivty profiles)
    cp = (p1 - p2)/np.sqrt(p4*p3[:,None])
    cp_rest1RL.append(cp) 
    print(i, subj)
    
    img = nib.load(data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj))
    data = img.get_fdata()  
    A = hcp.normalize(data) # fine-scale data
    B = hcp.parcellate(A, atlas)[:, :360] # targets: coarse-scale data
    # Store columnw-wise sum in A and B, as they would be used at few places
    sA, sB = A.sum(0), B.sum(0)
    # Basically there are four parts in the main formula, compute them one-by-one
    p1 = N*np.einsum('ij,ik->kj',A,B)
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)
    # Finally compute Pearson Correlation Coefficient as 2D array (connectivty profiles)
    cp = (p1 - p2)/np.sqrt(p4*p3[:,None])
    cp_rest2LR.append(cp) 
    print(i, subj)
    
    img = nib.load(data_path + disk + subj + '/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj))
    data = img.get_fdata()  
    A = hcp.normalize(data) # fine-scale data
    B = hcp.parcellate(A, atlas)[:, :360] # targets: coarse-scale data
    # Store columnw-wise sum in A and B, as they would be used at few places
    sA, sB = A.sum(0), B.sum(0)
    # Basically there are four parts in the main formula, compute them one-by-one
    p1 = N*np.einsum('ij,ik->kj',A,B)
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)
    # Finally compute Pearson Correlation Coefficient as 2D array (connectivty profiles)
    cp = (p1 - p2)/np.sqrt(p4*p3[:,None])
    cp_rest2RL.append(cp)
    print(i, subj)
    
    # old/slow method for calculating cp
    # cp = np.zeros((target_nmbr, grayordinate))
    # for i in range(target_nmbr):
    #     for j in range(grayordinate): 
    #         cp[i,j] = np.corrcoef(data_p[:,i], data_n[:,j])[0, 1]
    
# saving connectivity profiles
os.chdir('/dcs05/ciprian/smart/farahani/SL-CHA/connectomes/') # '/dcl01/smart/data/fvfarahani/searchlight/connectivity_profiles/'

np.save('cp_rest1LR_200sbj', cp_rest1LR)  
np.save('cp_rest1RL_200sbj', cp_rest1RL) 
np.save('cp_rest2LR_200sbj', cp_rest2LR) 
np.save('cp_rest2RL_200sbj', cp_rest2RL) 

#%% Prepare CP for PyMVPA (Left or Right?): fMRI data points with mapped nodes (indices) to surface file, 
# no auxilliary array, no masking in HA is recommended since we already defined the node indices
# Datasets should have feature attribute `node_indices` containing spatial coordinates of all features

#% loading precalculated connectivity profiles
os.chdir('/dcs05/ciprian/smart/farahani/SL-CHA/connectomes/')
# mapping data points to surface file
hemi = 'left' # 'left' or 'right'
if hemi == 'left':
    roi = 29696 # left
    node_indices = np.zeros((roi,), dtype=int)
    aux = hcp.left_cortex_data(np.arange(roi), fill=-1)
    nan_indices = np.where(aux==-1)[0]
    for i in range(roi):
        node_indices[i] = np.where(aux==i)[0]
    hcp_struct = hcp.struct.cortex_left
elif hemi == 'right':    
    roi = 29716 # left
    node_indices = np.zeros((roi,), dtype=int)
    aux = hcp.right_cortex_data(np.arange(roi), fill=-1)
    nan_indices = np.where(aux==-1)[0]
    for i in range(roi):
        node_indices[i] = np.where(aux==i)[0]    
    hcp_struct = hcp.struct.cortex_right

# REST1LR --> training SL-HA
cp_rest1LR = np.load('cp_rest1LR_200sbj.npy')
dss_rest1LR = []
for k in range(len(cp_rest1LR)):
    ds = Dataset(cp_rest1LR[k][:, hcp_struct])
    ds.fa['node_indices'] = node_indices # set feature attributes
    zscore(ds, chunks_attr=None) # normalize features (vertices) to have unit variance (GLM parameters estimates for each voxel at this point).
    dss_rest1LR.append(ds)

# Each run has 360(718) target regions and 29696 (or 29716) features per subject.
print(dss_rest1LR[0].shape) 
del cp_rest1LR

# REST1RL
cp_rest1RL = np.load('cp_rest1RL_200sbj.npy')
dss_rest1RL = []
for k in range(len(cp_rest1RL)):
    ds = Dataset(cp_rest1RL[k][:, hcp_struct])
    ds.fa['node_indices'] = node_indices
    zscore(ds, chunks_attr=None)
    dss_rest1RL.append(ds)

# Each run has 360(718) target regions and 29696 (or 29716) features per subject.
print(dss_rest1RL[0].shape)
del cp_rest1RL

# REST2LR
cp_rest2LR = np.load('cp_rest2LR_200sbj.npy')
dss_rest2LR = []
for k in range(len(cp_rest2LR)):
    ds = Dataset(cp_rest2LR[k][:, hcp_struct])
    ds.fa['node_indices'] = node_indices
    zscore(ds, chunks_attr=None)
    dss_rest2LR.append(ds)

# Each run has 360(718) target regions and 29696 (or 29716) features per subject.
print(dss_rest2LR[0].shape)
del cp_rest2LR

# REST2RL
cp_rest2RL = np.load('cp_rest2RL_200sbj.npy')
dss_rest2RL = []
for k in range(len(cp_rest2RL)):       
    ds = Dataset(cp_rest2RL[k][:, hcp_struct])
    ds.fa['node_indices'] = node_indices
    zscore(ds, chunks_attr=None)
    dss_rest2RL.append(ds)

# Each run has 360(718) target regions and 29696 (or 29716) features per subject.
print(dss_rest2RL[0].shape)
del cp_rest2RL

#%% Create SearchlightHyperalignment instance

start = timeit.default_timer()
# The QueryEngine is used to find voxel/vertices within a searchlight. 
# This SurfaceQueryEngine use a searchlight radius of 5 mm based on the fsaverage surface.

os.chdir('/dcs05/ciprian/smart/farahani/SL-CHA/surfaces/')
n_sbj = len(dss_rest1LR)

# idx = int(os.getenv("SGE_TASK_ID"))
idx = 1

def select_surface(idx):
    if idx == 1:
        if hemi == 'left': 
            surface, name = 'S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii', 'L.inflated'
        elif hemi == 'right':
            surface, name = 'S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii', 'R.inflated'
    elif idx == 2:
        if hemi == 'left': 
            surface, name = 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii', 'L.midthickness'
        elif hemi == 'right':
            surface, name = 'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii', 'R.midthickness'
    elif idx == 3:
        if hemi == 'left': 
            surface, name = 'S1200.L.pial_MSMAll.32k_fs_LR.surf.gii', 'L.pial'
        elif hemi == 'right':
            surface, name = 'S1200.R.pial_MSMAll.32k_fs_LR.surf.gii', 'R.pial'
    elif idx == 4:
        if hemi == 'left': 
            surface, name = 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii', 'L.very_inflated'
        elif hemi == 'right':
            surface, name = 'S1200.R.very_inflated_MSMAll.32k_fs_LR.surf.gii', 'R.very_inflated'
    return surface, name

surface, name = select_surface(idx)

sl_radius = 10 # r = 10.0 -> 5.3X computation time than r = 5.0
# sl_radius = int(os.getenv("SGE_TASK_ID"))
qe = SurfaceQueryEngine(read_surface(surface), radius=sl_radius) 

hyper = SearchlightHyperalignment(
    queryengine = qe,
    compute_recon = False, # We don't need to project back from common space to subject space
    nproc = 4, # Number of processes to use. Change "Docker - Preferences - Advanced - CPUs" accordingly.
) 

# mask_node_ids = list(node_indices),
# exclude_from_model = list(nan_indices), # almost similar results with "mask_node_ids"
# roi_ids=np.unique(ds3.fa.node_indices),
# "roi_ids" and "mask_node_ids"
# combine_neighbormappers = True, # no differences in my case

# Create common template space with training data
os.chdir('/dcl02/leased/smart/data/SL-CHA/mappers/')
mappers = hyper(dss_rest1LR)

stop = timeit.default_timer()
print('Run time of the SearchlightHyperalignment:', stop - start)

h5save('mappers_'+ str(n_sbj) + 'sbj_' + str(sl_radius) + 'r_' + name + '.hdf5.gz', mappers, compression=9)

#mappers = h5load('mappers.hdf5.gz') # load pre-computed mappers

#%% Create ConnectivityHyperalignment instance

sl_radius = 5.0
qe = SurfaceQueryEngine(read_surface('S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii'), radius=sl_radius)

cha = ConnectivityHyperalignment(
    #seed_indices=[0, 300, 750, 900],  # based on the voxels' indices
    #seed_radius=13,
    seed_queryengines=qe,
    queryengine=qe,
)

"""
seed_queryengines is used to compute connectivity targets 
(sometimes also called connectivity seeds) by averaging responses in each searchlight.  
In the connectivity hyperalignment paper they used connectivity seeds from a
low-resolution surface (AFNI's ico8), and that in combination with a 13 mm
radius can make sure the entire cortical surface is covered by the searchlights.
"""

# Create common template space with training data
cmappers = cha(dss_train)
h5save('cmappers.hdf5.gz', cmappers, compression=9)

#%% Project data (cp or ts) to the common space

# loading (precalculated) mappers
mappers = h5load('/dcs05/ciprian/smart/farahani/SL-CHA/mappers/mappers_200sbj_10r_L.inflated.hdf5.gz') # load pre-computed mappers
# inflated, midthickness, pial, very_inflated

dss_aligned_rest1LR = [mapper.forward(ds) for ds, mapper in zip(dss_rest1LR, mappers)]
_ = [zscore(ds, chunks_attr=None) for ds in dss_aligned_rest1LR]

dss_aligned_rest1RL = [mapper.forward(ds) for ds, mapper in zip(dss_rest1RL, mappers)]
_ = [zscore(ds, chunks_attr=None) for ds in dss_aligned_rest1RL]

dss_aligned_rest2LR = [mapper.forward(ds) for ds, mapper in zip(dss_rest2LR, mappers)]
_ = [zscore(ds, chunks_attr=None) for ds in dss_aligned_rest2LR]

dss_aligned_rest2RL = [mapper.forward(ds) for ds, mapper in zip(dss_rest2RL, mappers)]
_ = [zscore(ds, chunks_attr=None) for ds in dss_aligned_rest2RL]

#%% Calculate inter-subject correlations (ISCs)

# 1) Average ISCs of connectivity profiles in each surface node 
def compute_average_similarity_node(dss, metric='correlation'):
    """
    Returns
    =======
    sim_node : ndarray
        A 1-D array with n_features elements, each element is the average
        pairwise correlation similarity on the corresponding feature.
    """
    n_features = dss[0].shape[1]
    sim_node = np.zeros((n_features, ))
    for i in range(n_features):
        data = np.array([ds.samples[:, i] for ds in dss])
        dist = pdist(data, metric)
        # "Correlation distance" is not the same as the correlation coefficient. 
        # A "distance" between two equal points is supposed to be 0.
        sim_node[i] = 1 - dist.mean()
    return sim_node

sim_rest1LR = compute_average_similarity_node(dss_rest1LR)
sim_rest1RL = compute_average_similarity_node(dss_rest1RL)
sim_rest2LR = compute_average_similarity_node(dss_rest2LR)
sim_rest2RL = compute_average_similarity_node(dss_rest2RL)
sim_aligned_rest1LR = compute_average_similarity_node(dss_aligned_rest1LR)
sim_aligned_rest1RL = compute_average_similarity_node(dss_aligned_rest1RL)
sim_aligned_rest2LR = compute_average_similarity_node(dss_aligned_rest2LR)
sim_aligned_rest2RL = compute_average_similarity_node(dss_aligned_rest2RL)

# 2) Individual whole cortex mean ISCs of connectivity profiles
def compute_average_similarity_subject(dss, k, metric='correlation'):
    """
    Returns
    =======
    sim_subject : ndarray
        A 1-D array with n_features elements, each element is the average
        pairwise correlation similarity on the corresponding feature.
    """
    n_features = dss[0].shape[1]
    sim_subject = np.zeros((n_features, ))
    for i in range(n_features):
        data = np.array([ds.samples[:, i] for ds in dss])
        dist = cdist(data[k:k+1], data, metric)
        sim_subject[i] = 1 - dist.mean()
    return sim_subject.mean()

n_sbj = len(dss_rest1LR)

sim_subject_rest1LR = np.zeros((n_sbj, ))
sim_subject_rest1RL = np.zeros((n_sbj, ))
sim_subject_rest2LR = np.zeros((n_sbj, ))
sim_subject_rest2RL = np.zeros((n_sbj, ))
sim_subject_aligned_rest1LR = np.zeros((n_sbj, ))
sim_subject_aligned_rest1RL = np.zeros((n_sbj, ))
sim_subject_aligned_rest2LR = np.zeros((n_sbj, ))
sim_subject_aligned_rest2RL = np.zeros((n_sbj, ))
for k in range(n_sbj):
    sim_subject_rest1LR[k] = compute_average_similarity_subject(dss_rest1LR, k)
    sim_subject_rest1RL[k] = compute_average_similarity_subject(dss_rest1RL, k)
    sim_subject_rest2LR[k] = compute_average_similarity_subject(dss_rest2LR, k)
    sim_subject_rest2RL[k] = compute_average_similarity_subject(dss_rest2RL, k)
    sim_subject_aligned_rest1LR[k] = compute_average_similarity_subject(dss_aligned_rest1LR, k)
    sim_subject_aligned_rest1RL[k] = compute_average_similarity_subject(dss_aligned_rest1RL, k)
    sim_subject_aligned_rest2LR[k] = compute_average_similarity_subject(dss_aligned_rest2LR, k)
    sim_subject_aligned_rest2RL[k] = compute_average_similarity_subject(dss_aligned_rest2RL, k)
    print(k)

#% Save similarity data (based on node[1] & subject[2]) for selected mappers
import pandas as pd

name = 'R.inflated'
sl_radius = 10

os.chdir('/dcs05/ciprian/smart/farahani/SL-CHA/ISC/')
columns = ["rest1LR", "rest1RL", "rest2LR", "rest2RL", "aligned_rest1LR", "aligned_rest1RL", "aligned_rest2LR", "aligned_rest2RL"]
data_node = np.array([sim_rest1LR, sim_rest1RL, sim_rest2LR, sim_rest2RL, 
                      sim_aligned_rest1LR, sim_aligned_rest1RL, sim_aligned_rest2LR, sim_aligned_rest2RL]).T
data_subject = np.array([sim_subject_rest1LR, sim_subject_rest1RL, sim_subject_rest2LR, sim_subject_rest2RL, 
                         sim_subject_aligned_rest1LR, sim_subject_aligned_rest1RL, sim_subject_aligned_rest2LR, sim_subject_aligned_rest2RL]).T
df_node = pd.DataFrame(data=data_node, columns=columns)
df_subject = pd.DataFrame(data=data_subject, columns=columns)
df_node.to_csv('sim_node_' + str(n_sbj) + 'sbj_' + str(sl_radius) + 'r_' + name + '.csv', sep='\t', encoding='utf-8')
df_subject.to_csv('sim_subject_' + str(n_sbj) + 'sbj_' + str(sl_radius) + 'r_' + name + '.csv', sep='\t', encoding='utf-8')  

#%% Load ISCs as dataframe, then visualize the results

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set working directory
os.chdir('/Volumes/Elements/Hyperalignment/HCP/200sbj/ISC/')

# Set variables
SURF = 'inflated'  # inflated, midthickness, pial, very_inflated
SUBJ = '200sbj_'  # '20sbj', '50sbj'
RADI = '10r_'     # '5r', '10r', '15r', '20r'

# Load data
df_node_l = pd.read_csv(f'sim_node_{SUBJ}{RADI}L.{SURF}.csv', sep='\t', index_col=0)
df_subject_l = pd.read_csv(f'sim_subject_{SUBJ}{RADI}L.{SURF}.csv', sep='\t', index_col=0)
df_node_r = pd.read_csv(f'sim_node_{SUBJ}{RADI}R.{SURF}.csv', sep='\t', index_col=0)
df_subject_r = pd.read_csv(f'sim_subject_{SUBJ}{RADI}R.{SURF}.csv', sep='\t', index_col=0)

# Merge data
df_node = pd.concat([df_node_l, df_node_r], ignore_index=True)
df_subject = ((df_subject_l * 29696) + (df_subject_r * 29716)) / 59412

# Extract variables
var1 = df_node['rest2LR'].values # Train, Test1, 2, 3
var2 = df_node['aligned_rest2LR'].values # Aligned0, 1, 2, 3
var3 = df_subject['rest2LR'].values # Test1, 2, 3
var4 = df_subject['aligned_rest2LR'].values # Aligned0, 1, 2, 3

# 1.1) Average ISCs in each surface node 
plt.figure(figsize=(8, 8))
plt.scatter(var1, var2, s=20, alpha=0.4, edgecolors='none')
plt.scatter(var1.mean(), var2.mean(), s=200, marker='o', color='r', edgecolors='k')
plt.xlim([-0.05, 0.8]) 
plt.ylim([-0.05, 0.8]) 
plt.xlabel('MSM-All', size=26)
plt.ylabel('Searchlight CHA', size=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot([-1, 1], [-1, 1], 'k--')
plt.text(var1.mean()-0.08, var2.mean()+0.03, '({:.2f}, {:.2f})'.format(var1.mean(), var2.mean()), fontsize=20, color='white', fontweight='bold')
#plt.title('Average pairwise correlation', size=18)
plt.tight_layout()
plt.savefig('ISC-Vertex.png', dpi=300, bbox_inches='tight')
plt.show()
# var2.mean()/var1.mean()

# 1.2) Distribution of ISCs across all voxels
import seaborn as sns
sns.set_style("white")
kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
plt.figure(figsize=(15,7), dpi= 300)
plt.ylabel('Frequency', fontsize=26)
plt.xlabel('Vertex ISC Values', fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
sns.distplot(var1, color="#FFD700", label="MSM-All", **kwargs)
sns.distplot(var2, color="#7F00FF", label="CHA", **kwargs)
plt.axvline(x=np.mean(list(var1)), linestyle='--', color='k', linewidth=2)
plt.axvline(x=np.mean(list(var2)), linestyle='--', color='k', linewidth=2)
plt.text(np.mean(list(var1)), plt.ylim()[1]*0.8, f"Mean: {np.mean(list(var1)):.2f}", va='top', ha='center', color='k', fontsize=18)
plt.text(np.mean(list(var2)), plt.ylim()[1]*0.7, f"Mean: {np.mean(list(var2)):.2f}", va='top', ha='center', color='k', fontsize=18)
plt.xlim(-0.05, 0.9)
plt.legend(prop={'size':20})
plt.savefig('distribution_ISC.png', dpi=300, bbox_inches='tight')


# 1.2) Scatter plot of individual ISCs before and after CHA with linear fit
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
sns.set_style('white')
# Calculate Pearson correlation coefficient
corr_coef, _ = pearsonr(var3, var4)
# Create scatter plot with linear regression line
ax = sns.regplot(x=var3, y=var4, scatter_kws={"s": 20})
# Add text with correlation coefficient to plot
ax.text(0.05, 0.95, f'r = {corr_coef:.2f}', transform=ax.transAxes, fontsize=16, verticalalignment='top')
# Set axis labels and tick sizes
ax.set_xlabel('MSM-All', fontsize='x-large')
ax.set_ylabel('Searchlight CHA', fontsize='x-large')
ax.tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
# Set square aspect ratio
plt.axis('square')
# Save plot
plt.savefig('scatter_ISC.png', dpi=300, bbox_inches='tight')

# 1.3) Visualize results on cortical brain map (ONLY LEF OR RIGHT)
view = 'medial' # {‘lateral’, ‘medial’, ‘dorsal’, ‘ventral’, ‘anterior’, ‘posterior’},
h = 'right' # which hemisphere to train HA? 'left' or 'right'
if h == 'left':
    hcp_mesh = hcp.mesh.inflated_left
    hcp_data = hcp.left_cortex_data
    hcp_mesh_sulc = hcp.mesh.sulc_left
    var5 = np.array(df_node_l.loc[:,'rest2LR']) # Train, Test1, 2, 3
    var6 = np.array(df_node_l.loc[:,'aligned_rest2LR']) # Aligned0, 1, 2, 3
elif h == 'right':    
    hcp_mesh = hcp.mesh.inflated_right
    hcp_data = hcp.right_cortex_data
    hcp_mesh_sulc = hcp.mesh.sulc_right
    var5 = np.array(df_node_r.loc[:,'rest2LR']) # Train, Test1, 2, 3
    var6 = np.array(df_node_r.loc[:,'aligned_rest2LR']) # Aligned0, 1, 2, 3
# hcp.mesh.keys() # white, pial, inflated, flat, ...
# inflated, midthickness, pial, very_inflated
# initial data
plotting.plot_surf_stat_map(hcp_mesh, hcp_data(var5),
    hemi=h, view=view, cmap='cold_hot', colorbar=False, vmax=0.9,
    threshold=0.40000005, bg_map=hcp_mesh_sulc) # bg_map: a sulcal depth map for realistic shading
# title='Surface left hemisphere (Test)'
plt.savefig('ISC200sbj360mmp_r10_' + view + '_test2.pdf', bbox_inches='tight')

# =============================================================================
# # interactive 3D visualization in a web browser
# view = plotting.view_surf(hcp.mesh.inflated_left, hcp.left_cortex_data(var5), vmax=0.9,
#     threshold=0.40000005, bg_map=hcp.mesh.sulc_left) 
# view.open_in_browser()
# =============================================================================

# aligned data
plotting.plot_surf_stat_map(hcp_mesh, hcp_data(var6), 
    hemi=h, view=view, cmap='cold_hot', colorbar=False, vmax=0.9,
    threshold=0.40000005, bg_map=hcp_mesh_sulc) # bg_map: a sulcal depth map for realistic shading
plt.savefig('ISC200sbj360mmp_r10_' + view + '_aligned2.pdf', bbox_inches='tight')

# =============================================================================
# # interactive 3D visualization in a web browser
# view = plotting.view_surf(hcp.mesh.inflated_left, hcp.left_cortex_data(var6), vmax=0.9,
#     threshold=0.40000005, bg_map=hcp.mesh.sulc_left) 
# view.open_in_browser()
# =============================================================================

"""view = plotting.view_surf(hcp.mesh.inflated, 
    hcp.cortex_data(hcp.mask(np.ones((91282,)), hcp.ca_parcels.map_all==22)), 
    threshold=0.1, bg_map=hcp.mesh.sulc)
view.open_in_browser()

#view = hcp.view_parcellation(hcp.mesh.inflated, hcp.mmp)
#view.open_in_browser()"""

#%% Fingerprinting (only unaligned data)
ts_train = []
ts_test1 = []
ts_test2 = []
ts_test3 = []

n_sbj = 30

atlas = hcp.ca_parcels

# load timeseries
for subj in subjects1[0:n_sbj]:           
    img = nib.load(data_path + disk[0] + subj + '/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj))
    X = img.get_fdata()
    Xn = hcp.normalize(X)
    # Xx = Xn[:, hcp.struct.thalamus_right]
    Xp = hcp.parcellate(Xn, atlas)
    ts_train.append(Xp)  

for subj in subjects1[0:n_sbj]:
    img = nib.load(data_path + disk[0] + subj + '/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj))
    X = img.get_fdata()  
    Xn = hcp.normalize(X)
    Xp = hcp.parcellate(Xn, atlas)
    ts_test1.append(Xp)

for subj in subjects1[0:n_sbj]: 
    img = nib.load(data_path + disk[0] + subj + '/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj))
    X = img.get_fdata()  
    Xn = hcp.normalize(X)
    Xp = hcp.parcellate(Xn, atlas)
    ts_test2.append(Xp)

for subj in subjects1[0:n_sbj]:    
    img = nib.load(data_path + disk[0] + subj + '/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj))
    X = img.get_fdata()  
    Xn = hcp.normalize(X)
    Xp = hcp.parcellate(Xn, atlas) 
    ts_test3.append(Xp)  

np.save('/Volumes/Elements/PCR/ts_train', ts_train)
np.save('/Volumes/Elements/PCR/ts_test1', ts_test1)   
np.save('/Volumes/Elements/PCR/ts_test2', ts_test2)
np.save('/Volumes/Elements/PCR/ts_test3', ts_test3)

from nilearn.connectome import ConnectivityMeasure
from scipy import stats
correlation_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional

os.chdir('/Volumes/Elements/PCR/') 

ts_train = np.load('ts_train.npy') 
ts_test1 = np.load('ts_test1.npy') 
ts_test2 = np.load('ts_test2.npy') 
ts_test3 = np.load('ts_test3.npy')
ts_REST1 = np.hstack((ts_train, ts_test1))
ts_REST2 = np.hstack((ts_test2, ts_test3))

corr_train = correlation_measure.fit_transform(ts_train)
corr_test1 = correlation_measure.fit_transform(ts_test1)
corr_test2 = correlation_measure.fit_transform(ts_test2)
corr_test3 = correlation_measure.fit_transform(ts_test3)
corr_REST1 = correlation_measure.fit_transform(ts_REST1)
corr_REST2 = correlation_measure.fit_transform(ts_REST2)
   
tril_train = []; tril_test1 = []; tril_test2 = []; tril_test3 = []
tril_REST1 = []; tril_REST2 = []

for k in range(corr_train.shape[0]):           
    
    tril0 = corr_train[k][np.tril_indices(corr_train.shape[2], k = -1)] # K = -1 [without diagonal] or 0 [with]
    tril1 = corr_test1[k][np.tril_indices(corr_test1.shape[2], k = -1)] # K = -1 [without diagonal] or 0 [with]
    tril2 = corr_test2[k][np.tril_indices(corr_test2.shape[2], k = -1)] # K = -1 [without diagonal] or 0 [with]
    tril3 = corr_test3[k][np.tril_indices(corr_test3.shape[2], k = -1)] # K = -1 [without diagonal] or 0 [with]
    tril_R1 = corr_REST1[k][np.tril_indices(corr_REST1.shape[2], k = -1)] # K = -1 [without diagonal] or 0 [with]
    tril_R2 = corr_REST2[k][np.tril_indices(corr_REST2.shape[2], k = -1)] # K = -1 [without diagonal] or 0 [with]

    tril_train.append(tril0)
    tril_test1.append(tril1)
    tril_test2.append(tril2)
    tril_test3.append(tril3)
    tril_REST1.append(tril_R1)
    tril_REST2.append(tril_R2)

# pairwise correlation accross subjects in different test sets
test = np.zeros((n_sbj,n_sbj))
for k in range(n_sbj):
    for l in range(n_sbj):
        test[k,l] = np.corrcoef(tril_REST1[k], tril_REST2[l])[0, 1]

# Get index for the highest value
index = test.argsort()[:,-1]

for k in range(n_sbj):
    test[k,:] = [0 if i < test[k,index[k]] else 1 for i in test[k,:]]
        
plt.imshow(test)
plt.colorbar()

sum_diagonal = sum(test[i][i] for i in range(n_sbj))   
sum_column = np.sum(test, axis=0) 
sum_row = np.sum(test, axis=1)   

# Plotting the lower triangular matrix for one subject
import pandas as pd 
import seaborn as sns

# E.g., first subject [0] in each set
#df_train = pd.DataFrame(corr_train[0]) 
df_REST1 = pd.DataFrame(corr_REST1[49]) 
df_REST2 = pd.DataFrame(corr_REST2[49])

def get_lower_tri_heatmap(df, output="triangular_matrix.png"):
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = False

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(66, 54))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(df, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=0, cbar_kws={"shrink": .5})
    # save to file
    fig = sns_plot.get_figure()
    fig.savefig(output)
    
#get_lower_tri_heatmap(df_train, output="triangular_train.png")  
get_lower_tri_heatmap(df_REST2, output="triangular_REST1.png")   

#%% #######################################################################
# **** Projecting Initial Timeseries into the CHA-derived Common Space ****
# #########################################################################

# Load pre-computed mappers  
mappers_l = h5load('/dcs05/ciprian/smart/farahani/SL-CHA/mappers/mappers_200sbj_10r_L.inflated.hdf5.gz')#[:n_sbj]
mappers_r = h5load('/dcs05/ciprian/smart/farahani/SL-CHA/mappers/mappers_200sbj_10r_R.inflated.hdf5.gz')#[:n_sbj]
n_sbj = len(mappers_l)

# Define the subject disk mapping
subject_disk_mapping = {
    range(100206, 128027): 1,
    range(128127, 154230): 2,
    range(179245, 209330): 3
}

output_dir = '/dcs05/ciprian/smart/farahani/SL-CHA/ts'

def process_data(data_path, output_dir, subjects, n_sbj, subject_disk_mapping, disks, mappers_l, mappers_r, prefix):
    # Determine the disk number based on the subject ID
    for i, subj in enumerate(subjects[:n_sbj]):
        sbj_id = int(subj)
        for subjects_range, disk_num in subject_disk_mapping.items():
            if sbj_id in subjects_range:
                disk = disks[disk_num - 1]
                break
        # Load original timeseries 
        img = nib.load(data_path + disk + subj + f'/MNINonLinear/Results/rfMRI_{prefix}/rfMRI_{prefix}_Atlas_MSMAll_hp2000_clean.dtseries.nii')
        data = img.get_fdata()
        data_n = hcp.normalize(data)
        ds_lh = Dataset(data_n[:, hcp.struct.cortex_left]) # zscore(ds, chunks_attr=None) # normalize features (vertices) to have unit variance
        ds_rh = Dataset(data_n[:, hcp.struct.cortex_right])
        ds_sc = Dataset(data_n[:, hcp.struct.subcortical])
        # Project timeseries to the common space for the current subject
        mapper_l, mapper_r = mappers_l[i], mappers_r[i]
        ds_aligned_lh, ds_aligned_rh = mapper_l.forward(ds_lh), mapper_r.forward(ds_rh)
        zscore(ds_aligned_lh, chunks_attr=None)
        zscore(ds_aligned_rh, chunks_attr=None)
        ds, ds_aligned = hstack((ds_lh, ds_rh, ds_sc)), hstack((ds_aligned_lh, ds_aligned_rh, ds_sc))
        # Save the numpy arrays for the current subject with a unique name
        np.save(f'{output_dir}/{prefix}_MSM/{prefix}_MSM_{subj}.npy', ds.samples)
        np.save(f'{output_dir}/{prefix}_CHA/{prefix}_CHA_{subj}.npy', ds_aligned.samples)
        print(i)

process_data(data_path, output_dir, subjects, n_sbj, subject_disk_mapping, disks, mappers_l, mappers_r, 'REST1_LR')
process_data(data_path, output_dir, subjects, n_sbj, subject_disk_mapping, disks, mappers_l, mappers_r, 'REST1_RL')
process_data(data_path, output_dir, subjects, n_sbj, subject_disk_mapping, disks, mappers_l, mappers_r, 'REST2_LR')
process_data(data_path, output_dir, subjects, n_sbj, subject_disk_mapping, disks, mappers_l, mappers_r, 'REST2_RL')

#%% #######################################################################
# *****                  Identification Accuracies                    *****
# #########################################################################

# qsub -cwd -t 1:360 id_acc.sh
# qsub -cwd -t 1,3,5,7 id_acc.sh

import os
roi = int(os.getenv("SGE_TASK_ID"))
# range(1, len(atlas.nontrivial_ids)+1 - extra) --> # region/network of interet
# extra = 19 # set to 19 for mmp, 358 for ca_parcels, and 0 otherwise

import itertools
from multiprocessing import Pool, cpu_count
num_cpus = 12 # num_cpus = multiprocessing.cpu_count()

# define sessions, number of subjects, corresponding roi
sessions1 = ['REST1_LR_MSM', 'REST1_RL_MSM', 'REST2_LR_MSM', 'REST2_RL_MSM']
sessions2 = ['REST1_LR_CHA', 'REST1_RL_CHA', 'REST2_LR_CHA', 'REST2_RL_CHA']
sessions = sessions1 + sessions2
n_sbj = 200

path = '/dcs05/ciprian/smart/farahani/SL-CHA/ts'

def col_corr(A, B):
    """
    Calculates the column-wise correlation between two matrices
    
    Args:
    A, B: input matrices of shape (n_samples, n_features)
    
    Returns:
    corr: the Pearson correlation coefficient between each pair of columns in 
    A and B as a 2D array of shape (n_features_A, n_features_B)
    
    This function calculates the column-wise correlation between two matrices using a 
    vectorized approach. It takes advantage of NumPy's optimized functions, such as einsum(), 
    to speed up the computation. This approach has the same results as the nested loop using 
    np.corrcoef() but is much faster.
    """
    # Check if input matrices have the same number of samples
    # assert A.shape[0] == B.shape[0], "Input matrices must have the same number of samples."
    # Get the column size from the input matrices
    N = A.shape[0] # It must be same for both matrices
    # Store column-wise sum in A and B, as they would be used at few places
    sA = A.sum(axis=0)
    sB = B.sum(axis=0)
    # Basically there are four parts in the main formula, compute them one-by-one
    p1 = N*np.einsum('ij,ik->kj', A, B)
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(axis=0)) - (sB**2)
    p4 = N*((A**2).sum(axis=0)) - (sA**2)
    # Finally compute Pearson Correlation Coefficient as 2D array 
    corr = ((p1 - p2)/np.sqrt(p4*p3[:,None])).T
    return corr


def extract_corr_flat(session_name, roi, n_sbj):
    roi_idx = np.where(hcp.mmp.map_all[:59412] == roi)[0] # only cortex
    corr_flat_all = []
    for subj in subjects[:n_sbj]:
        ts = np.load(f'{path}/{session_name}/{session_name}_{subj}.npy')
        # Bipartite Correlation (A: mean(ts) of ROIs - B: vertices of a region)
        A = hcp.parcellate(ts, hcp.mmp)[:,:360] # coarse matrix
        B = ts[:, roi_idx] # fine matrix
        corr = col_corr(A, B)
        # Perform Fisher r-to-z transform: make matrix to be of normal distribution
        zcorr = np.arctanh(corr)
        # Flatten the corr array
        corr_flat = zcorr.flatten()
        corr_flat_all.append(corr_flat)
    return np.array(corr_flat_all)


def process_session_pair(pair):
    # extract correlation matrices for each session
    flat1 = extract_corr_flat(pair[0], roi=roi, n_sbj=n_sbj)
    flat2 = extract_corr_flat(pair[1], roi=roi, n_sbj=n_sbj)
    # compute similarity matrices for each pair
    sim1 = col_corr(flat1.T, flat2.T)
    sim2 = sim1.T #sim2 = col_corr(flat2.T, flat1.T)
    # Normalizing similarity matrix; VERY IMPORTANT1
    sim1 = hcp.normalize(sim1)
    sim2 = hcp.normalize(sim2) 
    sim = hcp.normalize((sim1+sim2)/2)
    sim_matrix = sim.copy()  # create a copy of sim
    # Calculate identification accuracy
    m = int(np.where(np.array(sessions) == pair[0])[0])
    n = int(np.where(np.array(sessions) == pair[1])[0])
    # Get index for the highest value
    index = sim.argsort()[:,-1]
    # binarize
    for k in range(n_sbj):
        sim[k,:] = [0 if i < sim[k,index[k]] else 1 for i in sim[k,:]]
    trace = np.trace(sim)
    acc = trace/n_sbj
    print(pair, acc)
    return (m, n, acc, sim_matrix)

# create pairs of sessions
session_pairs1 = list(itertools.combinations(sessions1, 2))
session_pairs2 = list(itertools.combinations(sessions2, 2))
session_pairs = session_pairs1 + session_pairs2

# initialize empty similarity and accuracy matrices
sim_matrices = {pair: np.zeros((n_sbj, n_sbj)) for pair in session_pairs}
acc_matrix = np.zeros((len(sessions1),len(sessions1)))

# run the loop over pairs of sessions in parallel
with Pool(num_cpus) as p:
    results = p.map(process_session_pair, session_pairs)

# populate the accuracy matrix and sim_matrices dictionary
for m, n, acc, sim_matrix in results:
    if n < 4 and m < 4: # MSM: upper triangular
        acc_matrix[m,n] = acc
    else: # CHA: lower triangular
        acc_matrix[n-4,m-4] = acc # pay attention: (p,q) not (q,p)
    sim_matrices[(sessions[m], sessions[n])] = sim_matrix

# save the acc_matrix for the given roi
output_dir = '/dcs05/ciprian/smart/farahani/SL-CHA/id_acc'
np.save(f'{output_dir}/acc_roi_{roi}.npy', acc_matrix)

#%% Plot identification accuracy results (workbench)

import nibabel as nib
import numpy as np
import hcp_utils as hcp

acc_path = '/Volumes/Elements/Hyperalignment/HCP/200sbj/id_acc'
mmp_path = '/Volumes/Elements/Hyperalignment/HCP/workbench/MMP/'

# Load the CIFTI2 file
nifti_file = mmp_path + '/S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
img = nib.load(nifti_file)

# Get the data array
data = img.get_fdata()

# Create a new random vector with the same shape as the original data array
new_data = np.random.rand(*data.shape) # (1, 59412)

for roi in range(1, 361):
    roi_idx = np.where(hcp.mmp.map_all[:59412] == roi)[0]
    try:
        acc = np.load(f'{acc_path}/acc_roi_{roi}.npy')[0,1] # REST1_LR & REST1_RL
        new_data[0, roi_idx] = acc
    except FileNotFoundError:
        pass # do nothing if the file doesn't exist

# Create a copy of the data array and replace it with the new random vector
new_img = nib.Cifti2Image(new_data, img.header)

# Save the modified CIFTI2 file
output_file = mmp_path + '/output/MSM_R1LR_R1RL.dscalar.nii'
nib.save(new_img, output_file)

#%% COARSE-SCALE TIMESERIES (WHOLE BRAIN PARCELLATION) + graph analysis
import numpy as np
import brainconn
from brainconn import degree, centrality, clustering, core, distance, modularity, utils, similarity
import networkx as nx

n_sbj = 30
n_roi = 360

ts_rest1LR = []
ts_rest1RL = []
ts_rest2LR = []
ts_rest2RL = []
ts_aligned_rest1LR = []
ts_aligned_rest1RL = []
ts_aligned_rest2LR = []
ts_aligned_rest2RL = []

# =============================================================================
# aux = np.zeros((1200,31870)) # add zeros columns (instead of subcorticals) to created numpy arrays for parcelation purpose
# 
# # --------- unaligned ---------
# for k in range(len(subjects1[0:n_sbj])):
#     ts_rest1LR.append(hcp.parcellate(np.append(dss_rest1LR[k].samples, aux, 1), hcp.mmp)[:,:360])
#     ts_rest1RL.append(hcp.parcellate(np.append(dss_rest1RL[k].samples, aux, 1), hcp.mmp)[:,:360])
#     ts_rest2LR.append(hcp.parcellate(np.append(dss_rest2LR[k].samples, aux, 1), hcp.mmp)[:,:360])
#     ts_rest2RL.append(hcp.parcellate(np.append(dss_rest2RL[k].samples, aux, 1), hcp.mmp)[:,:360])
#     print(k) 
# 
# # --------- aligned ---------
# for k in range(len(subjects1[0:n_sbj])):
#     ts_aligned_rest1LR.append(hcp.parcellate(np.append(dss_aligned_rest1LR[k].samples, aux, 1), hcp.mmp)[:,:360])
#     ts_aligned_rest1RL.append(hcp.parcellate(np.append(dss_aligned_rest1RL[k].samples, aux, 1), hcp.mmp)[:,:360])
#     ts_aligned_rest2LR.append(hcp.parcellate(np.append(dss_aligned_rest2LR[k].samples, aux, 1), hcp.mmp)[:,:360])
#     ts_aligned_rest2RL.append(hcp.parcellate(np.append(dss_aligned_rest2RL[k].samples, aux, 1), hcp.mmp)[:,:360])
#     print(k)    
# 
# np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp/ts_rest1LR_30sbj', ts_rest1LR)
# np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp/ts_rest1RL_30sbj', ts_rest1RL)    
# np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp/ts_rest2LR_30sbj', ts_rest2LR)
# np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp/ts_rest2RL_30sbj', ts_rest2RL)
# np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp/ts_aligned_rest1LR_30sbj', ts_aligned_rest1LR)
# np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp/ts_aligned_rest1RL_30sbj', ts_aligned_rest1RL)    
# np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp/ts_aligned_rest2LR_30sbj', ts_aligned_rest2LR)
# np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp/ts_aligned_rest2RL_30sbj', ts_aligned_rest2RL)
# 
# =============================================================================
# load timeseries
ts_rest1LR = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/ts_mmp/ts_rest1LR_30sbj.npy') 
ts_rest1RL = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/ts_mmp/ts_rest1RL_30sbj.npy')
ts_rest2LR = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/ts_mmp/ts_rest2LR_30sbj.npy')
ts_rest2RL = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/ts_mmp/ts_rest2RL_30sbj.npy')
ts_aligned_rest1LR = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/ts_mmp/ts_aligned_rest1LR_30sbj.npy')
ts_aligned_rest1RL = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/ts_mmp/ts_aligned_rest1RL_30sbj.npy')
ts_aligned_rest2LR = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/ts_mmp/ts_aligned_rest2LR_30sbj.npy')
ts_aligned_rest2RL = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/ts_mmp/ts_aligned_rest2RL_30sbj.npy')

#% CALCULATE CORRELATION (whole brain)
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional

corr_rest1LR = abs(correlation_measure.fit_transform(ts_rest1LR))
corr_rest1RL = abs(correlation_measure.fit_transform(ts_rest1RL))
corr_rest2LR = abs(correlation_measure.fit_transform(ts_rest2LR))
corr_rest2RL = abs(correlation_measure.fit_transform(ts_rest2RL))
corr_aligned_rest1LR = abs(correlation_measure.fit_transform(ts_aligned_rest1LR))
corr_aligned_rest1RL = abs(correlation_measure.fit_transform(ts_aligned_rest1RL))
corr_aligned_rest2LR = abs(correlation_measure.fit_transform(ts_aligned_rest2LR))
corr_aligned_rest2RL = abs(correlation_measure.fit_transform(ts_aligned_rest2RL))

# if concatenation is neeeded
"""ts_REST1 = np.hstack((ts_rest1LR, ts_rest1RL))
ts_REST2 = np.hstack((ts_rest2LR, ts_rest2RL))
ts_aligned_REST1 = np.hstack((ts_aligned_rest1LR, ts_aligned_rest1RL))
ts_aligned_REST2 = np.hstack((ts_aligned_rest2LR, ts_aligned_rest2RL))

corr_REST1 = correlation_measure.fit_transform(ts_REST1)
corr_REST2 = correlation_measure.fit_transform(ts_REST2)
corr_aligned_REST1 = correlation_measure.fit_transform(ts_aligned_REST1)
corr_aligned_REST2 = correlation_measure.fit_transform(ts_aligned_REST2)"""

# remove the self-connections (zero diagonal) and create weighted graphs
adj_wei = [[] for i in range(8)] # 8 sets (list of lists; wrong way -> adj_wei = [[]] * 8)
adj_bin = [[] for i in range(8)]
con_len = [[] for i in range(8)] # weighted connection-length matrix for 8 sets
thld = 0.3 # threshold -> for binarization
for k in range(n_sbj): 
    np.fill_diagonal(corr_rest1LR[k], 0)
    np.fill_diagonal(corr_rest1RL[k], 0)
    np.fill_diagonal(corr_rest2LR[k], 0)
    np.fill_diagonal(corr_rest2RL[k], 0)
    np.fill_diagonal(corr_aligned_rest1LR[k], 0)
    np.fill_diagonal(corr_aligned_rest1RL[k], 0)
    np.fill_diagonal(corr_aligned_rest2LR[k], 0)
    np.fill_diagonal(corr_aligned_rest2RL[k], 0)
    # weighted
    adj_wei[0].append(corr_rest1LR[k])
    adj_wei[1].append(corr_rest1RL[k])
    adj_wei[2].append(corr_rest2LR[k])
    adj_wei[3].append(corr_rest2RL[k])
    adj_wei[4].append(corr_aligned_rest1LR[k])
    adj_wei[5].append(corr_aligned_rest1RL[k])
    adj_wei[6].append(corr_aligned_rest2LR[k])
    adj_wei[7].append(corr_aligned_rest2RL[k])
    # weighted connection-length matrix (connection lengths is needed prior to computation of weighted distance-based measures, such as distance and betweenness centrality)
    # L_ij = 1/W_ij for all nonzero L_ij; higher connection weights intuitively correspond to shorter lengths
    con_len[0].append(utils.weight_conversion(adj_wei[0][k], 'lengths', copy=True))
    con_len[1].append(utils.weight_conversion(adj_wei[1][k], 'lengths', copy=True))
    con_len[2].append(utils.weight_conversion(adj_wei[2][k], 'lengths', copy=True))
    con_len[3].append(utils.weight_conversion(adj_wei[3][k], 'lengths', copy=True))
    con_len[4].append(utils.weight_conversion(adj_wei[4][k], 'lengths', copy=True))
    con_len[5].append(utils.weight_conversion(adj_wei[5][k], 'lengths', copy=True))
    con_len[6].append(utils.weight_conversion(adj_wei[6][k], 'lengths', copy=True))
    con_len[7].append(utils.weight_conversion(adj_wei[7][k], 'lengths', copy=True))
    # binary
    adj_bin[0].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[0][k], thld, copy=True)))
    adj_bin[1].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[1][k], thld, copy=True)))
    adj_bin[2].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[2][k], thld, copy=True)))
    adj_bin[3].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[3][k], thld, copy=True)))
    adj_bin[4].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[4][k], thld, copy=True)))
    adj_bin[5].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[5][k], thld, copy=True)))
    adj_bin[6].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[6][k], thld, copy=True)))
    adj_bin[7].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[7][k], thld, copy=True)))
   
# define global measures
lam = np.zeros((8,n_sbj)) # lambda (characteristic path length)
glb = np.zeros((8,n_sbj)) # global efficieny
clc = np.zeros((8,n_sbj)) # global clustering coefficients
tra = np.zeros((8,n_sbj)) # Transitivity
ass = np.zeros((8,n_sbj)) # assortativity
mod = np.zeros((8,n_sbj)) # modularity

# compute global measures
for i in range(8):
    for k in range(n_sbj):                
        dis = distance.distance_wei(con_len[i][k])[0] # TIME CONSUMING  
        lam[i,k] = distance.charpath(dis, include_diagonal=False, include_infinite=False)[0]
        glb[i,k] = distance.charpath(dis, include_diagonal=False, include_infinite=False)[1]
        #glb[i,k] = distance.efficiency_wei(adj_wei[i][k], local=False) # time consuming -> for binary matrices is much faster
        #glb[i,k] = distance.efficiency_bin(adj_bin[i][k], local=False)
        clc[i,k] = np.mean(clustering.clustering_coef_bu(adj_bin[i][k]))
        tra[i,k] = np.mean(clustering.transitivity_bu(adj_bin[i][k]))
        ass[i,k] = core.assortativity_bin(adj_bin[i][k], flag=0) # 0: undirected graph
        mod[i,k] = modularity.modularity_louvain_und(adj_bin[i][k], gamma=1, hierarchy=False, seed=None)[1] 
    print(i)      
      
# define/compute local measures  
deg_l = np.zeros((8,30,360))
stg_l = np.zeros((8,30,360))
eig_l = np.zeros((8,30,360))
clc_l = np.zeros((8,30,360))
eff_l = np.zeros((8,30,360))
#par_l = np.zeros((8,30,360))
#zsc_l = np.zeros((8,30,360))
#rch_l = np.zeros((8,30,360))
kco_l = np.zeros((8,30,360))        
for i in range(8):
    for k in range(n_sbj): 
        deg_l[i,k,:] = degree.degrees_und(adj_bin[i][k])
        stg_l[i,k,:] = degree.strengths_und(adj_wei[i][k])
        eig_l[i,k,:] = centrality.eigenvector_centrality_und(adj_bin[i][k])
        clc_l[i,k,:] = clustering.clustering_coef_bu(adj_bin[i][k])
        #eff_l[i,k,:] = distance.efficiency_wei(adj_wei[i][k], local=True) # [Time consuming]^n: 
        eff_l[i,k,:] = distance.efficiency_bin(adj_bin[i][k], local=True) # [Time consuming]^n: 
        #par_l[i,k,:] = centrality.participation_coef(adj_bin[i][k], degree='undirected')
        #zsc_l[i,k,:] = centrality.module_degree_zscore(adj_bin[i][k], flag=0) # 0: undirected graph
        #rch_l[i,k,:] = core.rich_club_bu(adj_bin[i][k])[0]
        kco_l[i,k,:] = centrality.kcoreness_centrality_bu(adj_bin[i][k])[0]
    print(i)     

# find the indices of 360 regions based on 12 networks of cole/anticevic
import hcp_utils as hcp
index = np.zeros((360,))
for roi in range(1,361):
    r = roi-1
    index_parcel = np.where(hcp.ca_parcels.map_all==roi)[0][0]
    index[r] = hcp.ca_network.map_all[index_parcel]
# create sorted index    
index_sorted = index.argsort(kind='stable')

# sort local measures based on the sorted index
deg_l = deg_l[:,:,index_sorted]
stg_l = stg_l[:,:,index_sorted]
eig_l = eig_l[:,:,index_sorted]
clc_l = clc_l[:,:,index_sorted]
eff_l = eff_l[:,:,index_sorted]
#par_l = par_l[:,:,index_sorted]
#zsc_l = zsc_l[:,:,index_sorted]
#rch_l = rch_l[:,:,index_sorted]
kco_l = kco_l[:,:,index_sorted]

# regression analysis based on local patterns
# https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html#sphx-glr-auto-examples-cross-decomposition-plot-pcr-vs-pls-py
# Loading questionnaires's indices
IQ = []
file_in = open('/Volumes/Elements/Hyperalignment/HCP/IQ.txt', 'r')
#file_in = open('/users/ffarahan/IQ.txt', 'r')
for z in file_in.read().split('\n'):
    IQ.append(float(z))

questionnaire = {'IQ': np.array(IQ)}
questionnaire_list = ['IQ']
# Set up a correlation vector (measure * Q_index) corresponding to a value for each region in the parcellation.
index = 'IQ' # Questionnaire index {AM, ESS, PW}

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

local_measure = [deg_l, stg_l, eig_l, clc_l, eff_l, kco_l] ; num = 6
mse_reg, r2_reg = [[] for i in range(len(local_measure))], [[] for i in range(len(local_measure))] # number of measures
pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
for i in range(len(local_measure)):
    for s in range(np.shape(pred_set)[0]):
        # Training/testing sets and target variable
        X_train, y_train = local_measure[i][pred_set[s][0],:,:], questionnaire[index]
        X_test, y_test = local_measure[i][pred_set[s][1],:,:], questionnaire[index]        
        # Create linear regression object
        reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization        
        # Train the model using the training sets
        reg.fit(X_train, y_train)        
        # Make predictions using the testing set
        y_pred_reg = reg.predict(X_test)       
        # The mean squared error
        mse_reg[i].append(mean_squared_error(y_test, y_pred_reg))        
        # The coefficient of determination: 1 is perfect prediction
        r2_reg[i].append(r2_score(y_test, y_pred_reg))
        
#%%
# catplot (multiple barplot)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
num = 6 # 6 measures
data = np.reshape(mse_reg, (num*6,))
df = pd.DataFrame(data=data, columns=["MSE"]) # index=rows
metric = np.repeat(['Degree', 'Strength', 'Eigenvector Centrality', 'Clustering Coefficient', 'Local Efficiency', 'K-coreness Centrality'], 6, axis=0) # 5 measures
df['Measure'] = metric
group = np.tile(['Test 1', 'Test 2', 'Test 3'], 2*num)
df['Prediction set'] = group  
alignment = np.tile(['MSMAll', 'MSMAll', 'MSMAll', 'CHA', 'CHA', 'CHA'], num)
df['Alignment'] = alignment 

sns.set(style="whitegrid")
ax = sns.catplot(x="Prediction set", y="MSE",
                hue="Alignment", col="Measure",
                data=df, kind="bar", legend=False, legend_out=False,
                height=3.5, aspect=1,
                palette=['#FFD700','#7F00FF'])
(ax.set_titles("{col_name}"))
   #.set_xticklabels(["T1", "T2", "T3"])
   #.set(ylim=(0, 1))
   #.despine(left=True)) 
plt.tight_layout()
plt.legend(loc='upper left')
plt.subplots_adjust(wspace = 0.15) # wspace=None, hspace=None
plt.savefig('/Users/Farzad/Desktop/Figures/catplot_global_coarse.pdf') 
plt.show()        

#%% boxplot (global measures, coarse scale)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string

sns.set_style("white", {'axes.grid':False})

datasetA1 = lam[[1,2,3],:].T; datasetA2 = lam[[5,6,7],:].T
datasetB1 = glb[[1,2,3],:].T; datasetB2 = glb[[5,6,7],:].T
datasetC1 = clc[[1,2,3],:].T; datasetC2 = clc[[5,6,7],:].T
datasetD1 = tra[[1,2,3],:].T; datasetD2 = tra[[5,6,7],:].T
datasetE1 = ass[[1,2,3],:].T; datasetE2 = ass[[5,6,7],:].T
datasetF1 = mod[[1,2,3],:].T; datasetF2 = mod[[5,6,7],:].T

ticks = ['Test 1', 'Test 2', 'Test 3']

dfA1 = pd.DataFrame(datasetA1, columns=ticks)
dfA2 = pd.DataFrame(datasetA2, columns=ticks)
dfB1 = pd.DataFrame(datasetB1, columns=ticks)
dfB2 = pd.DataFrame(datasetB2, columns=ticks)
dfC1 = pd.DataFrame(datasetC1, columns=ticks)
dfC2 = pd.DataFrame(datasetC2, columns=ticks)
dfD1 = pd.DataFrame(datasetD1, columns=ticks)
dfD2 = pd.DataFrame(datasetD2, columns=ticks)
dfE1 = pd.DataFrame(datasetE1, columns=ticks)
dfE2 = pd.DataFrame(datasetE2, columns=ticks)
dfF1 = pd.DataFrame(datasetF1, columns=ticks)
dfF2 = pd.DataFrame(datasetF2, columns=ticks)

names = []
valsA1, xsA1, valsA2, xsA2 = [],[], [],[]
valsB1, xsB1, valsB2, xsB2 = [],[], [],[]
valsC1, xsC1, valsC2, xsC2 = [],[], [],[]
valsD1, xsD1, valsD2, xsD2 = [],[], [],[]
valsE1, xsE1, valsE2, xsE2 = [],[], [],[]
valsF1, xsF1, valsF2, xsF2 = [],[], [],[]

for i, col in enumerate(dfA1.columns):
    valsA1.append(dfA1[col].values)
    valsA2.append(dfA2[col].values)
    valsB1.append(dfB1[col].values)
    valsB2.append(dfB2[col].values)
    valsC1.append(dfC1[col].values)
    valsC2.append(dfC2[col].values)
    valsD1.append(dfD1[col].values)
    valsD2.append(dfD2[col].values)
    valsE1.append(dfE1[col].values)
    valsE2.append(dfE2[col].values)
    valsF1.append(dfF1[col].values)
    valsF2.append(dfF2[col].values)
    names.append(col)
    # Add some random "jitter" to the data points
    xsA1.append(np.random.normal(i*3-0.45, 0.07, dfA1[col].values.shape[0]))
    xsA2.append(np.random.normal(i*3+0.45, 0.07, dfA2[col].values.shape[0]))
    xsB1.append(np.random.normal(i*3-0.45, 0.07, dfB1[col].values.shape[0]))
    xsB2.append(np.random.normal(i*3+0.45, 0.07, dfB2[col].values.shape[0]))
    xsC1.append(np.random.normal(i*3-0.45, 0.07, dfC1[col].values.shape[0]))
    xsC2.append(np.random.normal(i*3+0.45, 0.07, dfC2[col].values.shape[0]))
    xsD1.append(np.random.normal(i*3-0.45, 0.07, dfD1[col].values.shape[0]))
    xsD2.append(np.random.normal(i*3+0.45, 0.07, dfD2[col].values.shape[0]))
    xsE1.append(np.random.normal(i*3-0.45, 0.07, dfE1[col].values.shape[0]))
    xsE2.append(np.random.normal(i*3+0.45, 0.07, dfE2[col].values.shape[0]))
    xsF1.append(np.random.normal(i*3-0.45, 0.07, dfF1[col].values.shape[0]))
    xsF2.append(np.random.normal(i*3+0.45, 0.07, dfF2[col].values.shape[0]))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, figsize=(12, 7))

bpA1 = ax1.boxplot(valsA1, labels=names, positions=np.array(range(len(datasetA1[0])))*3-0.45, sym='', widths=.65)
bpA2 = ax1.boxplot(valsA2, labels=names, positions=np.array(range(len(datasetA2[0])))*3+0.45, sym='', widths=.65)
bpB1 = ax2.boxplot(valsB1, labels=names, positions=np.array(range(len(datasetB1[0])))*3-0.45, sym='', widths=.65)
bpB2 = ax2.boxplot(valsB2, labels=names, positions=np.array(range(len(datasetB2[0])))*3+0.45, sym='', widths=.65)
bpC1 = ax3.boxplot(valsC1, labels=names, positions=np.array(range(len(datasetC1[0])))*3-0.45, sym='', widths=.65)
bpC2 = ax3.boxplot(valsC2, labels=names, positions=np.array(range(len(datasetC2[0])))*3+0.45, sym='', widths=.65)
bpD1 = ax4.boxplot(valsD1, labels=names, positions=np.array(range(len(datasetD1[0])))*3-0.45, sym='', widths=.65)
bpD2 = ax4.boxplot(valsD2, labels=names, positions=np.array(range(len(datasetD2[0])))*3+0.45, sym='', widths=.65)
bpE1 = ax5.boxplot(valsE1, labels=names, positions=np.array(range(len(datasetE1[0])))*3-0.45, sym='', widths=.65)
bpE2 = ax5.boxplot(valsE2, labels=names, positions=np.array(range(len(datasetE2[0])))*3+0.45, sym='', widths=.65)
bpF1 = ax6.boxplot(valsF1, labels=names, positions=np.array(range(len(datasetF1[0])))*3-0.45, sym='', widths=.65)
bpF2 = ax6.boxplot(valsF2, labels=names, positions=np.array(range(len(datasetF2[0])))*3+0.45, sym='', widths=.65)
# Optional: change the color of 'boxes', 'whiskers', 'caps', 'medians', and 'fliers'
plt.setp(bpA1['medians'], color='r') # or color='#D7191C' ...
plt.setp(bpA2['medians'], linewidth=1, linestyle='-', color='r')
plt.setp(bpB1['medians'], color='r')
plt.setp(bpB2['medians'], linewidth=1, linestyle='-', color='r')
plt.setp(bpC1['medians'], color='r')
plt.setp(bpC2['medians'], linewidth=1, linestyle='-', color='r')
plt.setp(bpD1['medians'], color='r')
plt.setp(bpD2['medians'], linewidth=1, linestyle='-', color='r')
plt.setp(bpE1['medians'], color='r')
plt.setp(bpE2['medians'], linewidth=1, linestyle='-', color='r')
plt.setp(bpF1['medians'], color='r')
plt.setp(bpF2['medians'], linewidth=1, linestyle='-', color='r')

palette = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'tan', 'orchid', 'cyan', 'gold', 'crimson']

for xA1, xA2, valA1, valA2, c in zip(xsA1, xsA2, valsA1, valsA2, palette):
    ax1.scatter(xA1, valA1, alpha=0.7, color='#FFD700') # plt.plot(xA1, valA1, 'r.', alpha=0.4)
    ax1.scatter(xA2, valA2, alpha=0.7, color='#7F00FF')
    
for xB1, xB2, valB1, valB2, c in zip(xsB1, xsB2, valsB1, valsB2, palette):
    ax2.scatter(xB1, valB1, alpha=0.7, color='#FFD700')
    ax2.scatter(xB2, valB2, alpha=0.7, color='#7F00FF')   
    
for xC1, xC2, valC1, valC2, c in zip(xsC1, xsC2, valsC1, valsC2, palette):
    ax3.scatter(xC1, valC1, alpha=0.7, color='#FFD700')
    ax3.scatter(xC2, valC2, alpha=0.7, color='#7F00FF') 
    
for xD1, xD2, valD1, valD2, c in zip(xsD1, xsD2, valsD1, valsD2, palette):
    ax4.scatter(xD1, valD1, alpha=0.7, color='#FFD700')
    ax4.scatter(xD2, valD2, alpha=0.7, color='#7F00FF')     

for xE1, xE2, valE1, valE2, c in zip(xsE1, xsE2, valsE1, valsE2, palette):
    ax5.scatter(xE1, valE1, alpha=0.7, color='#FFD700')
    ax5.scatter(xE2, valE2, alpha=0.7, color='#7F00FF') 
    
for xF1, xF2, valF1, valF2, c in zip(xsF1, xsF2, valsF1, valsF2, palette):
    ax6.scatter(xF1, valF1, alpha=0.7, color='#FFD700')
    ax6.scatter(xF2, valF2, alpha=0.7, color='#7F00FF')     

# Use the pyplot interface to customize any subplot...
# First subplot
plt.sca(ax1)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Path Length", fontweight='normal', fontsize=14)
#plt.xlabel("Test Sets", fontweight='normal', fontsize=16)
plt.plot([], c='#FFD700', label='MSM-All', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='#7F00FF', label='CHA', marker='o', linestyle='None', markersize=8)
# Statistical annotation
xs1 = np.array([-0.5, 2.5, 5.5])
xs2 = np.array([0.5, 3.5, 6.5])
for x1, x2 in zip(xs1, xs2):  # e.g., column 25%
    y, h, col = max(datasetA1[:,int((x1+x2)/6)].max(), datasetA2[:,int((x1+x2)/6)].max()) + 0.4, 0.12, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=14)      
# Create empty plot with blank marker containing the extra label
#plt.text(20.81, 5.18, "*", ha='center', va='bottom', color=col, size=14, zorder=10) 
#plt.plot([], [], " ", label='Significant Mean ($P\leq 0.05$)', color='black')    
#plt.legend(prop={'size':16}, loc="lower left")
  
# Second subplot
plt.sca(ax2)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Global Efficieny", fontweight='normal', fontsize=14)
#plt.xlabel("Test Sets", fontweight='normal', fontsize=16)
plt.plot([], c='#FFD700', label='MSM-All', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='#7F00FF', label='CHA', marker='o', linestyle='None', markersize=8)
# Statistical annotation
xs1 = np.array([-0.5, 2.5, 5.5])
xs2 = np.array([0.5, 3.5, 6.5])
for x1, x2 in zip(xs1, xs2):  # e.g., column 25%
    y, h, col = max(datasetB1[:,int((x1+x2)/6)].max(), datasetB2[:,int((x1+x2)/6)].max()) + 0.025, 0.007, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=14)
#plt.legend(prop={'size':14}, loc="lower left")

# Third subplot
plt.sca(ax3)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Clustering Coeeficient", fontweight='normal', fontsize=14)
#plt.xlabel("Test Sets", fontweight='normal', fontsize=16)
plt.plot([], c='#FFD700', label='MSM-All', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='#7F00FF', label='CHA', marker='o', linestyle='None', markersize=8)
# Statistical annotation
xs1 = np.array([-0.5, 2.5, 5.5])
xs2 = np.array([0.5, 3.5, 6.5])
for x1, x2 in zip(xs1, xs2):  # e.g., column 25%
    y, h, col = max(datasetC1[:,int((x1+x2)/6)].max(), datasetC2[:,int((x1+x2)/6)].max()) + 0.015, 0.005, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=14)  
#plt.legend(prop={'size':14})

# Forth subplot
plt.sca(ax4)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Transitivity", fontweight='normal', fontsize=14)
#plt.xlabel("Test Sets", fontweight='normal', fontsize=16)
plt.plot([], c='#FFD700', label='MSM-All', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='#7F00FF', label='CHA', marker='o', linestyle='None', markersize=8)
# Statistical annotation
xs1 = np.array([-0.5, 2.5, 5.5])
xs2 = np.array([0.5, 3.5, 6.5])
for x1, x2 in zip(xs1, xs2):  # e.g., column 25%
    y, h, col = max(datasetD1[:,int((x1+x2)/6)].max(), datasetD2[:,int((x1+x2)/6)].max()) + 0.022, 0.006, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=14)  
#plt.legend(prop={'size':14})

# Fifth subplot
plt.sca(ax5)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Assortativity", fontweight='normal', fontsize=14)
#plt.xlabel("Test Sets", fontweight='normal', fontsize=16)
plt.plot([], c='#FFD700', label='MSM-All', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='#7F00FF', label='CHA', marker='o', linestyle='None', markersize=8)
# Statistical annotation
xs1 = np.array([-0.5, 2.5, 5.5])
xs2 = np.array([0.5, 3.5, 6.5])
for x1, x2 in zip(xs1, xs2):  # e.g., column 25%
    y, h, col = max(datasetE1[:,int((x1+x2)/6)].max(), datasetE2[:,int((x1+x2)/6)].max()) + 0.05, 0.012, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=14)  
#plt.legend(prop={'size':14})

# Sixth subplot
plt.sca(ax6)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Modularity (single-layer)", fontweight='normal', fontsize=14)
#plt.xlabel("Test Sets", fontweight='normal', fontsize=16)
plt.plot([], c='#FFD700', label='MSM-All', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='#7F00FF', label='CHA', marker='o', linestyle='None', markersize=8)
# Statistical annotation
xs1 = np.array([-0.5, 2.5, 5.5])
xs2 = np.array([0.5, 3.5, 6.5])
for x1, x2 in zip(xs1, xs2):  # e.g., column 25%
    y, h, col = max(datasetF1[:,int((x1+x2)/6)].max(), datasetF2[:,int((x1+x2)/6)].max()) + 0.014, 0.004, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=14)  
#plt.legend(prop={'size':14})

# Unified legend  
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

# Annotate Subplots in a Figure with A, B, C 
for n, ax in enumerate((ax1, ax2, ax3, ax4, ax5, ax6)):
    ax.text(-0.2, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=18, weight='bold')

sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)

# Adjust the layout of the plot
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/Figures/Global_Boxplot.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300) 
plt.show() 

#%% shaded ERROR BAR (global measures, coarse scale)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns
import string

mean_A1 = np.mean(stg_l[2,:,:], axis=0)
std_A1 = np.std(stg_l[2,:,:], axis=0)
mean_A2 = np.mean(stg_l[6,:,:], axis=0)
std_A2 = np.std(stg_l[6,:,:], axis=0)

mean_B1 = np.mean(eig_l[2,:,:], axis=0)
std_B1 = np.std(eig_l[2,:,:], axis=0)
mean_B2 = np.mean(eig_l[6,:,:], axis=0)
std_B2 = np.std(eig_l[6,:,:], axis=0)

mean_C1 = np.mean(clc_l[2,:,:], axis=0)
std_C1 = np.std(clc_l[2,:,:], axis=0)
mean_C2 = np.mean(clc_l[6,:,:], axis=0)
std_C2 = np.std(clc_l[6,:,:], axis=0)

mean_D1 = np.mean(kco_l[2,:,:], axis=0)
std_D1 = np.std(kco_l[2,:,:], axis=0)
mean_D2 = np.mean(kco_l[6,:,:], axis=0)
std_D2 = np.std(kco_l[6,:,:], axis=0)

x = np.arange(len(mean_A1))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, figsize=(12, 10))

g1 = 'MSM-All'; g2 = 'CHA'; c1 = '#FFD700'; c2 = '#7F00FF'
# general plot settings
split = np.array([-0.5, 5.5, 59.5, 98.5, 154.5, 177.5, 200.5, 250.5, 265.5, 342.5, 349.5, 353.5, 359.5])
color = ['#0020FF', '#7830F0', '#3EFCFD', '#B51DB4', '#00F300', '#009091', 
         '#FFFE16', '#FB64FE', '#FF2E00', '#C47A31', '#FFB300', '#5A9B00']
labels = ['Primary Visual', 'Secondary Visual', 'Somatomotor', 'Cingulo-Opercular', 'Dorsal Attention', 'Language',
          'Frontoparietal', 'Auditory', 'Default Mode', 'Posterior Multimodal', 'Ventral Multimodal', 'Orbito-Affective']

plt.sca(ax1)
ebA1 = ax1.plot(x, mean_A1, '-ko', label=g1, markerfacecolor=c1, linewidth=0.8, markersize=3, markeredgewidth=0.5)
ax1.fill_between(x, mean_A1 - std_A1, mean_A1 + std_A1, color=c1, alpha=0.3)
ebA2 = ax1.plot(x, mean_A2, '-ko', label=g2, markerfacecolor=c2, linewidth=0.8, markersize=3, markeredgewidth=0.5)
ax1.fill_between(x, mean_A2 - std_A2, mean_A2 + std_A2, color=c2, alpha=0.3)
plt.ylabel("Degree", fontweight='normal', fontsize=10)
ax1.get_yaxis().set_label_coords(-0.04,0.5) # Aligning y-axis labels
plt.axvline(x=999.5, color='k', linestyle='-', linewidth=1.5) # l/r separator -> "NECESSARY FOR rectangle patcehs -> clip_on=False"
"""# significance
plt.axvline(x=15-1, color='r', linestyle='--', linewidth=1.5)
"""
# Add rectangle objects as tick labels
plt.xlim([-1, 360])
y_min, y_max = ax1.get_ylim()
h = (y_max-y_min)/15; space = h/5; i = y_min - h # intercept
xy = split[:-1] # anchor points
w = split[1:] - xy # rectangle width(s)
# ticks and labels along the bottom edge are off
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.tick_params(axis='y', labelsize=8)
for j in range(len(xy)): # plot rectangles one-by-one
    ax1.add_patch(patches.Rectangle((xy[j], i), width=w[j], height=h, facecolor=color[j], clip_on=False, linewidth=0.4, edgecolor='k'))

plt.sca(ax2)
ebB1 = ax2.plot(x, mean_B1, '-ko', label=g1, markerfacecolor=c1, linewidth=0.8, markersize=3, markeredgewidth=0.5)
ax2.fill_between(x, mean_B1 - std_B1, mean_B1 + std_B1, color=c1, alpha=0.3)
ebB2 = ax2.plot(x, mean_B2, '-ko', label=g2, markerfacecolor=c2, linewidth=0.8, markersize=3, markeredgewidth=0.5)
ax2.fill_between(x, mean_B2 - std_B2, mean_B2 + std_B2, color=c2, alpha=0.3)
plt.ylabel("Eigenector Centrality", fontweight='normal', fontsize=10)
ax2.get_yaxis().set_label_coords(-0.04,0.5)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.axvline(x=999.5, color='k', linestyle='-', linewidth=1.5) # l/r separator
"""# significance
plt.axvline(x=99-1, color='r', linestyle='--', linewidth=1.5)
"""
# Add rectangle objects as tick labels
plt.xlim([-1, 360])
y_min, y_max = ax2.get_ylim()
h = (y_max-y_min)/15; space = h/5; i = y_min - h # intercept
xy = split[:-1] # anchor points
w = split[1:] - xy # rectangle width(s)
# ticks and labels along the bottom edge are off
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.tick_params(axis='y', labelsize=8)
for j in range(len(xy)): # plot rectangles one-by-one
    ax2.add_patch(patches.Rectangle((xy[j], i), width=w[j], height=h, facecolor=color[j], clip_on=False, linewidth=0.4, edgecolor='k'))

plt.sca(ax3)
ebC1 = ax3.plot(x, mean_C1, '-ko', label=g1, markerfacecolor=c1, linewidth=0.8, markersize=3, markeredgewidth=0.5)
ax3.fill_between(x, mean_C1 - std_C1, mean_C1 + std_C1, color=c1, alpha=0.3)
ebC2 = ax3.plot(x, mean_C2, '-ko', label=g2, markerfacecolor=c2, linewidth=0.8, markersize=3, markeredgewidth=0.5)
ax3.fill_between(x, mean_C2 - std_C2, mean_C2 + std_C2, color=c2, alpha=0.3)
plt.ylabel("Clustering Coefficient", fontweight='normal', fontsize=10)
ax3.get_yaxis().set_label_coords(-0.04,0.5)
plt.axvline(x=999.5, color='k', linestyle='-', linewidth=1.5) # l/r separator
"""# significance
plt.axvline(x=94-1, color='r', linestyle='--', linewidth=1.5)
"""
# Add rectangle objects as tick labels
plt.xlim([-1, 360])
y_min, y_max = ax3.get_ylim()
h = (y_max-y_min)/15; space = h/5; i = y_min - h # intercept
xy = split[:-1] # anchor points
w = split[1:] - xy # rectangle width(s)
# ticks and labels along the bottom edge are off
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.tick_params(axis='y', labelsize=8)
for j in range(len(xy)): # plot rectangles one-by-one
    ax3.add_patch(patches.Rectangle((xy[j], i), width=w[j], height=h, facecolor=color[j], clip_on=False, linewidth=0.4, edgecolor='k'))

plt.sca(ax4)
ebD1 = ax4.plot(x, mean_D1, '-ko', label=g1, markerfacecolor=c1, linewidth=0.8, markersize=3, markeredgewidth=0.5)
ax4.fill_between(x, mean_D1 - std_D1, mean_D1 + std_D1, color=c1, alpha=0.3)
ebD2 = ax4.plot(x, mean_D2, '-ko', label=g2, markerfacecolor=c2, linewidth=0.8, markersize=3, markeredgewidth=0.5)
ax4.fill_between(x, mean_D2 - std_D2, mean_D2 + std_D2, color=c2, alpha=0.3)
plt.ylabel("Local Efficiency", fontweight='normal', fontsize=10)
ax4.get_yaxis().set_label_coords(-0.04,0.5)
plt.axvline(x=999.5, color='k', linestyle='-', linewidth=1.5) # l/r separator
"""# significance
plt.axvline(x=15-1, color='r', linestyle='--', linewidth=1.5, label='Significant Variation')
"""
# Add rectangle objects as tick labels
plt.xlim([-1, 360])
y_min, y_max = ax4.get_ylim()
h = (y_max-y_min)/15; space = h/5; i = y_min - h # intercept
xy = split[:-1] # anchor points
w = split[1:] - xy # rectangle width(s)
# ticks and labels along the bottom edge are off
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.tick_params(axis='y', labelsize=8)
for j in range(len(xy)): # plot rectangles one-by-one
    ax4.add_patch(patches.Rectangle((xy[j], i), width=w[j], height=h, facecolor=color[j], clip_on=False, linewidth=0.4, edgecolor='k', label=labels[j]))

plt.legend(prop={'size':9}, ncol=7, frameon=False, bbox_to_anchor=(.48, -.06), loc='upper center')

# Annotate Subplots in a Figure with A, B, C, D (as well as L & R)
for n, ax in enumerate((ax1, ax2, ax3, ax4)):
    ax.text(-0.07, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=14, weight='bold')
    #ax.text(0.258, 1.015, 'L', transform=ax.transAxes, 
            #size=14, weight='regular')
    #ax.text(0.731, 1.015, 'R', transform=ax.transAxes, 
            #size=14, weight='regular')

sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)

# Adjust the layout of the plot
plt.tight_layout()

plt.savefig('/Users/Farzad/Desktop/Figures/ShadedErrorbar.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300) 

plt.show()

#%% Fingerprinting and Prediction (Coarse-scale)
from scipy import stats
from brainconn import similarity
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
n_sbj = 30

# load timeseries
ts_rest1LR = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/ts_rest1LR_30sbj.npy') 
ts_rest1RL = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/ts_rest1RL_30sbj.npy')
ts_rest2LR = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/ts_rest2LR_30sbj.npy')
ts_rest2RL = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/ts_rest2RL_30sbj.npy')
ts_aligned_rest1LR = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/ts_aligned_rest1LR_30sbj.npy')
ts_aligned_rest1RL = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/ts_aligned_rest1RL_30sbj.npy')
ts_aligned_rest2LR = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/ts_aligned_rest2LR_30sbj.npy')
ts_aligned_rest2RL = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/ts_aligned_rest2RL_30sbj.npy')

#% CALCULATE CORRELATION (whole brain)
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional

corr_rest1LR = abs(correlation_measure.fit_transform(ts_rest1LR))
corr_rest1RL = abs(correlation_measure.fit_transform(ts_rest1RL))
corr_rest2LR = abs(correlation_measure.fit_transform(ts_rest2LR))
corr_rest2RL = abs(correlation_measure.fit_transform(ts_rest2RL))
corr_aligned_rest1LR = abs(correlation_measure.fit_transform(ts_aligned_rest1LR))
corr_aligned_rest1RL = abs(correlation_measure.fit_transform(ts_aligned_rest1RL))
corr_aligned_rest2LR = abs(correlation_measure.fit_transform(ts_aligned_rest2LR))
corr_aligned_rest2RL = abs(correlation_measure.fit_transform(ts_aligned_rest2RL))

adj_wei = [[] for i in range(8)] # 8 sets (list of lists; wrong way -> adj_wei = [[]] * 8)
tril = [[] for i in range(8)]
for k in range(n_sbj): 
    np.fill_diagonal(corr_rest1LR[k], 0)
    np.fill_diagonal(corr_rest1RL[k], 0)
    np.fill_diagonal(corr_rest2LR[k], 0)
    np.fill_diagonal(corr_rest2RL[k], 0)
    np.fill_diagonal(corr_aligned_rest1LR[k], 0)
    np.fill_diagonal(corr_aligned_rest1RL[k], 0)
    np.fill_diagonal(corr_aligned_rest2LR[k], 0)
    np.fill_diagonal(corr_aligned_rest2RL[k], 0)
    # weighted
    adj_wei[0].append(corr_rest1LR[k])
    adj_wei[1].append(corr_rest1RL[k])
    adj_wei[2].append(corr_rest2LR[k])
    adj_wei[3].append(corr_rest2RL[k])
    adj_wei[4].append(corr_aligned_rest1LR[k])
    adj_wei[5].append(corr_aligned_rest1RL[k])
    adj_wei[6].append(corr_aligned_rest2LR[k])
    adj_wei[7].append(corr_aligned_rest2RL[k])
    # lower triangular
    tril[0].append(corr_rest1LR[k][np.tril_indices(corr_rest1LR.shape[2], k = -1)]) # K = -1 [without diagonal] or 0 [with]
    tril[1].append(corr_rest1RL[k][np.tril_indices(corr_rest1RL.shape[2], k = -1)])
    tril[2].append(corr_rest2LR[k][np.tril_indices(corr_rest2LR.shape[2], k = -1)])
    tril[3].append(corr_rest2RL[k][np.tril_indices(corr_rest2RL.shape[2], k = -1)])
    tril[4].append(corr_aligned_rest1LR[k][np.tril_indices(corr_aligned_rest1LR.shape[2], k = -1)])
    tril[5].append(corr_aligned_rest1RL[k][np.tril_indices(corr_aligned_rest1RL.shape[2], k = -1)])
    tril[6].append(corr_aligned_rest2LR[k][np.tril_indices(corr_aligned_rest2LR.shape[2], k = -1)])
    tril[7].append(corr_aligned_rest2RL[k][np.tril_indices(corr_aligned_rest2RL.shape[2], k = -1)])    

tril = np.array(tril)
    
#% SIMILARITY (i.e.,correlation coefficient between two flattened adjacency matrices) 
sim = np.zeros((n_sbj,n_sbj))
for k in range(n_sbj):
    for l in range(n_sbj):
        #sim[k,l] = similarity.corr_flat_und(adj_wei[0][k], adj_wei[1][l])
        sim[k,l] = np.corrcoef(tril[6][k], tril[7][l])[0, 1] # faster
# Get index for the highest value
index = sim.argsort()[:,-1]
# binarize
for k in range(n_sbj):
    sim[k,:] = [0 if i < sim[k,index[k]] else 1 for i in sim[k,:]]
# plot        
plt.imshow(sim)
plt.colorbar()

# REGRESSION ANALYSIS based on FC patterns (lower triangular)
# Loading questionnaires's indices
IQ = []
file_in = open('/Volumes/Elements/Hyperalignment/HCP/IQ.txt', 'r')
for z in file_in.read().split('\n'):
    IQ.append(float(z))
questionnaire = {'IQ': np.array(IQ)}
questionnaire_list = ['IQ']
# Set up a correlation vector (measure * Q_index) corresponding to a value for each region in the parcellation.
index = 'IQ' # Questionnaire index {AM, ESS, PW}
# Building model
mse_reg, r2_reg = [], []
pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
for s in range(np.shape(pred_set)[0]):
    # Training/testing sets and target variable
    X_train, y_train = tril[pred_set[s][0],:,:], questionnaire[index]
    X_test, y_test = tril[pred_set[s][1],:,:], questionnaire[index]        
    # Create linear regression object
    reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization        
    # Train the model using the training sets
    reg.fit(X_train, y_train)        
    # Make predictions using the testing set
    y_pred_reg = reg.predict(X_test)       
    # The mean squared error
    mse_reg.append(mean_squared_error(y_test, y_pred_reg))        
    # The coefficient of determination: 1 is perfect prediction
    r2_reg.append(r2_score(y_test, y_pred_reg))

#%% NEW METHOD 1: pairwise fine-scale corrlations across time-series -> not good

# qsub -t 1:240 fc_1.sh

import os
import pandas as pd
import numpy as np
import time, timeit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score


# idx = int(os.getenv("SGE_TASK_ID"))-1
idx = 0 # 0 <= idx < 240

#path_coarse = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/ts_rest1LR_30sbj.npy'
path_fine = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries_regional/mmp/'
#path_fine = '/dcl01/smart/data/fvfarahani/searchlight/timeseries_regional/'
runs = ['rest1LR_', 'rest1RL_', 'rest2LR_', 'rest2RL_',
            'aligned_rest1LR_', 'aligned_rest1RL_', 'aligned_rest2LR_', 'aligned_rest2RL_']
n_run = len(runs)
n_sbj = 30
n_roi = 360
N = 1200 # Get number of rows in either A or B (i.e., dimension with simialr size)

# auxilliary dataframe for getting the current number of runs and subjects
df = pd.DataFrame({'run':np.repeat(runs, n_sbj, axis=0),
                   'sbj':np.tile(np.arange(n_sbj), n_run)})

run, sbj = df['run'][idx], df['sbj'][idx]

fc = np.zeros((n_roi,n_roi))

start_time = time.time() # ~12 hours (42220 sec) run for each subject; ~3000-7000 seconds in JHEPCE

for r1 in range(1,n_roi+1):
    i = r1-1
    A = np.load(path_fine + 'ts_' + run + 'mmp_' + str(r1) + '.npy')[sbj]
    for r2 in range(1,n_roi+1):
        if r2 <= r1: # need one triangular for comutation (matrix is symmetric)  
            j = r2-1        
            B = np.load(path_fine + 'ts_' + run + 'mmp_' + str(r2) + '.npy')[sbj]            
            # Store columnw-wise sum in A and B, as they would be used at few places
            sA = A.sum(0)
            sB = B.sum(0)
            # Basically there are four parts in the main formula, compute them one-by-one
            p1 = N*np.einsum('ij,ik->kj',A,B)
            p2 = sA*sB[:,None]
            p3 = N*((B**2).sum(0)) - (sB**2)
            p4 = N*((A**2).sum(0)) - (sA**2)
            # Finally compute Pearson Correlation Coefficient as 2D array 
            fc[i,j] = ((p1 - p2)/np.sqrt(p4*p3[:,None])).mean()

# add the upper triangular and devide the diagonal by 2 cause it is counted twice
fc = fc + fc.T
fc[np.diag_indices_from(fc)] /= 2

np.save('/Users/Farzad/Desktop/fc_' + run + str(sbj+1), fc)
#np.save('/dcl01/smart/data/fvfarahani/searchlight/fc_1/fc_' + run + str(sbj+1), fc)

print('fc_' + run + str(sbj+1))            
end = timeit.timeit()
print("--- %s seconds ---" % (time.time() - start_time)) 


#####################################################
# After calculating FCs for each idx through JHEPCE #
#####################################################

fc = [[] for i in range(8)] # 8 sets 
tril = [[] for i in range(8)] # 8 sets 
fc_path = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/fc_1/'
for k in range(n_sbj):
    fc[0].append(np.load(fc_path + 'fc_rest1LR_' + str(k+1) + '.npy'))
    fc[1].append(np.load(fc_path + 'fc_rest1RL_' + str(k+1) + '.npy'))
    fc[2].append(np.load(fc_path + 'fc_rest2LR_' + str(k+1) + '.npy'))
    fc[3].append(np.load(fc_path + 'fc_rest2RL_' + str(k+1) + '.npy'))
    fc[4].append(np.load(fc_path + 'fc_aligned_rest1LR_' + str(k+1) + '.npy'))
    fc[5].append(np.load(fc_path + 'fc_aligned_rest1RL_' + str(k+1) + '.npy'))
    fc[6].append(np.load(fc_path + 'fc_aligned_rest2LR_' + str(k+1) + '.npy'))
    fc[7].append(np.load(fc_path + 'fc_aligned_rest2RL_' + str(k+1) + '.npy'))
    # lower triangular
    tril[0].append(fc[0][k][np.tril_indices(fc[0][k].shape[0], k = 0)]) # K = -1 [without diagonal] or 0 [with]
    tril[1].append(fc[1][k][np.tril_indices(fc[1][k].shape[0], k = 0)])
    tril[2].append(fc[2][k][np.tril_indices(fc[2][k].shape[0], k = 0)])
    tril[3].append(fc[3][k][np.tril_indices(fc[3][k].shape[0], k = 0)])
    tril[4].append(fc[4][k][np.tril_indices(fc[4][k].shape[0], k = 0)])
    tril[5].append(fc[5][k][np.tril_indices(fc[5][k].shape[0], k = 0)])
    tril[6].append(fc[6][k][np.tril_indices(fc[6][k].shape[0], k = 0)])
    tril[7].append(fc[7][k][np.tril_indices(fc[7][k].shape[0], k = 0)]) 

tril = np.array(tril)
    
#% SIMILARITY (i.e.,correlation coefficient between two flattened adjacency matrices) 
sim = np.zeros((n_sbj,n_sbj))
for k in range(n_sbj):
    for l in range(n_sbj):
        #sim[k,l] = similarity.corr_flat_und(adj_wei[0][k], adj_wei[1][l])
        sim[k,l] = np.corrcoef(tril[0][k], tril[1][l])[0, 1] # faster
# Get index for the highest value
index = sim.argsort()[:,-1]
# binarize
for k in range(n_sbj):
    sim[k,:] = [0 if i < sim[k,index[k]] else 1 for i in sim[k,:]]
# plot        
plt.imshow(sim)
plt.colorbar()

# REGRESSION ANALYSIS based on FC patterns (lower triangular)
# Loading questionnaires's indices
IQ = []
file_in = open('/Volumes/Elements/Hyperalignment/HCP/IQ.txt', 'r')
for z in file_in.read().split('\n'):
    IQ.append(float(z))
questionnaire = {'IQ': np.array(IQ)}
questionnaire_list = ['IQ']
# Set up a correlation vector (measure * Q_index) corresponding to a value for each region in the parcellation.
index = 'IQ' # Questionnaire index {AM, ESS, PW}
# Building model
mse_reg, r2_reg = [], []
pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
for s in range(np.shape(pred_set)[0]):
    # Training/testing sets and target variable
    X_train, y_train = tril[pred_set[s][0]], questionnaire[index]
    X_test, y_test = tril[pred_set[s][1]], questionnaire[index]        
    # Create linear regression object
    reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization        
    # Train the model using the training sets
    reg.fit(X_train, y_train)        
    # Make predictions using the testing set
    y_pred_reg = reg.predict(X_test)       
    # The mean squared error
    mse_reg.append(mean_squared_error(y_test, y_pred_reg))        
    # The coefficient of determination: 1 is perfect prediction
    r2_reg.append(r2_score(y_test, y_pred_reg))

#%% NEW METHOD 2: A * A.T (correlation between vertices of a region, and mean timeseries of all regions -> A)

# qsub -t 1:240 fc_2.sh

import os
import pandas as pd
import numpy as np
import time, timeit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats.stats import pearsonr
from brainconn import similarity

# idx = int(os.getenv("SGE_TASK_ID"))-1
idx = 0 # 0 <= idx < 240

path_coarse = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/'
#path_coarse = '/dcl01/smart/data/fvfarahani/searchlight/timeseries/'
path_fine = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries_regional/mmp/'
#path_fine = '/dcl01/smart/data/fvfarahani/searchlight/timeseries_regional/'
runs = ['rest1LR_', 'rest1RL_', 'rest2LR_', 'rest2RL_',
            'aligned_rest1LR_', 'aligned_rest1RL_', 'aligned_rest2LR_', 'aligned_rest2RL_']
n_run = len(runs)
n_sbj = 30
n_roi = 360
N = 1200 # Get number of rows in either A or B (i.e., dimension with simialr size)

# auxilliary dataframe for getting the current number of runs and subjects
df = pd.DataFrame({'run':np.repeat(runs, n_sbj, axis=0),
                   'sbj':np.tile(np.arange(n_sbj), n_run)})

run, sbj = df['run'][idx], df['sbj'][idx]

fc = np.zeros((n_roi,n_roi))

start_time = time.time()       
        
A = np.load(path_coarse + 'ts_' + run + '30sbj.npy')[sbj]    
# ~4000-8000 seconds in JHEPCE
for r1 in range(1,n_roi+1):
    i = r1-1
    B = np.load(path_fine + 'ts_' + run + 'mmp_' + str(r1) + '.npy')[sbj]    
    # Store columnw-wise sum in A and B, as they would be used at few places
    sA = A.sum(0)
    sB = B.sum(0)
    # Basically there are four parts in the main formula, compute them one-by-one
    p1 = N*np.einsum('ij,ik->kj',A,B)
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)
    # Finally compute Pearson Correlation Coefficient as 2D array 
    pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None])).T
    sym1 = pcorr@pcorr.T
    tril1 = sym1[np.tril_indices(sym1.shape[0], k = 0)] # K = -1 [without diagonal] or 0 [with]
    
    for r2 in range(1,n_roi+1):
        if r2 < r1: # just need one triangular (matrix is symmetric)               
            j = r2-1
            B = np.load(path_fine + 'ts_' + run + 'mmp_' + str(r2) + '.npy')[sbj]
            # Store columnw-wise sum in A and B, as they would be used at few places
            sA = A.sum(0)
            sB = B.sum(0)
            # Basically there are four parts in the main formula, compute them one-by-one
            p1 = N*np.einsum('ij,ik->kj',A,B)
            p2 = sA*sB[:,None]
            p3 = N*((B**2).sum(0)) - (sB**2)
            p4 = N*((A**2).sum(0)) - (sA**2)
            # Finally compute Pearson Correlation Coefficient as 2D array 
            pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None])).T
            sym2 = pcorr@pcorr.T
            tril2 = sym2[np.tril_indices(sym2.shape[0], k = 0)]
            
            # SIMILARITY of two matrices
            #fc[idx,i,j] = similarity.corr_flat_und(sym1, sym2)                    
            #fc[idx,i,j] = np.corrcoef(tril1, tril2)[0, 1] 
            fc[i,j] = pearsonr(tril1, tril2)[0] # a little faster than 1st approach, similar to second
            
# add the upper triangular
fc = fc + fc.T
np.fill_diagonal(fc, 1) # diagonal is always 1

#np.save('/Users/Farzad/Desktop/fc_' + run + str(sbj+1), fc)
np.save('/dcl01/smart/data/fvfarahani/searchlight/fc_2/fc_' + run + str(sbj+1), fc)

print('fc_' + run + str(sbj+1))            
end = timeit.timeit()
print("--- %s seconds ---" % (time.time() - start_time)) 

#####################################################
# After calculating FCs for each idx through JHEPCE #
#####################################################

fc = [[] for i in range(8)] # 8 sets 
tril = [[] for i in range(8)] # 8 sets 
fc_path = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/fc_2/'
for k in range(n_sbj):
    fc[0].append(np.load(fc_path + 'fc_rest1LR_' + str(k+1) + '.npy'))
    fc[1].append(np.load(fc_path + 'fc_rest1RL_' + str(k+1) + '.npy'))
    fc[2].append(np.load(fc_path + 'fc_rest2LR_' + str(k+1) + '.npy'))
    fc[3].append(np.load(fc_path + 'fc_rest2RL_' + str(k+1) + '.npy'))
    fc[4].append(np.load(fc_path + 'fc_aligned_rest1LR_' + str(k+1) + '.npy'))
    fc[5].append(np.load(fc_path + 'fc_aligned_rest1RL_' + str(k+1) + '.npy'))
    fc[6].append(np.load(fc_path + 'fc_aligned_rest2LR_' + str(k+1) + '.npy'))
    fc[7].append(np.load(fc_path + 'fc_aligned_rest2RL_' + str(k+1) + '.npy'))
    # lower triangular
    tril[0].append(fc[0][k][np.tril_indices(fc[0][k].shape[0], k = -1)]) # K = -1 [without diagonal] or 0 [with]
    tril[1].append(fc[1][k][np.tril_indices(fc[1][k].shape[0], k = -1)])
    tril[2].append(fc[2][k][np.tril_indices(fc[2][k].shape[0], k = -1)])
    tril[3].append(fc[3][k][np.tril_indices(fc[3][k].shape[0], k = -1)])
    tril[4].append(fc[4][k][np.tril_indices(fc[4][k].shape[0], k = -1)])
    tril[5].append(fc[5][k][np.tril_indices(fc[5][k].shape[0], k = -1)])
    tril[6].append(fc[6][k][np.tril_indices(fc[6][k].shape[0], k = -1)])
    tril[7].append(fc[7][k][np.tril_indices(fc[7][k].shape[0], k = -1)]) 

tril = np.array(tril)
    
#% SIMILARITY (i.e.,correlation coefficient between two flattened adjacency matrices) 
sim = np.zeros((n_sbj,n_sbj))
for k in range(n_sbj):
    for l in range(n_sbj):
        #sim[k,l] = similarity.corr_flat_und(adj_wei[0][k], adj_wei[1][l])
        sim[k,l] = np.corrcoef(tril[0][k], tril[1][l])[0, 1] # faster
# Get index for the highest value
index = sim.argsort()[:,-1]
# binarize
for k in range(n_sbj):
    sim[k,:] = [0 if i < sim[k,index[k]] else 1 for i in sim[k,:]]
# plot        
plt.imshow(sim)
plt.colorbar()

# REGRESSION ANALYSIS based on FC patterns (lower triangular)
# Loading questionnaires's indices
IQ = []
file_in = open('/Volumes/Elements/Hyperalignment/HCP/IQ.txt', 'r')
for z in file_in.read().split('\n'):
    IQ.append(float(z))
questionnaire = {'IQ': np.array(IQ)}
questionnaire_list = ['IQ']
# Set up a correlation vector (measure * Q_index) corresponding to a value for each region in the parcellation.
index = 'IQ' # Questionnaire index {AM, ESS, PW}
# Building model
mse_reg, r2_reg = [], []
pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
for s in range(np.shape(pred_set)[0]):
    # Training/testing sets and target variable
    X_train, y_train = tril[pred_set[s][0]], questionnaire[index]
    X_test, y_test = tril[pred_set[s][1]], questionnaire[index]        
    # Create linear regression object
    reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization        
    # Train the model using the training sets
    reg.fit(X_train, y_train)        
    # Make predictions using the testing set
    y_pred_reg = reg.predict(X_test)       
    # The mean squared error
    mse_reg.append(mean_squared_error(y_test, y_pred_reg))        
    # The coefficient of determination: 1 is perfect prediction
    r2_reg.append(r2_score(y_test, y_pred_reg))

"""
# testing fastest approach for calculating correlations
pcorr1 = []
pcorr2 = []
pcorr3 = []

# First approach
start_time = time.time()
for k in range(n_sbj):
    cp = np.zeros((n_roi, n_rgn)) # cp = np.zeros((target_nmbr,go), dtype="float16")
    for i in range(n_roi):
        for j in range(n_rgn): 
            cp[i,j] = np.corrcoef(A[k][:,i], B[k][:,j])[0, 1]
    pcorr1.append(cp) 
end = timeit.timeit()
print("--- %s seconds ---" % (time.time() - start_time))

# Second approach
start_time = time.time()
for k in range(n_sbj):
    cp = np.zeros((n_roi, n_rgn)) # cp = np.zeros((target_nmbr,go), dtype="float16")
    for i in range(n_roi):
        for j in range(n_rgn): 
            cp[i,j],_ = pearsonr(A[k][:,i], B[k][:,j])
    pcorr2.append(cp) 
end = timeit.timeit()
print("--- %s seconds ---" % (time.time() - start_time))

# Third approach (vectorized approach) --> SUPER FASTER ~250X
# https://stackoverflow.com/questions/33650188/efficient-pairwise-correlation-for-two-matrices-of-features
start_time = time.time()
# Get number of rows in either A or B (i.e., dimension with simialr size)
N = B[0].shape[0]
for k in range(n_sbj):    
    # Store columnw-wise sum in A and B, as they would be used at few places
    sA = A[k].sum(0)
    sB = B[k].sum(0)
    # Basically there are four parts in the main formula, compute them one-by-one
    p1 = N*np.einsum('ij,ik->kj',A[k],B[k])
    p2 = sA*sB[:,None]
    p3 = N*((B[k]**2).sum(0)) - (sB**2)
    p4 = N*((A[k]**2).sum(0)) - (sA**2)
    # Finally compute Pearson Correlation Coefficient as 2D array 
    pcorr3.append(((p1 - p2)/np.sqrt(p4*p3[:,None])).T)
end = timeit.timeit()
print("--- %s seconds ---" % (time.time() - start_time))"""


#%% NEW METHOD 3: Spectral Embedding

# qsub -t 1:240 fc_3.sh

import os
import pandas as pd
import numpy as np
import time, timeit
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats.stats import pearsonr
from brainconn import similarity

# idx = int(os.getenv("SGE_TASK_ID"))-1
idx = 0 # 0 <= idx < 240

path_coarse = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/'
#path_coarse = '/dcl01/smart/data/fvfarahani/searchlight/timeseries/'
path_fine = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries_regional/mmp/'
#path_fine = '/dcl01/smart/data/fvfarahani/searchlight/timeseries_regional/'
runs = ['rest1LR_', 'rest1RL_', 'rest2LR_', 'rest2RL_',
            'aligned_rest1LR_', 'aligned_rest1RL_', 'aligned_rest2LR_', 'aligned_rest2RL_']
n_run = len(runs)
n_sbj = 30
n_roi = 360
N = 1200 # Get number of rows in either A or B (i.e., dimension with simialr size)

# auxilliary dataframe for getting the current number of runs and subjects
df = pd.DataFrame({'run':np.repeat(runs, n_sbj, axis=0),
                   'sbj':np.tile(np.arange(n_sbj), n_run)})

run, sbj = df['run'][idx], df['sbj'][idx]

fc = np.zeros((n_roi,n_roi))

start_time = time.time()       
        
A = np.load(path_coarse + 'ts_' + run + '30sbj.npy')[sbj]    
# ~4000-8000 seconds in JHEPCE
for r1 in range(1,n_roi+1):
    i = r1-1
    B = np.load(path_fine + 'ts_' + run + 'mmp_' + str(r1) + '.npy')[sbj]    
    # Store columnw-wise sum in A and B, as they would be used at few places
    sA = A.sum(0)
    sB = B.sum(0)
    # Basically there are four parts in the main formula, compute them one-by-one
    p1 = N*np.einsum('ij,ik->kj',A,B)
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)
    # Finally compute Pearson Correlation Coefficient as 2D array 
    pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None])).T
    embedding = SpectralEmbedding(n_components=2)
    pcorr_transformed = embedding.fit_transform(pcorr)
    eig1 = pcorr_transformed.flatten(order='F') # flatten in column-major
    
    for r2 in range(1,n_roi+1):
        if r2 < r1: # just need one triangular (matrix is symmetric)               
            j = r2-1
            B = np.load(path_fine + 'ts_' + run + 'mmp_' + str(r2) + '.npy')[sbj]
            # Store columnw-wise sum in A and B, as they would be used at few places
            sA = A.sum(0)
            sB = B.sum(0)
            # Basically there are four parts in the main formula, compute them one-by-one
            p1 = N*np.einsum('ij,ik->kj',A,B)
            p2 = sA*sB[:,None]
            p3 = N*((B**2).sum(0)) - (sB**2)
            p4 = N*((A**2).sum(0)) - (sA**2)
            # Finally compute Pearson Correlation Coefficient as 2D array 
            pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None])).T
            embedding = SpectralEmbedding(n_components=2)
            pcorr_transformed = embedding.fit_transform(pcorr)
            eig2 = pcorr_transformed.flatten(order='F') # flatten in column-major
            
            # SIMILARITY of two sets of eigenpams
            fc[i,j] = pearsonr(eig1, eig2)[0]
            
# add the upper triangular
fc = fc + fc.T
np.fill_diagonal(fc, 1) # diagonal is always 1

#np.save('/Users/Farzad/Desktop/fc_' + run + str(sbj+1), fc)
np.save('/dcl01/smart/data/fvfarahani/searchlight/fc_3/fc_' + run + str(sbj+1), fc)

print('fc_' + run + str(sbj+1))            
end = timeit.timeit()
print("--- %s seconds ---" % (time.time() - start_time)) 


#####################################################
# After calculating FCs for each idx through JHEPCE #
#####################################################

fc = [[] for i in range(8)] # 8 sets 
tril = [[] for i in range(8)] # 8 sets 
fc_path = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/fc_3/'
for k in range(n_sbj):
    fc[0].append(np.load(fc_path + 'fc_rest1LR_' + str(k+1) + '.npy'))
    fc[1].append(np.load(fc_path + 'fc_rest1RL_' + str(k+1) + '.npy'))
    fc[2].append(np.load(fc_path + 'fc_rest2LR_' + str(k+1) + '.npy'))
    fc[3].append(np.load(fc_path + 'fc_rest2RL_' + str(k+1) + '.npy'))
    fc[4].append(np.load(fc_path + 'fc_aligned_rest1LR_' + str(k+1) + '.npy'))
    fc[5].append(np.load(fc_path + 'fc_aligned_rest1RL_' + str(k+1) + '.npy'))
    fc[6].append(np.load(fc_path + 'fc_aligned_rest2LR_' + str(k+1) + '.npy'))
    fc[7].append(np.load(fc_path + 'fc_aligned_rest2RL_' + str(k+1) + '.npy'))
    # lower triangular
    tril[0].append(fc[0][k][np.tril_indices(fc[0][k].shape[0], k = -1)]) # K = -1 [without diagonal] or 0 [with]
    tril[1].append(fc[1][k][np.tril_indices(fc[1][k].shape[0], k = -1)])
    tril[2].append(fc[2][k][np.tril_indices(fc[2][k].shape[0], k = -1)])
    tril[3].append(fc[3][k][np.tril_indices(fc[3][k].shape[0], k = -1)])
    tril[4].append(fc[4][k][np.tril_indices(fc[4][k].shape[0], k = -1)])
    tril[5].append(fc[5][k][np.tril_indices(fc[5][k].shape[0], k = -1)])
    tril[6].append(fc[6][k][np.tril_indices(fc[6][k].shape[0], k = -1)])
    tril[7].append(fc[7][k][np.tril_indices(fc[7][k].shape[0], k = -1)]) 

tril = np.array(tril)
    
#% SIMILARITY (i.e.,correlation coefficient between two flattened adjacency matrices) 
sim = np.zeros((n_sbj,n_sbj))
for k in range(n_sbj):
    for l in range(n_sbj):
        #sim[k,l] = similarity.corr_flat_und(adj_wei[0][k], adj_wei[1][l])
        sim[k,l] = np.corrcoef(tril[4][k], tril[5][l])[0, 1] # faster
# Get index for the highest value
index = sim.argsort()[:,-1]
# binarize
for k in range(n_sbj):
    sim[k,:] = [0 if i < sim[k,index[k]] else 1 for i in sim[k,:]]
# plot        
plt.imshow(sim)
plt.colorbar()

# REGRESSION ANALYSIS based on FC patterns (lower triangular)
# Loading questionnaires's indices
IQ = []
file_in = open('/Volumes/Elements/Hyperalignment/HCP/IQ.txt', 'r')
for z in file_in.read().split('\n'):
    IQ.append(float(z))
questionnaire = {'IQ': np.array(IQ)}
questionnaire_list = ['IQ']
# Set up a correlation vector (measure * Q_index) corresponding to a value for each region in the parcellation.
index = 'IQ' # Questionnaire index {AM, ESS, PW}
# Building model
mse_reg, r2_reg = [], []
pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
for s in range(np.shape(pred_set)[0]):
    # Training/testing sets and target variable
    X_train, y_train = tril[pred_set[s][0]], questionnaire[index]
    X_test, y_test = tril[pred_set[s][1]], questionnaire[index]        
    # Create linear regression object
    reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization        
    # Train the model using the training sets
    reg.fit(X_train, y_train)        
    # Make predictions using the testing set
    y_pred_reg = reg.predict(X_test)       
    # The mean squared error
    mse_reg.append(mean_squared_error(y_test, y_pred_reg))        
    # The coefficient of determination: 1 is perfect prediction
    r2_reg.append(r2_score(y_test, y_pred_reg))


#%% NEW METHOD 4: Bootstrapping 

import numpy as np
import random
import time, timeit

#path_fine = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries_regional/mmp/'
path_fine = '/dcl01/smart/data/fvfarahani/searchlight/timeseries_regional/'

p1 = 2.00 # ratio 1
p2 = 1.00 # ratio 2

runs = ['rest1LR_', 'rest1RL_', 'rest2LR_', 'rest2RL_',
            'aligned_rest1LR_', 'aligned_rest1RL_', 'aligned_rest2LR_', 'aligned_rest2RL_']

n_run = len(runs)
n_sbj = 30
n_roi = 360
n_ts = 1200

#start_time = time.time()  

ts_sample = [np.zeros((n_sbj,n_ts,n_roi)) for i in range(n_run)] # 8 sets 

for roi in range(1,n_roi+1):       
    r = roi-1
    ts = [[] for i in range(n_run)] # 8 sets 
    
    # loading fine-scale timeseries within a region
    ts[0] = np.load(path_fine + 'ts_' + runs[0] + 'mmp_' + str(roi) + '.npy') 
    ts[1] = np.load(path_fine + 'ts_' + runs[1] + 'mmp_' + str(roi) + '.npy') 
    ts[2] = np.load(path_fine + 'ts_' + runs[2] + 'mmp_' + str(roi) + '.npy') 
    ts[3] = np.load(path_fine + 'ts_' + runs[3] + 'mmp_' + str(roi) + '.npy')
    ts[4] = np.load(path_fine + 'ts_' + runs[4] + 'mmp_' + str(roi) + '.npy') 
    ts[5] = np.load(path_fine + 'ts_' + runs[5] + 'mmp_' + str(roi) + '.npy') 
    ts[6] = np.load(path_fine + 'ts_' + runs[6] + 'mmp_' + str(roi) + '.npy') 
    ts[7] = np.load(path_fine + 'ts_' + runs[7] + 'mmp_' + str(roi) + '.npy') 
    
    # Bootstrapping, then averaging
    if random.uniform(0, 1) < p2:
        random_indices = np.random.choice(ts[0].shape[2], size=round(ts[0].shape[2]*p1), replace=True)    
        ts_sample[0][:,:,r] = ts[0][:, :, random_indices].mean(axis=2)
        ts_sample[1][:,:,r] = ts[1][:, :, random_indices].mean(axis=2)
        ts_sample[2][:,:,r] = ts[2][:, :, random_indices].mean(axis=2)
        ts_sample[3][:,:,r] = ts[3][:, :, random_indices].mean(axis=2)
        ts_sample[4][:,:,r] = ts[4][:, :, random_indices].mean(axis=2)
        ts_sample[5][:,:,r] = ts[5][:, :, random_indices].mean(axis=2)
        ts_sample[6][:,:,r] = ts[6][:, :, random_indices].mean(axis=2)
        ts_sample[7][:,:,r] = ts[7][:, :, random_indices].mean(axis=2)
    else:
        ts_sample[0][:,:,r] = ts[0].mean(axis=2)
        ts_sample[1][:,:,r] = ts[1].mean(axis=2)
        ts_sample[2][:,:,r] = ts[2].mean(axis=2)
        ts_sample[3][:,:,r] = ts[3].mean(axis=2)
        ts_sample[4][:,:,r] = ts[4].mean(axis=2)
        ts_sample[5][:,:,r] = ts[5].mean(axis=2)
        ts_sample[6][:,:,r] = ts[6].mean(axis=2)
        ts_sample[7][:,:,r] = ts[7].mean(axis=2)

#end = timeit.timeit()
#print("--- %s seconds ---" % (time.time() - start_time)) 

#np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_sample', ts_sample)
#ts_sample = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/ts_sample.npy')

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats.stats import pearsonr
from brainconn import similarity

fc = [[] for i in range(8)] # 8 sets 
tril = [[] for i in range(8)] # 8 sets 

#% CALCULATE CORRELATION (whole brain)
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional

fc[0] = correlation_measure.fit_transform(ts_sample[0])
fc[1] = correlation_measure.fit_transform(ts_sample[1])
fc[2] = correlation_measure.fit_transform(ts_sample[2])
fc[3] = correlation_measure.fit_transform(ts_sample[3])
fc[4] = correlation_measure.fit_transform(ts_sample[4])
fc[5] = correlation_measure.fit_transform(ts_sample[5])
fc[6] = correlation_measure.fit_transform(ts_sample[6])
fc[7] = correlation_measure.fit_transform(ts_sample[7])

for k in range(n_sbj): 
    # lower triangular
    tril[0].append(fc[0][k][np.tril_indices(fc[0].shape[2], k = -1)]) # K = -1 [without diagonal] or 0 [with]
    tril[1].append(fc[1][k][np.tril_indices(fc[0].shape[2], k = -1)])
    tril[2].append(fc[2][k][np.tril_indices(fc[0].shape[2], k = -1)])
    tril[3].append(fc[3][k][np.tril_indices(fc[0].shape[2], k = -1)])
    tril[4].append(fc[4][k][np.tril_indices(fc[0].shape[2], k = -1)])
    tril[5].append(fc[5][k][np.tril_indices(fc[0].shape[2], k = -1)])
    tril[6].append(fc[6][k][np.tril_indices(fc[0].shape[2], k = -1)])
    tril[7].append(fc[7][k][np.tril_indices(fc[0].shape[2], k = -1)])    

tril = np.array(tril)
"""    
#% SIMILARITY (i.e.,correlation coefficient between two flattened adjacency matrices) 
sim = np.zeros((n_sbj,n_sbj))
for k in range(n_sbj):
    for l in range(n_sbj):
        #sim[k,l] = similarity.corr_flat_und(adj_wei[0][k], adj_wei[1][l])
        sim[k,l] = np.corrcoef(tril[4][k], tril[5][l])[0, 1] # faster
# Get index for the highest value
index = sim.argsort()[:,-1]
# binarize
for k in range(n_sbj):
    sim[k,:] = [0 if i < sim[k,index[k]] else 1 for i in sim[k,:]]
# plot        
plt.imshow(sim)
plt.colorbar()"""

# REGRESSION ANALYSIS based on FC patterns (lower triangular)
# Loading questionnaires's indices
IQ = []
#file_in = open('/Volumes/Elements/Hyperalignment/HCP/IQ.txt', 'r')
file_in = open('/users/ffarahan/IQ.txt', 'r')
for z in file_in.read().split('\n'):
    IQ.append(float(z))

questionnaire = {'IQ': np.array(IQ)}
questionnaire_list = ['IQ']
# Set up a correlation vector (measure * Q_index) corresponding to a value for each region in the parcellation.
index = 'IQ' # Questionnaire index {AM, ESS, PW}
# Building model
mse_reg, r2_reg = [], []
pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
for s in range(np.shape(pred_set)[0]):
    # Training/testing sets and target variable
    X_train, y_train = tril[pred_set[s][0],:,:], questionnaire[index]
    X_test, y_test = tril[pred_set[s][1],:,:], questionnaire[index]        
    # Create linear regression object
    reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization        
    # Train the model using the training sets
    reg.fit(X_train, y_train)        
    # Make predictions using the testing set
    y_pred_reg = reg.predict(X_test)       
    # The mean squared error
    mse_reg.append(mean_squared_error(y_test, y_pred_reg))        
    # The coefficient of determination: 1 is perfect prediction
    r2_reg.append(r2_score(y_test, y_pred_reg))

print(r2_reg)

#%% FINE-SCALE TIMESERIES in each REGION/NETWROK

atlas = hcp.mmp # {‘mmp’, ‘ca_parcels’, ‘ca_network’, ‘yeo7’, ‘yeo17’}

extra = 19 # set to 19 for mmp, 358 for ca_parcels, and 0 otherwise

# rest1LR
for roi in range(1, len(atlas.nontrivial_ids)+1 - extra): # region/network of interet
    roi_idx = np.where(atlas.map_all[:59412] == roi)[0] # only cortex          
    ts_rest1LR = []; ts_aligned_rest1LR = []
    for k in range(len(subjects1[0:n_sbj])):
        ts = dss_rest1LR[k].samples[:, roi_idx]
        ts_aligned = dss_aligned_rest1LR[k].samples[:, roi_idx]
        ts_rest1LR.append(ts)
        ts_aligned_rest1LR.append(ts_aligned)
    # Save fine-grained timeseries
    print(roi)
    np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp_roi/ts_rest1LR_mmp_' + str(roi), ts_rest1LR)
    np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp_roi/ts_aligned_rest1LR_mmp_' + str(roi), ts_aligned_rest1LR)
    #np.save('/users/ffarahan/data/ts_rest1LR_yeo17_' + str(roi), ts_rest1LR)
    #np.save('/users/ffarahan/data/ts_aligned_rest1LR_yeo17_' + str(roi), ts_aligned_rest1LR)

# rest1RL
for roi in range(1, len(atlas.nontrivial_ids)+1 - extra): # region/network of interet
    roi_idx = np.where(atlas.map_all[:59412] == roi)[0] # only cortex  
    ts_rest1RL = []; ts_aligned_rest1RL = []
    for k in range(len(subjects1[0:n_sbj])):
        ts = dss_rest1RL[k].samples[:, roi_idx]
        ts_aligned = dss_aligned_rest1RL[k].samples[:, roi_idx]
        ts_rest1RL.append(ts)
        ts_aligned_rest1RL.append(ts_aligned)
    # Save fine-grained timeseries
    print(roi)
    np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp_roi/ts_rest1RL_mmp_' + str(roi), ts_rest1RL)
    np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp_roi/ts_aligned_rest1RL_mmp_' + str(roi), ts_aligned_rest1RL)
    #np.save('/users/ffarahan/data/ts_rest1RL_yeo17_' + str(roi), ts_rest1RL)    
    #np.save('/users/ffarahan/data/ts_aligned_rest1RL_yeo17_' + str(roi), ts_aligned_rest1RL)

# rest2LR
for roi in range(1, len(atlas.nontrivial_ids)+1 - extra): # region/network of interet
    roi_idx = np.where(atlas.map_all[:59412] == roi)[0] # only cortex  
    ts_rest2LR = []; ts_aligned_rest2LR = []
    for k in range(len(subjects1[0:n_sbj])):
        ts = dss_rest2LR[k].samples[:, roi_idx]
        ts_aligned = dss_aligned_rest2LR[k].samples[:, roi_idx]
        ts_rest2LR.append(ts)
        ts_aligned_rest2LR.append(ts_aligned)
    # Save fine-grained timeseries
    print(roi)
    np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp_roi/ts_rest2LR_mmp_' + str(roi), ts_rest2LR)
    np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp_roi/ts_aligned_rest2LR_mmp_' + str(roi), ts_aligned_rest2LR)
    #np.save('/users/ffarahan/data/ts_rest2LR_yeo17_' + str(roi), ts_rest2LR)
    #np.save('/users/ffarahan/data/ts_aligned_rest2LR_yeo17_' + str(roi), ts_aligned_rest2LR)

# rest2RL
for roi in range(1, len(atlas.nontrivial_ids)+1 - extra): # region/network of interet
    roi_idx = np.where(atlas.map_all[:59412] == roi)[0] # only cortex 
    ts_rest2RL = []; ts_aligned_rest2RL = []
    for k in range(len(subjects1[0:n_sbj])):
        ts = dss_rest2RL[k].samples[:, roi_idx]
        ts_aligned = dss_aligned_rest2RL[k].samples[:, roi_idx]
        ts_rest2RL.append(ts)
        ts_aligned_rest2RL.append(ts_aligned)
    # Save fine-grained timeseries
    print(roi)
    np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp_roi/ts_rest2RL_mmp_' + str(roi), ts_rest2RL)
    np.save('/dcl01/smart/data/fvfarahani/searchlight/ts_mmp_roi/ts_aligned_rest2RL_mmp_' + str(roi), ts_aligned_rest2RL)
    #np.save('/users/ffarahan/data/ts_rest2RL_yeo17_' + str(roi), ts_rest2RL)
    #np.save('/users/ffarahan/data/ts_aligned_rest2RL_yeo17_' + str(roi), ts_aligned_rest2RL)  

#%% Compute MODULARITY
from scipy import sparse
from sknetwork.clustering import Louvain, modularity

n_sbj = len(corr_rest2LR)
roi = corr_rest2LR.shape[1]
density, upper, lower = 0.05, 1, 0 # density-based thresholding
n = round(density*roi*(roi-1)/2)*2 # number of strongest connections * 2 (nodes)
q = np.zeros((n_sbj,2))

for k in range(n_sbj):
    corr_1 = corr_rest2LR[k]
    corr_2 = corr_aligned_rest2LR[k]
    # remove self connections
    np.fill_diagonal(corr_1, 0)
    np.fill_diagonal(corr_2, 0)
    # absolute value 
    corr_1 = np.absolute(corr_1)  
    corr_2 = np.absolute(corr_2)
    # binarizing the individual correlation matrix
    flat_1 = np.sort(corr_1.flatten()) # flatten and sort the matrix
    flat_2 = np.sort(corr_2.flatten())
    threshold_1 = flat_1[-n]
    threshold_2 = flat_2[-n]
    adjacency_1 = np.where(corr_1 > threshold_1, upper, lower) # adjacency = corr
    adjacency_2 = np.where(corr_2 > threshold_2, upper, lower)
    # represent the adjacency matrix in the Compressed Sparse Row format
    adjacency_1 = sparse.csr_matrix(adjacency_1) # biadjacency.shape
    adjacency_2 = sparse.csr_matrix(adjacency_2)
    # clustering by Louvain
    louvain_1 = Louvain() # default:resolution=1; modularity='dugue','newman'or'potts'
    louvain_2 = Louvain()
    labels_1 = louvain_1.fit_transform(adjacency_1)
    labels_2 = louvain_2.fit_transform(adjacency_2)
    labels_unique_1, counts_1 = np.unique(labels_1, return_counts=True) 
    labels_unique_2, counts_2 = np.unique(labels_2, return_counts=True)  
    # metrics
    q[k,0] = modularity(adjacency_1, labels_1)
    q[k,1] = modularity(adjacency_2, labels_2)

# Implementation of permutation test
def perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

perm = 10000
pval = perm_test(q[:,0], q[:,1], perm)

# boxplot
datasetA1 = q[:,0]
datasetA2 = q[:,1]

ticks = ['5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%']



# scatterplot
plt.figure(figsize=(6, 6))
plt.scatter(q[:,0], q[:,1])
plt.xlabel('MSM-All', size='x-large') # fontsize=20
plt.ylabel('Searchlight CHA', size='x-large')
#plt.title('Average pairwise correlation', size='xx-large')
plt.plot([.15, .6], [.15, .6], 'k--')
plt.show()


## create edge/node lists for visualization in Gephi
import pandas as pd
df = np.triu(adjacency.toarray())
# create edge list
path = '???'
edge_list = df.stack().reset_index()
edge_list.columns = ['Source','Target', 'Weight']
edge_list = edge_list[(edge_list[['Weight']] != 0).all(axis=1)]
edge_list.to_csv(path + 'edge_list.csv', sep=',', index=False) # sep='\t'
# create node list    
node_list = pd.DataFrame({'id': np.array(['{}'.format(x) for x in range(roi)]),
                   'label': np.array(['vertex_{}'.format(x) for x in range(roi)])})
node_list.to_csv(path + 'edge_list.csv', sep=',', index=False) # sep='\t'

#%% ICA analysis
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import pandas as pd
import time
import seaborn as sns

dss_rest1LR_l = []
dss_rest1LR_r = []

# load pre-computed mappers 
os.chdir('/dcl01/smart/data/fvfarahani/searchlight/mappers/') #!rm .DS_Store
mappers_l = h5load('mappers_30sbj_10r_L.midthickness.hdf5.gz') 
mappers_r = h5load('mappers_30sbj_10r_R.midthickness.hdf5.gz')   
#mappers_l = h5load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/mappers_30sbj_10r_L.midthickness.hdf5.gz')
#mappers_r = h5load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/mappers_30sbj_10r_R.midthickness.hdf5.gz')
n_sbj = len(mappers_l)    
    
# load timeseries
for subj in subjects1[0:n_sbj]:           
    img = nib.load(data_path + disk[0] + subj + '/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii'.format(subj=subj))
    rest1LR = img.get_fdata()
    rest1LR_n = hcp.normalize(rest1LR)
    ds_l = Dataset(rest1LR_n[:, hcp.struct.cortex_left])
    ds_r = Dataset(rest1LR_n[:, hcp.struct.cortex_right])
    ds_l.fa['node_indices'] = np.arange(ds_l.shape[1], dtype=int)
    ds_r.fa['node_indices'] = np.arange(ds_r.shape[1], dtype=int)
    zscore(ds_l, chunks_attr=None) # normalize features (vertices) to have unit variance (GLM parameters estimates for each voxel at this point).
    zscore(ds_r, chunks_attr=None)
    dss_rest1LR_l.append(ds_l)
    dss_rest1LR_r.append(ds_r)
    
# projecting data to common space
dss_aligned_rest1LR_l = [mapper.forward(ds) for ds, mapper in zip(dss_rest1LR_l, mappers_l)]
_ = [zscore(ds, chunks_attr=None) for ds in dss_aligned_rest1LR_l] 
dss_aligned_rest1LR_r = [mapper.forward(ds) for ds, mapper in zip(dss_rest1LR_r, mappers_r)]
_ = [zscore(ds, chunks_attr=None) for ds in dss_aligned_rest1LR_r] 

# extract PCs from each subject separately
dss_pca_rest1LR = []
dss_pca_aligned_rest1LR = []
n_components = 300
pca_a = PCA(n_components=n_components) #30
pca_b = PCA(n_components=n_components) #30
for k in range(n_sbj):        
    rest1LR_n = np.hstack((dss_rest1LR_l[k].samples, dss_rest1LR_r[k].samples))
    rest1LR_pca = pca_a.fit_transform(rest1LR_n.T).T
    dss_pca_rest1LR.append(rest1LR_pca)
    rest1LR_n_aligned = np.hstack((dss_aligned_rest1LR_l[k].samples, dss_aligned_rest1LR_r[k].samples))
    rest1LR_pca_aligned = pca_b.fit_transform(rest1LR_n_aligned.T).T
    dss_pca_aligned_rest1LR.append(rest1LR_pca_aligned)
    print(k)

print('Explained variation per principal component: {}'.format(np.sum(pca_a.explained_variance_ratio_)))
# Explained variation per principal component (200): 0.37393121462155504
# Explained variation per principal component (300): 0.4687384126121066
print('Explained variation per principal component: {}'.format(np.sum(pca_b.explained_variance_ratio_)))
# Explained variation per principal component (200): 0.6180059339425897
# Explained variation per principal component (300): 0.6915767332201846

pca_unaligned = np.array(dss_pca_rest1LR)
pca_unaligned = np.reshape(pca_unaligned, (pca_unaligned.shape[0]*pca_unaligned.shape[1], pca_unaligned.shape[2]))     
pca_aligned = np.array(dss_pca_aligned_rest1LR)
pca_aligned = np.reshape(pca_aligned, (pca_aligned.shape[0]*pca_aligned.shape[1], pca_aligned.shape[2])) 

os.chdir('/dcl01/smart/data/fvfarahani/searchlight/PCs/')
np.save('PC_unaligned_rest1LR_' + str(n_components), pca_unaligned) 
np.save('PC_aligned_rest1LR_' + str(n_components), pca_aligned) 

# Apply ICA
pca_unaligned = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/PCs/PC_unaligned_rest1LR_300.npy')
pca_aligned = np.load('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/PCs/PC_aligned_rest1LR_300.npy')
import copy
data_pca_a = copy.copy(pca_unaligned)
data_pca_b = copy.copy(pca_aligned)

n_components = 300 #5, 10
pca_a = PCA(n_components=n_components)
pca_b = PCA(n_components=n_components)
data_a = pca_a.fit_transform(data_pca_a.T).T
data_b = pca_b.fit_transform(data_pca_b.T).T
print('Explained variation per principal component: {}'.format(np.sum(pca_a.explained_variance_ratio_)))
print('Explained variation per principal component: {}'.format(np.sum(pca_b.explained_variance_ratio_)))

data = copy.copy(data_b)

n_components = 5 #5, 10
ica = FastICA(n_components=n_components, random_state=0) # transformer
components_masked = ica.fit_transform(data.T).T
# Normalize estimated components, for thresholding to make sense
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)
# Threshold
thld = 1
components_masked[np.abs(components_masked) < thld] = 0

# interactive 3D visualization in a web browser
for i in range(n_components):
    rois = hcp.cortical_components(np.abs(components_masked[i,:])>0, cutoff=300)[2] # 0, 1, 2 -> n_components, sizes, rois
    view = plotting.view_surf(hcp.mesh.inflated, hcp.cortex_data(hcp.mask(components_masked[i,:], rois!=0)), # vmax=0.9,
    # view = plotting.view_surf(hcp.mesh.inflated, hcp.cortex_data(components_masked[i,:]), # vmax=0.9,
        threshold=thld, bg_map=hcp.mesh.sulc) 
    view.open_in_browser()  

view = 'lateral' # {‘lateral’, ‘medial’, ‘dorsal’, ‘ventral’, ‘anterior’, ‘posterior’},
h = 'right' # which hemisphere to train HA? 'left' or 'right'   
for i in range(n_components):    
    rois = hcp.cortical_components(np.abs(components_masked[i,:])>0, cutoff=300)[2] # 0, 1, 2 -> n_components, sizes, rois
    plotting.plot_surf_stat_map(hcp.mesh.inflated, hcp.cortex_data(hcp.mask(components_masked[i,:], rois!=0)), 
    hemi=h, view=view, cmap='cold_hot', colorbar=True, # vmax=0.9,
    threshold=1, bg_map=hcp.mesh.sulc)

# T-Distributed Stochastic Neighbouring Entities (t-SNE)
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
atlas = hcp.yeo7 # yeo7, yeo17
feat_cols = ['t'+str(i) for i in range(data.shape[0])]
df = pd.DataFrame(data.T, columns=feat_cols)
df['y'] = atlas.map_all[:59412]
df['label'] = df['y'].apply(lambda i: str(i))
print('Size of the dataframe: {}'.format(df.shape))
data_tsne = df[feat_cols].values

# PCA analysis
n_components = 50
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(data_tsne)
print('Cumulative explained variation for ' + str(n_components) + ' principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))

# t-SNE on PCA-reduced data
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300, random_state=0)
tsne_pca_results = tsne.fit_transform(pca_result)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
# visualisation
df['tsne-pca50-one'] = tsne_pca_results[:,0]
df['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(16,10))
# Create an array with the colors you want to use (https://imagecolorpicker.com/en)
if atlas == hcp.yeo7:
    colors = ["#ffffff", "#a251ac", "#789ac0", "#409832", "#e065fe", "#f6fdc9", "#efb943", "#d9717d"]
elif atlas == hcp.yeo17:
    # more attention is needed for 17 networks
    #colors_yeo17 = ["#ffffff", "#781387", "#ff0101", "#4682b4", "#2acca3", "#0c30fe", "#4a9c3c", "#00760f", "#c53afa", "#ff97d5", "#e79422", "#873149", "#778cb0", "#fffe00", "#cd3f4e", "#000083", "#dcf8a5", "#7a8732"] # yeo color order
    colors = ["#ffffff", "#781387", "#ff0101", "#4682b4", "#2acca3", "#4a9c3c", "#00760f", "#c53afa", "#ff97d5", "#dcf8a5", "#7a8732", "#778cb0", "#e79422", "#873149", "#0c30fe", "#000083", "#fffe00", "#cd3f4e"] # hcp_utils color order
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette(colors), # sns.color_palette("hls", 7)
    data=df,
    legend="full",
    alpha=0.3,
)    

#%% Benchmark inter-subject correlations (180 regions, not voxels)

dss_test1 = []
dss_aligned1 = []

for k in range(len(subjects1)):
    ds = Dataset(list(ts_l_unaligned_mmp_test1)[k])
    ds.fa['node_indices'] = np.arange(ds.shape[1], dtype=int)
    zscore(ds, chunks_attr=None) # normalize features (parcells) to have unit variance.
    dss_test1.append(ds) 

for k in range(len(subjects1)):
    ds = Dataset(list(ts_l_aligned_mmp_test1)[k])
    ds.fa['node_indices'] = np.arange(ds.shape[1], dtype=int)
    zscore(ds, chunks_attr=None) # normalize features (parcells) to have unit variance.
    dss_aligned1.append(ds) 
    
def compute_average_similarity(dss, metric='correlation'):
    """
    Returns
    =======
    sim : ndarray
        A 1-D array with n_features elements, each element is the average
        pairwise correlation similarity on the corresponding feature.
    """
    n_features = dss[0].shape[1]
    sim = np.zeros((n_features, ))
    for i in range(n_features):
        data = np.array([ds.samples[:, i] for ds in dss])
        dist = pdist(data, metric)
        sim[i] = 1 - dist.mean()
    return sim

#sim_train = compute_average_similarity(dss_train)
sim_test1 = compute_average_similarity(dss_test1)
#sim_test2 = compute_average_similarity(dss_test2)
#sim_test3 = compute_average_similarity(dss_test3)

#sim_aligned0 = compute_average_similarity(dss_aligned0)
sim_aligned1 = compute_average_similarity(dss_aligned1)
#sim_aligned2 = compute_average_similarity(dss_aligned2)
#sim_aligned3 = compute_average_similarity(dss_aligned3)

plt.figure(figsize=(6, 6))
plt.scatter(sim_test1, sim_aligned1)
plt.xlim([-0.05, .2]) # main: [-.2, .5]
plt.ylim([-0.05, .2]) # main: [-.2, .5]
plt.xlabel('Surface alignment', size='xx-large')
plt.ylabel('SL Hyperalignment', size='xx-large')
#plt.title('Average pairwise correlation', size='xx-large')
plt.plot([-1, 1], [-1, 1], 'k--')
plt.show()   
    
#%% Plotting the lower triangular matrix for one subject
import pandas as pd 
import seaborn as sns

# E.g., first subject [0] in each set
#df_train = pd.DataFrame(corr_train[0]) 
df_test1 = pd.DataFrame(corr_test1[5]) 

def get_lower_tri_heatmap(df, output="triangular_matrix.png"):
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = False

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(33, 27))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(df, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=0.01, cbar_kws={"shrink": .5})
    # save to file
    fig = sns_plot.get_figure()
    fig.savefig(output)
    
#get_lower_tri_heatmap(df_train, output="triangular_train.png")  
get_lower_tri_heatmap(df_test1, output="triangular_test1.png")  
     

#%% Graph analysis

# install "brainconn" from their github repository
# !pip install git+https://github.com/FIU-Neuro/brainconn#egg=brainconn

# necessary imports
import os # os.path as op
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import brainconn
import networkx as nx
from nilearn.connectome import ConnectivityMeasure
from brainconn import degree, centrality, clustering, core, distance, modularity, utils
from netneurotools import stats as nnstats
import nilearn.plotting as plotting
import hcp_utils as hcp
import scipy.io as sio
from scipy.io import savemat

# calculating correlations
correlation_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional
os.chdir('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/ts_mmp_roi/') # /mmp/ or /yeo17/
#os.chdir('/dcl01/smart/data/fvfarahani/searchlight/ts_yeo17/')
#os.chdir('/users/ffarahan/')
atls = 'mmp' # mmp or yeo17 or ca_network
num_roi = 360 # 360, 17, 12
n_sbj = 30



# ======= save corr/bin matrices for modularity analysis (regions) ======================================================================
# path = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/corr_regional/' + atls + '/'
# runs = ['rest1LR', 'rest1RL', 'rest2LR', 'rest2RL', 'aligned_rest1LR', 'aligned_rest1RL', 'aligned_rest2LR', 'aligned_rest2RL'] 
# 
# for run in runs:
#     for roi in range(1,num_roi+1):  # region/network of interet
#         r = roi-1; # must start from zero
#         ts = np.load('ts_' + run + '_' + atls + '_' + str(roi) + '.npy') 
#         corr = correlation_measure.fit_transform(ts)
#         
#         bin_mat = np.zeros(np.shape(corr))
#         for k in range(30):
#             bin_mat[k] = utils.binarize(utils.threshold_proportional(corr[k], thld, copy=True))
#             #print(k)
#         # bin_mat = np.float32(bin_mat) # decress the size (int type does not work with GenLouvain)
#             
#         corr_path = path + atls + '_' + str(roi) + '/'
#         # Check whether the specified path exists or not
#         isExist = os.path.exists(corr_path)        
#         if not isExist:
#             # Create a new directory because it does not exist 
#             os.mkdir(corr_path)
#                 
#         savemat(corr_path + 'bin_' + run + '.mat', {'bin_' + run: bin_mat}, do_compression=False) # do_compression=True -> convert float64 to int
# 
#         print('ROI: {}'.format(roi))
#     
#     print('Run: {}'.format(run))
# =============================================================================


# ======= save corr/bin matrices for modularity analysis (networks) ======================================================================
# path = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/corr_regional/' + atls + '/'
# runs = ['rest1LR', 'rest1RL', 'rest2LR', 'rest2RL', 'aligned_rest1LR', 'aligned_rest1RL', 'aligned_rest2LR', 'aligned_rest2RL'] 
# 
# for run in runs:
#     for roi in range(1,num_roi+1):  # region/network of interet
#         r = roi-1; # must start from zero
#         ts = np.load('ts_' + run + '_' + atls + '_' + str(roi) + '.npy') 
#         corr = correlation_measure.fit_transform(ts)
#         
#         bin_mat = np.zeros(np.shape(corr))
#         for k in range(30):
#             bin_mat[k] = utils.binarize(utils.threshold_proportional(corr[k], thld, copy=True))
#             print(k)
#         bin_mat = np.float32(bin_mat) # decress the size (int type does not work with GenLouvain)
#         # calculate # of vertices per network (only cortex)
#         vtx_num = np.where(hcp.yeo17.map_all[:59412] == roi)[0] 
#         
#         # split the entire matrices to 4 pieces (error is Matrix too large to save with matlab 5 format)        
#         if len(vtx_num) % 2 == 0: # if number of vetices is Even 
#             upper_half = np.dsplit(np.hsplit(bin_mat, 2)[0], 2)
#             lower_half = np.dsplit(np.hsplit(bin_mat, 2)[1], 2)   
#             bin_mat_ul = upper_half[0] # upper left
#             bin_mat_ur = upper_half[1] # upper right
#             bin_mat_ll = lower_half[0] # lower left
#             bin_mat_lr = lower_half[1] # lowre right
#             
#             corr_path = path + atls + '_' + str(roi) + '/'
#             # Check whether the specified path exists or not
#             isExist = os.path.exists(corr_path)        
#             if not isExist:
#                 # Create a new directory because it does not exist 
#                 os.mkdir(corr_path)
#                 
#             savemat(corr_path + 'bin_' + run + '_ul.mat', {'bin_' + run + '_ul': bin_mat_ul}, do_compression=True) # do_compression=True -> convert float64 to int
#             savemat(corr_path + 'bin_' + run + '_ur.mat', {'bin_' + run + '_ur': bin_mat_ur}, do_compression=True) 
#             savemat(corr_path + 'bin_' + run + '_ll.mat', {'bin_' + run + '_ll': bin_mat_ll}, do_compression=True) 
#             savemat(corr_path + 'bin_' + run + '_lr.mat', {'bin_' + run + '_lr': bin_mat_lr}, do_compression=True) 
#             
#         else: # if number of vetices is Odd 
#             h = round(len(vtx_num)/2)
#             bin_mat_ul = bin_mat[:,:h,:h] # upper left
#             bin_mat_ur = bin_mat[:,:h,h:] # upper right
#             bin_mat_ll = bin_mat[:,h:,:h] # lower left
#             bin_mat_lr = bin_mat[:,h:,h:] # lowre right
#             
#             corr_path = path + atls + '_' + str(roi) + '/'
#             # Check whether the specified path exists or not
#             isExist = os.path.exists(corr_path)        
#             if not isExist:
#                 # Create a new directory because it does not exist 
#                 os.mkdir(corr_path)
#                 
#             savemat(corr_path + 'bin_' + run + '_ul.mat', {'bin_' + run + '_ul': bin_mat_ul}, do_compression=True) # do_compression=True -> convert float64 to int
#             savemat(corr_path + 'bin_' + run + '_ur.mat', {'bin_' + run + '_ur': bin_mat_ur}, do_compression=True) 
#             savemat(corr_path + 'bin_' + run + '_ll.mat', {'bin_' + run + '_ll': bin_mat_ll}, do_compression=True) 
#             savemat(corr_path + 'bin_' + run + '_lr.mat', {'bin_' + run + '_lr': bin_mat_lr}, do_compression=True) 
#         
#         print('ROI: {}'.format(roi))
#     
#     print('Run: {}'.format(run))
# =============================================================================

num_loc_measure = 8
num_glb_measure = 6
num_measure = num_loc_measure + num_glb_measure

# global measures + std
lam = [[[] for i in range(8)] for i in range(num_roi)] # lambda (characteristic path length)
eff = [[[] for i in range(8)] for i in range(num_roi)] # global efficieny
clc = [[[] for i in range(8)] for i in range(num_roi)] # global clustering coefficients
tra = [[[] for i in range(8)] for i in range(num_roi)] # Transitivity
ass = [[[] for i in range(8)] for i in range(num_roi)] # assortativity
mod = [[[] for i in range(8)] for i in range(num_roi)] # modularity

std_lam = [[] for i in range(8)]
std_eff = [[] for i in range(8)]
std_clc = [[] for i in range(8)]
std_tra = [[] for i in range(8)]
std_ass = [[] for i in range(8)]
std_mod = [[] for i in range(8)]

# local std
std_deg_l = [[] for i in range(8)] # 8 sets
std_stg_l = [[] for i in range(8)]
std_eig_l = [[] for i in range(8)]
std_clc_l = [[] for i in range(8)]
std_eff_l = [[] for i in range(8)]
std_par_l = [[] for i in range(8)]
std_zsc_l = [[] for i in range(8)]
#std_rch_l = [[] for i in range(8)]
std_kco_l = [[] for i in range(8)]

# Loading questionnaires's indices (for correlation analysis)
IQ = []
file_in = open('/Volumes/Elements/Hyperalignment/HCP/IQ.txt', 'r')
#file_in = open('/users/ffarahan/IQ.txt', 'r') # temporal
for z in file_in.read().split('\n'):
    IQ.append(float(z))
#questionnaire = {'AM': np.array(am), 'ESS': np.array(ess), 'PW': np.array(pw)}
questionnaire = {'IQ': np.array(IQ)}
questionnaire_list = ['IQ']
# Set up a correlation vector (measure * Q_index) corresponding to a value for each region in the parcellation.
index = 'IQ' # Questionnaire index {AM, ESS, PW}
n_perm = 500

n_components = 30
mse_reg, r2_reg = [], []
mse_pcr, r2_pcr = [], []
mse_pls, r2_pls = [], []
for i in range(num_loc_measure):
    mse_reg.append(np.zeros([num_roi, 6])) # 6 distinct pred_sets
    r2_reg.append(np.zeros([num_roi, 6]))
    mse_pcr.append(np.zeros([num_roi, 6]))
    r2_pcr.append(np.zeros([num_roi, 6]))
    mse_pls.append(np.zeros([num_roi, 6]))
    r2_pls.append(np.zeros([num_roi, 6]))

ci = sio.loadmat('/Volumes/Elements/Modularity/var_mmp/S_a_1.0,-1.0.mat', squeeze_me=True)['S_a']
#%%
for roi in range(1,num_roi+1): # example, 149 PFm # range(1,num_roi+1)
#for roi in range(1,18): # region/network of interet
    
    r = roi-1; # must start from zero
    
    ts_rest1LR = np.load('ts_rest1LR_' + atls + '_' + str(roi) + '.npy') # np.load('/dcl01/smart/data/fvfarahani/searchlight/timeseries_regional/ts_rest1LR_yeo17_' + str(roi) + '.npy')
    ts_rest1RL = np.load('ts_rest1RL_' + atls + '_' + str(roi) + '.npy') # np.load('/dcl01/smart/data/fvfarahani/searchlight/timeseries_regional/ts_rest1RL_yeo17_' + str(roi) + '.npy')
    ts_rest2LR = np.load('ts_rest2LR_' + atls + '_' + str(roi) + '.npy') # np.load('/users/ffarahan/data/ts_rest2LR_yeo17_' + str(roi) + '.npy')
    ts_rest2RL = np.load('ts_rest2RL_' + atls + '_' + str(roi) + '.npy') # np.load('/users/ffarahan/data/ts_rest2RL_yeo17_' + str(roi) + '.npy')
    ts_aligned_rest1LR = np.load('ts_aligned_rest1LR_' + atls + '_' + str(roi) + '.npy') # np.load('/dcl01/smart/data/fvfarahani/searchlight/timeseries_regional/ts_aligned_rest1LR_yeo17_' + str(roi) + '.npy')
    ts_aligned_rest1RL = np.load('ts_aligned_rest1RL_' + atls + '_' + str(roi) + '.npy') # np.load('/dcl01/smart/data/fvfarahani/searchlight/timeseries_regional/ts_aligned_rest1RL_yeo17_' + str(roi) + '.npy')
    ts_aligned_rest2LR = np.load('ts_aligned_rest2LR_' + atls + '_' + str(roi) + '.npy') # np.load('/users/ffarahan/data/ts_aligned_rest2LR_yeo17_' + str(roi) + '.npy')
    ts_aligned_rest2RL = np.load('ts_aligned_rest2RL_' + atls + '_' + str(roi) + '.npy') # np.load('/users/ffarahan/data/ts_aligned_rest2RL_yeo17_' + str(roi) + '.npy')
    
    # if concatenation is neeeded
    #ts_REST1 = np.hstack((ts_rest1LR, ts_rest1RL))
    #ts_REST2 = np.hstack((ts_rest2LR, ts_rest2RL))
    #ts_aligned_REST1 = np.hstack((ts_aligned_rest1LR, ts_aligned_rest1RL))
    #ts_aligned_REST2 = np.hstack((ts_aligned_rest2LR, ts_aligned_rest2RL))
    
    corr_rest1LR = correlation_measure.fit_transform(ts_rest1LR)
    corr_rest1RL = correlation_measure.fit_transform(ts_rest1RL)
    corr_rest2LR = correlation_measure.fit_transform(ts_rest2LR)
    corr_rest2RL = correlation_measure.fit_transform(ts_rest2RL)
    corr_aligned_rest1LR = correlation_measure.fit_transform(ts_aligned_rest1LR)
    corr_aligned_rest1RL = correlation_measure.fit_transform(ts_aligned_rest1RL)
    corr_aligned_rest2LR = correlation_measure.fit_transform(ts_aligned_rest2LR)
    corr_aligned_rest2RL = correlation_measure.fit_transform(ts_aligned_rest2RL)    
    #corr_REST1 = correlation_measure.fit_transform(ts_REST1)
    #corr_REST2 = correlation_measure.fit_transform(ts_REST2)
    #corr_aligned_REST1 = correlation_measure.fit_transform(ts_aligned_REST1)
    #corr_aligned_REST2 = correlation_measure.fit_transform(ts_aligned_REST2)
    
    # make directory and save as mat file
    """
    #path = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/corr_regional/' + atls + '/'
    path = '/dcl01/smart/data/fvfarahani/searchlight/corr_regional/' + atls + '/'
    os.mkdir(path + atls + '_{}'.format(roi))
    
    # Save arrays into a MATLAB-style .mat file
    from scipy.io import savemat 
    corr_path = path + atls + '_' + str(roi) + '/'
    
    savemat(corr_path + 'corr_rest1LR.mat', {'corr_rest1LR': corr_rest1LR})
    savemat(corr_path + 'corr_rest1RL.mat', {'corr_rest1RL': corr_rest1RL})
    savemat(corr_path + 'corr_rest2LR.mat', {'corr_rest2LR': corr_rest2LR})
    savemat(corr_path + 'corr_rest2RL.mat', {'corr_rest2RL': corr_rest2RL})
    savemat(corr_path + 'corr_aligned_rest1LR.mat', {'corr_aligned_rest1LR': corr_aligned_rest1LR})
    savemat(corr_path + 'corr_aligned_rest1RL.mat', {'corr_aligned_rest1RL': corr_aligned_rest1RL})
    savemat(corr_path + 'corr_aligned_rest2LR.mat', {'corr_aligned_rest2LR': corr_aligned_rest2LR})
    savemat(corr_path + 'corr_aligned_rest2RL.mat', {'corr_aligned_rest2RL': corr_aligned_rest2RL})
    #savemat(corr_path + 'corr_REST1.mat', {'corr_REST1': corr_REST1})
    #savemat(corr_path + 'corr_REST2.mat', {'corr_REST2': corr_REST2})   
    #savemat(corr_path + 'corr_aligned_REST1.mat', {'corr_aligned_REST1': corr_aligned_REST1})
    #savemat(corr_path + 'corr_aligned_REST2.mat', {'corr_aligned_REST2': corr_aligned_REST2})
    """
    
    print(roi)
    
    """ # if needed: writing correlation matrices as txt files
    corr_path = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/corr_regional/Yeo17_11/'
    n_sbj = 30
    for k in range(n_sbj):  
        np.savetxt(corr_path + '/unaligned/rest1LR/' + 'sbj' + str(k+1) + '.txt', corr_rest1LR[k], fmt='%.18e', delimiter=' ', newline='\n')
        np.savetxt(corr_path + '/unaligned/rest1RL/' + 'sbj' + str(k+1) + '.txt', corr_rest1RL[k], fmt='%.18e', delimiter=' ', newline='\n')
        np.savetxt(corr_path + '/unaligned/rest2LR/' + 'sbj' + str(k+1) + '.txt', corr_rest2LR[k], fmt='%.18e', delimiter=' ', newline='\n')
        np.savetxt(corr_path + '/unaligned/rest2RL/' + 'sbj' + str(k+1) + '.txt', corr_rest2RL[k], fmt='%.18e', delimiter=' ', newline='\n')
        np.savetxt(corr_path + '/aligned/rest1LR/' + 'sbj' + str(k+1) + '.txt', corr_aligned_rest1LR[k], fmt='%.18e', delimiter=' ', newline='\n')
        np.savetxt(corr_path + '/aligned/rest1RL/' + 'sbj' + str(k+1) + '.txt', corr_aligned_rest1RL[k], fmt='%.18e', delimiter=' ', newline='\n')
        np.savetxt(corr_path + '/aligned/rest2LR/' + 'sbj' + str(k+1) + '.txt', corr_aligned_rest2LR[k], fmt='%.18e', delimiter=' ', newline='\n')
        np.savetxt(corr_path + '/aligned/rest2RL/' + 'sbj' + str(k+1) + '.txt', corr_aligned_rest2RL[k], fmt='%.18e', delimiter=' ', newline='\n')       
        #np.savetxt(corr_path + '/unaligned/REST1/' + 'sbj' + str(k+1) + '.txt', corr_REST1[k], fmt='%.18e', delimiter=' ', newline='\n')
        #np.savetxt(corr_path + '/unaligned/REST2/' + 'sbj' + str(k+1) + '.txt', corr_REST2[k], fmt='%.18e', delimiter=' ', newline='\n')
        #np.savetxt(corr_path + '/aligned/REST1/' + 'sbj' + str(k+1) + '.txt', corr_aligned_REST1[k], fmt='%.18e', delimiter=' ', newline='\n')
        #np.savetxt(corr_path + '/aligned/REST2/' + 'sbj' + str(k+1) + '.txt', corr_aligned_REST2[k], fmt='%.18e', delimiter=' ', newline='\n')
        print(k)    
    """
    
    # remove the self-connections (zero diagonal) and create weighted graphs
    adj_wei = [[] for i in range(8)] # 8 sets (list of lists; wrong way -> adj_wei = [[]] * 8)
    adj_bin = [[] for i in range(8)]
    con_len = [[] for i in range(8)] # weighted connection-length matrix for 8 sets
    thld = 0.3
    for k in range(n_sbj): 
        np.fill_diagonal(corr_rest1LR[k], 0)
        np.fill_diagonal(corr_rest1RL[k], 0)
        np.fill_diagonal(corr_rest2LR[k], 0)
        np.fill_diagonal(corr_rest2RL[k], 0)
        np.fill_diagonal(corr_aligned_rest1LR[k], 0)
        np.fill_diagonal(corr_aligned_rest1RL[k], 0)
        np.fill_diagonal(corr_aligned_rest2LR[k], 0)
        np.fill_diagonal(corr_aligned_rest2RL[k], 0)
        # weighted
        adj_wei[0].append(corr_rest1LR[k])
        adj_wei[1].append(corr_rest1RL[k])
        adj_wei[2].append(corr_rest2LR[k])
        adj_wei[3].append(corr_rest2RL[k])
        adj_wei[4].append(corr_aligned_rest1LR[k])
        adj_wei[5].append(corr_aligned_rest1RL[k])
        adj_wei[6].append(corr_aligned_rest2LR[k])
        adj_wei[7].append(corr_aligned_rest2RL[k])
        # weighted connection-length matrix (connection lengths is needed prior to computation of weighted distance-based measures, such as distance and betweenness centrality)
        # L_ij = 1/W_ij for all nonzero L_ij; higher connection weights intuitively correspond to shorter lengths
        con_len[0].append(utils.weight_conversion(adj_wei[0][k], 'lengths', copy=True))
        con_len[1].append(utils.weight_conversion(adj_wei[1][k], 'lengths', copy=True))
        con_len[2].append(utils.weight_conversion(adj_wei[2][k], 'lengths', copy=True))
        con_len[3].append(utils.weight_conversion(adj_wei[3][k], 'lengths', copy=True))
        con_len[4].append(utils.weight_conversion(adj_wei[4][k], 'lengths', copy=True))
        con_len[5].append(utils.weight_conversion(adj_wei[5][k], 'lengths', copy=True))
        con_len[6].append(utils.weight_conversion(adj_wei[6][k], 'lengths', copy=True))
        con_len[7].append(utils.weight_conversion(adj_wei[7][k], 'lengths', copy=True))
        # binary
        adj_bin[0].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[0][k], thld, copy=True)))
        adj_bin[1].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[1][k], thld, copy=True)))
        adj_bin[2].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[2][k], thld, copy=True)))
        adj_bin[3].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[3][k], thld, copy=True)))
        adj_bin[4].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[4][k], thld, copy=True)))
        adj_bin[5].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[5][k], thld, copy=True)))
        adj_bin[6].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[6][k], thld, copy=True)))
        adj_bin[7].append(brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[7][k], thld, copy=True)))    
       
    # plot weighted/binary adjacency matrix [0,1,2,3,4,5,6,7]
    #fig, ax = plt.subplots(figsize=(7, 7))
    #x.imshow(adj_bin[0][2])
    #fig.show()    
       
    # define local measures    
    deg_l = [[] for i in range(8)] # 8 sets
    stg_l = [[] for i in range(8)]
    eig_l = [[] for i in range(8)]
    clc_l = [[] for i in range(8)]
    eff_l = [[] for i in range(8)]
    par_l = [[] for i in range(8)]
    zsc_l = [[] for i in range(8)]
    #rch_l = [[] for i in range(8)]
    kco_l = [[] for i in range(8)]
    
    
    # compute global measures
    for i in range(8):
        for k in range(n_sbj): 
            dis = distance.distance_wei(con_len[i][k])[0] # TIME CONSUMING  
            lam[r][i].append(distance.charpath(dis, include_diagonal=False, include_infinite=True)[0])
            eff[r][i].append(distance.charpath(dis, include_diagonal=False, include_infinite=True)[1])
            clc[r][i].append(np.mean(clustering.clustering_coef_wu(adj_wei[i][k])))
            tra[r][i].append(clustering.transitivity_wu(adj_wei[i][k]))
            ass[r][i].append(core.assortativity_wei(adj_wei[i][k], flag=0)) # 0: undirected graph
            mod[r][i].append(modularity.modularity_louvain_und(adj_wei[i][k], gamma=1, hierarchy=False, seed=None)[1])                
                                   
    # compute local measures
    for i in range(8):
        for k in range(n_sbj): 
            deg_l[i].append(degree.degrees_und(adj_bin[i][k]))
            stg_l[i].append(degree.strengths_und(adj_wei[i][k]))
            eig_l[i].append(centrality.eigenvector_centrality_und(adj_bin[i][k]))
            clc_l[i].append(clustering.clustering_coef_bu(adj_bin[i][k]))
            eff_l[i].append(distance.efficiency_bin(adj_bin[i][k], local=True)) #eff_l[i].append(distance.efficiency_wei(adj_wei[i][k], local=True)) # TIME CONSUMING             
            par_l[i].append(centrality.participation_coef(adj_bin[i][k], ci[r][i,k,:], degree='undirected'))
            zsc_l[i].append(centrality.module_degree_zscore(adj_bin[i][k], ci[r][i,k,:], flag=0)) # 0: undirected graph
            #rch_l[i].append(core.rich_club_bu(adj_bin[i][k])[0])
            kco_l[i].append(centrality.kcoreness_centrality_bu(adj_bin[i][k])[0])
            #print('Subject: ', k)
        #print('Set: ', i)            
            
    deg_l = np.array(deg_l) 
    stg_l = np.array(stg_l) 
    eig_l = np.array(eig_l) 
    clc_l = np.array(clc_l)
    eff_l = np.array(eff_l)
    par_l = np.array(par_l) 
    zsc_l = np.array(zsc_l) 
    #rch_l = np.array(rch_l) 
    kco_l = np.array(kco_l) 
     
    #from networkx.algorithms.smallworld import sigma
    #G = nx.from_numpy_matrix(adj_wei[0][1])     
    #sigma(G, niter=10, nrand=5, seed=None)
       
    # standard deviations of all nodes(vertices)/brains
    for i in range(8):
        # local measures        
        std_deg_l[i].append(np.std(deg_l[i], axis=0))
        std_stg_l[i].append(np.std(stg_l[i], axis=0))
        std_eig_l[i].append(np.std(eig_l[i], axis=0))
        std_clc_l[i].append(np.std(clc_l[i], axis=0))
        std_eff_l[i].append(np.std(eff_l[i], axis=0))
        std_par_l[i].append(np.std(par_l[i], axis=0))
        std_zsc_l[i].append(np.std(zsc_l[i], axis=0))
        #std_rch_l[i].append(np.std(rch_l[i], axis=0))
        std_kco_l[i].append(np.std(kco_l[i], axis=0))
        # global measures
        std_lam[i].append(np.std(lam[r][i], axis=0))
        std_eff[i].append(np.std(eff[r][i], axis=0))
        std_clc[i].append(np.std(clc[r][i], axis=0))
        std_tra[i].append(np.std(tra[r][i], axis=0))
        std_ass[i].append(np.std(ass[r][i], axis=0))
        std_mod[i].append(np.std(mod[r][i], axis=0))
        
    # Regression models
    # https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html#sphx-glr-auto-examples-cross-decomposition-plot-pcr-vs-pls-py
    from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    local_measure = [deg_l, stg_l, eig_l, clc_l, eff_l, par_l, zsc_l, kco_l] #eloc_w, rch_l, 
    pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
    for i in range(len(local_measure)):
        for s in range(np.shape(pred_set)[0]):
            # Training/testing sets and target variable
            X_train, y_train = local_measure[i][pred_set[s][0]], questionnaire[index]
            X_test, y_test = local_measure[i][pred_set[s][1]], questionnaire[index]
            
            # Create linear regression object
            #reg = LinearRegression() # Ordinary least squares Linear Regression
            reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization
            #reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            #reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]) # Ridge regression with built-in cross-validation (Leave-One-Out)
            #pcr = make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression()) # n_components must be between 0 and min(n_samples, n_features)
            #pls = PLSRegression(n_components=n_components)
            
            # Train the model using the training sets
            reg.fit(X_train, y_train)
            #pcr.fit(X_train, y_train)
            #pls.fit(X_train, y_train)
            
            # Make predictions using the testing set
            y_pred_reg = reg.predict(X_test)
            #y_pred_pcr = pcr.predict(X_test)
            #y_pred_pls = pls.predict(X_test)
            
            # The mean squared error
            mse_reg[i][r,s] = mean_squared_error(y_test, y_pred_reg)
            #mse_pcr[i][r,s] = mean_squared_error(y_test, y_pred_pcr)
            #mse_pls[i][r,s] = mean_squared_error(y_test, y_pred_pls)
            
            # The coefficient of determination: 1 is perfect prediction
            r2_reg[i][r,s] = r2_score(y_test, y_pred_reg)
            #r2_pcr[i][r,s] = r2_score(y_test, y_pred_pcr)
            #r2_pls[i][r,s] = r2_score(y_test, y_pred_pls)
        
    print('ROI: ', roi)

#%%
np.save('/Users/Farzad/Desktop/Figures/mse_glob_reg', mse_glob_reg)
np.save('/Users/Farzad/Desktop/Figures/mse_reg', mse_reg)

mse_glob_reg = np.load('/Users/Farzad/Desktop/Figures/mse_glob_reg.npy')
mse_reg = np.load('/Users/Farzad/Desktop/Figures/mse_reg.npy')

lam = np.array(lam)
eff = np.array(eff)
clc = np.array(clc)
tra = np.array(tra)
ass = np.array(ass)
mod = np.array(mod)

std_deg_l = [np.hstack(std_deg_l[i]).squeeze() for i in range(8)]
std_stg_l = [np.hstack(std_stg_l[i]).squeeze() for i in range(8)]
std_eig_l = [np.hstack(std_eig_l[i]).squeeze() for i in range(8)]
std_clc_l = [np.hstack(std_clc_l[i]).squeeze() for i in range(8)]
std_eff_l = [np.hstack(std_eff_l[i]).squeeze() for i in range(8)]
std_par_l = [np.hstack(std_par_l[i]).squeeze() for i in range(8)]
std_zsc_l = [np.hstack(std_zsc_l[i]).squeeze() for i in range(8)]
#std_rch_l = [np.hstack(std_rch_l[i]).squeeze() for i in range(8)]
std_kco_l = [np.hstack(std_kco_l[i]).squeeze() for i in range(8)]
std_lam = [np.hstack(std_lam[i]).squeeze() for i in range(8)]
std_eff = [np.hstack(std_eff[i]).squeeze() for i in range(8)]
std_clc = [np.hstack(std_clc[i]).squeeze() for i in range(8)]
std_tra = [np.hstack(std_tra[i]).squeeze() for i in range(8)]
std_ass = [np.hstack(std_ass[i]).squeeze() for i in range(8)]
std_mod = [np.hstack(std_mod[i]).squeeze() for i in range(8)]

var = [std_deg_l, std_stg_l, std_eig_l, std_clc_l, std_eff_l, std_par_l, std_zsc_l, std_kco_l] # std_rch_l, 
       #std_lam, std_eff, std_clc, std_tra, std_ass, std_mod]
var_name = ['Degree Centrality', 'Strengths', 'Eigenvector Centrality', 'Clustering Coefficient', 'Local Efficiency', 'Participation Coefficient', 'Within-module Degree Z-score', 'K-coreness'] # 'Rich Club Coefficient', 
           # 'Path Length', 'Global Efficiency', 'Mean Clustering Coefficient', 'Transitivity', 'Assortativity', 'Modularity'] 
run = 2 # rset1LR -> 1; rest1RL -> 2; rest2LR -> 3; rest2RL -> 4

# plot std
for i, name in enumerate(var_name):
    # Min-max normalization
    x = (var[i][run-1]-var[i][run-1].min())/(var[i][run-1].max()-var[i][run-1].min()) # MSM-All
    y = (var[i][run+3]-var[i][run+3].min())/(var[i][run+3].max()-var[i][run+3].min()) # CHA
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.xlabel('MSM-All', size='x-large') # fontsize=20
    plt.ylabel('Searchlight CHA', size='x-large')
    plt.title(var_name[i], size='xx-large')
    plt.plot([0, 1], [0, 1], 'k--')
    #plt.plot(np.mean(x), np.mean(y), "or") # plot average point
    plt.show()

# scatter plot on MSE
i = 1 # measure: 0, 1, 2, 3, 4
s = 1 # prediction sets: 1, 2, 3
x = mse_reg[i][:,s-1] # MSM-All
y = mse_reg[i][:,s+2] # CHA

plt.figure(figsize=(6, 6))
plt.scatter(x, y)
plt.xlabel('MSM-All', size='x-large') # fontsize=20
plt.ylabel('Searchlight CHA', size='x-large')
plt.title('Mean squared error (MSE)', size='xx-large')
limit_range = [min(min(x),min(y)), max(max(x),max(y))]
plt.plot(limit_range, limit_range, 'k--')
#plt.axis('square')
plt.show()

# correlation between global measures and fluid intelligence
corr1 = []
corr2 = []
for r in range(num_roi):
    corr1.append(nnstats.permtest_pearsonr(ass[r,3,:], questionnaire[index], n_perm=n_perm))
    corr2.append(nnstats.permtest_pearsonr(ass[r,7,:], questionnaire[index], n_perm=n_perm))       
# Exctract significant correlations # [0]:corr, [1]:pvalue
sig1 = np.asarray([t[0] for t in corr1 if t[1] <= 0.1])
sig2 = np.asarray([t[0] for t in corr2 if t[1] <= 0.1])
corr1 = np.asarray(corr1)
corr2 = np.asarray(corr2)
# results
np.mean(np.abs(corr2[:,0]))/np.mean(np.abs(corr1[:,0]))
len(sig1)
len(sig2)

# regression analysis based on global patterns
import scipy.io as sio
global_measure = [eff, clc, ass, mod] # [lam, eff, clc, ass, mod]
num = 4
mse_glob_reg, r2_glob_reg = [[] for i in range(len(global_measure))], [[] for i in range(len(global_measure))] # number of measures
pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
for i in range(len(global_measure)):
    for s in range(np.shape(pred_set)[0]):
        # Training/testing sets and target variable
        X_train, y_train = (global_measure[i][:,pred_set[s][0],:]).T, questionnaire[index]
        X_test, y_test = (global_measure[i][:,pred_set[s][1],:]).T, questionnaire[index]        
        # Create linear regression object
        reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization        
        # Train the model using the training sets
        reg.fit(X_train, y_train)        
        # Make predictions using the testing set
        y_pred_reg = reg.predict(X_test)       
        # The mean squared error
        mse_glob_reg[i].append(mean_squared_error(y_test, y_pred_reg))        
        # The coefficient of determination: 1 is perfect prediction
        r2_glob_reg[i].append(r2_score(y_test, y_pred_reg))
# catplot (multiple barplot)
sns.set(style="whitegrid")
data = np.reshape(mse_glob_reg, (num*6,)) # 5 measures
df = pd.DataFrame(data=data, columns=["MSE"]) # index=rows
metric = np.repeat(['Global Efficiency', 'Mean Clustering Coefficient', 'Assortativity', 'Modularity'], 6, axis=0) # 5 measures
df['Measure'] = metric
group = np.tile(['Test 1', 'Test 2', 'Test 3'], 2*num)
df['Prediction set'] = group  
alignment = np.tile(['MSMAll', 'MSMAll', 'MSMAll', 'CHA', 'CHA', 'CHA'], num)
df['Alignment'] = alignment 
sns.set(style="whitegrid")
ax = sns.catplot(x="Prediction set", y="MSE",
                hue="Alignment", col="Measure",
                data=df, kind="bar",
                height=4, aspect=.7,
                palette=['#FFD700','#7F00FF'])

(ax.set_titles("{col_name}"))
   #.set_xticklabels(["T1", "T2", "T3"])
   #.set(ylim=(0, 1))
   #.despine(left=True)) 
#plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/Figures/catplot_global.pdf') 
plt.show()

#%% Brain maps based on MSEs
i = 0 # measure: 0, 1, 2, 3, 4, 5, ... [deg_l, stg_l, eig_l, clc_l, eff_l, par_l, zsc_l, kco_l]
s = 2 # prediction sets: 1, 2, 3
x = mse_reg[i][:,s-1] # MSM-All
y = mse_reg[i][:,s+2] # CHA
z = x-y # MSE(MSM-All)-MSE(CHA)
z = z/max(abs(z)) # normalize between -1 to 1
z = np.pad(z, (0, 19), 'constant'); # pad N=19 zeros to the right(end) of an array 

# Create  colormap using matplotlib
import matplotlib.colors as mcolors
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).   """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
c = mcolors.ColorConverter().to_rgb
# ONLY CHANGE THIS PART
# https://colordesigner.io/gradient-generator
my_cmap = make_colormap(
    [c('#363603'), c('#c2c208'), 0.15, 
     c('#c2c208'), c('yellow'), 0.30,
     c('yellow'), c('#fafabb'), 0.45,
     c('#fafabb'), c('white'), 0.50,
     c('white'), c('#e4b8ff'), 0.55, 
     c('#e4b8ff'), c('purple'), 0.75,
     c('purple'), c('#200033')])
#N = 1000
#array_dg = np.random.uniform(0, 10, size=(N, 2))
#colors = np.random.uniform(-2, 2, size=(N,))
#plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=my_cmap)
#plt.colorbar()
#plt.show()

# visualization using "plot_surf_stat_map"
atlas = hcp.mmp
view = 'lateral' # {‘lateral’, ‘medial’, ‘dorsal’, ‘ventral’, ‘anterior’, ‘posterior’},
h = 'left' # which hemisphere to train HA? 'left' or 'right'
if view == 'medial':
    if h == 'left':
        hcp_mesh = hcp.mesh.inflated_left
        hcp_data = hcp.left_cortex_data
        hcp_mesh_sulc = hcp.mesh.sulc_left
    elif h == 'right':
        hcp_mesh = hcp.mesh.inflated_right
        hcp_data = hcp.right_cortex_data
        hcp_mesh_sulc = hcp.mesh.sulc_right
else:
    hcp_mesh = hcp.mesh.inflated
    hcp_data = hcp.cortex_data
    hcp_mesh_sulc = hcp.mesh.sulc
    
plotting.plot_surf_stat_map(hcp_mesh, 
    hcp_data(hcp.unparcellate(z, atlas)), 
    hemi=h, view=view, cmap=my_cmap, colorbar=True, # vmax=0.9, # cmap='RdYlBu_r', 'cold_hot', 'seismic_r' # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    threshold=0.0000005, bg_map=hcp_mesh_sulc) # bg_map: a sulcal depth map for realistic shading
plt.savefig(view + '_aligned.pdf', dpi=300)

# interactive 3D visualization in a web browser ("view_surf")
atlas = hcp.mmp
view = 'medial' # {‘whole’ or ‘medial’}
h = 'right' # 'left' or 'right'
if view == 'medial':
    if h == 'left':
        hcp_mesh = hcp.mesh.inflated_left
        hcp_data = hcp.left_cortex_data
        hcp_mesh_sulc = hcp.mesh.sulc_left
    elif h == 'right':
        hcp_mesh = hcp.mesh.inflated_right
        hcp_data = hcp.right_cortex_data
        hcp_mesh_sulc = hcp.mesh.sulc_right
else:
    hcp_mesh = hcp.mesh.inflated
    hcp_data = hcp.cortex_data
    hcp_mesh_sulc = hcp.mesh.sulc
    
figure = plotting.view_surf(hcp_mesh, # vmax=0.9,
    hcp_data(hcp.unparcellate(z, atlas)), cmap=my_cmap, # seismic_r
    threshold=0.0000005, symmetric_cmap=False, colorbar=True, 
    bg_map=hcp_mesh_sulc)
figure.open_in_browser()

#%% Raincloud plots (with annotation) and Barplot 

# Raincloud plots
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt # pip install ptitprince
# https://d212y8ha88k086.cloudfront.net/manuscripts/16574/2509d3d1-e074-4b6a-86d4-497f4cb0895c_15191_-_rogier_kievit.pdf?doi=10.12688/wellcomeopenres.15191.1&numberOfBrowsableCollections=8&numberOfBrowsableInstitutionalCollections=0&numberOfBrowsableGateways=14
mse_reg = np.load('/Users/Farzad/Desktop/Figures/mse_reg.npy')

var_name = ['Degree Centrality', 'Strengths', 'Eigenvector Centrality', 'Clustering Coefficient', 'Local Efficiency', 'Participation Coefficient', 'Within-module Degree', 'K-coreness'] # 'Rich Club Coefficient', 
idx = [0,3,4,5,6,7]
mse_loc = [mse_reg[index] for index in idx] # Create list of chosen list items
sns.set_style("white")
f, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=True, figsize=(13, 8)) # dpi=300
for i, ax in enumerate(axes.flatten()):
    data = np.reshape(mse_loc[i].T, (6*360,)) # concatenate MSE array of a given measure
    df = pd.DataFrame(data=data, columns=["MSE"]) # index=rows
    group = np.concatenate((np.repeat(['1'],360,axis=0), np.repeat(['2'],360,axis=0), np.repeat(['3'],360,axis=0)) * 2, axis=0)
    df['Test Set'] = group  
    alignment = np.repeat(['MSMAll', 'CHA'], 3*360, axis=0) # 3 prediction sets
    df['Alignment'] = alignment  
    # replace values greater than specific value with median
    flat = np.array(df['MSE']).flatten()
    flat.sort()
    df.loc[df['MSE']>flat[-10],'MSE'] = df['MSE'].median() # remove from 10th largest value
    
    pt.RainCloud(x='Test Set', y='MSE', hue='Alignment', data=df, 
          palette=['#FFD700','#7F00FF'], width_viol=.7, width_box=.25,
          jitter=1, move=0, orient='h', alpha=.75, dodge=True,
          scale='area', cut=2, bw=.2, offset=None, ax=ax, # scale='width' or 'area'
          point_size=1, edgecolor='black', linewidth=1, pointplot=False) 
        
    sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)
    ax.set_title(var_name[idx[i]], fontsize=12) # title of plot
    ax.set_xlabel('MSE', fontsize = 12) # xlabel
    ax.set_ylabel('Test Set', fontsize = 12) # ylabel
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.get_legend().remove()    

#plt.legend(prop={'size':16}, frameon=False, bbox_to_anchor=(.48, -.06), loc='upper center')
# Adjust the layout of the plot (so titles and xlabels don't overlap?)
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/Figures/Raincloud_plott.pdf') 
plt.show() 



#%%
# Barplot: ΔMSE based on cole/anticevic networks
# find the indices of 360 regions based on 12 networks of cole/anticevic
import hcp_utils as hcp

index = np.zeros((360,))
for roi in range(1,361):
    r = roi-1
    index_parcel = np.where(hcp.ca_parcels.map_all==roi)[0][0] # first one is enough
    index[r] = hcp.ca_network.map_all[index_parcel]

var_name = ['Degree Centrality', 'Strengths', 'Eigenvector Centrality', 'Clustering Coefficient', 'Local Efficiency', 'Participation Coefficient', 'Within-module Degree', 'K-coreness'] # 'Rich Club Coefficient', 
idx = [0,3,4,5,6,7]
mse_loc = [mse_reg[index] for index in idx] # Create list of chosen list items
sns.set(style="white")
f, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(13, 8), dpi=300)
all_net = 0
for i, ax in enumerate(axes.flatten()):   
    ΔMSE1 = mse_loc[i][:,0] - mse_loc[i][:,3] # set 1
    ΔMSE2 = mse_loc[i][:,1] - mse_loc[i][:,4] # set 2
    ΔMSE3 = mse_loc[i][:,2] - mse_loc[i][:,5] # set 3
    ΔMSE = np.mean([ΔMSE1,ΔMSE2,ΔMSE3], axis=0)
    # create dataframe of ΔMSE and Network's assignemnet
    df = pd.DataFrame(data=[index, ΔMSE]).T
    df.columns = ['Network', 'ΔMSE']
    # calculate mean values of ΔMSE across networks
    df_group = df.groupby('Network', as_index=False)['ΔMSE'].mean()
    # barplot
    sns.barplot(x='Network', y='ΔMSE', data=df_group, ax=ax, saturation=0.75,
                palette=['#0020FF', '#7830F0', '#3EFCFD', '#B51DB4', 
                         '#00F300', '#009091', '#FFFE16', '#FB64FE', 
                         '#FF2E00', '#C47A31', '#FFB300', '#5A9B00'])
    sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)
    ax.set_title(var_name[idx[i]], fontsize=14) # title of plot
    ax.set(xlabel=None)
    ax.set_xticklabels(["Primary Visual", "Secondary Visual", "Somatomotor", "Cingulo-Opercular",
                        "Dorsal Attention", "Language", "Frontoparietal", "Auditory",
                        "Default", "Posterior Multimodal", "Ventral Multimodal", "Orbito-Affective"], rotation = 90)
    ax.axhline(y=df['ΔMSE'].mean(), color='r', linestyle='--', linewidth=1.5)    
    all_net = all_net + df_group
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/Figures/ΔMSE_Networks.pdf') 
plt.show() 

# Barplot: short version
index = np.zeros((360,))
for roi in range(1,361):
    r = roi-1
    index_parcel = np.where(hcp.ca_parcels.map_all==roi)[0][0] # first one is enough
    index[r] = hcp.ca_network.map_all[index_parcel]
var_name = ['Degree Centrality', 'Within-module Degree', 'K-coreness'] 
mse_loc = [mse_reg[index] for index in [0,6,7]] # Create list of chosen list items
sns.set(style="white")
f, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(13, 5), dpi=300)
for i, ax in enumerate(axes.flatten()):   
    ΔMSE1 = mse_loc[i][:,0] - mse_loc[i][:,3] # set 1
    ΔMSE2 = mse_loc[i][:,1] - mse_loc[i][:,4] # set 2
    ΔMSE3 = mse_loc[i][:,2] - mse_loc[i][:,5] # set 3
    ΔMSE = np.mean([ΔMSE1,ΔMSE2,ΔMSE3], axis=0)
    # create dataframe of ΔMSE and Network's assignemnet
    df = pd.DataFrame(data=[index, ΔMSE]).T
    df.columns = ['Network', 'ΔMSE']
    # calculate mean values of ΔMSE across networks
    df_group = df.groupby('Network', as_index=False)['ΔMSE'].mean()
    # barplot
    sns.barplot(x='Network', y='ΔMSE', data=df_group, ax=ax, saturation=0.75,
                palette=['#0020FF', '#7830F0', '#3EFCFD', '#B51DB4', 
                         '#00F300', '#009091', '#FFFE16', '#FB64FE', 
                         '#FF2E00', '#C47A31', '#FFB300', '#5A9B00'])
    sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)
    ax.set_title(var_name[i], fontsize=14) # title of plot
    ax.set(xlabel=None)
    ax.set_xticklabels(["Primary Visual", "Secondary Visual", "Somatomotor", "Cingulo-Opercular",
                        "Dorsal Attention", "Language", "Frontoparietal", "Auditory",
                        "Default", "Posterior Multimodal", "Ventral Multimodal", "Orbito-Affective"], rotation = 90)
    ax.axhline(y=df['ΔMSE'].mean(), color='r', linestyle='--', linewidth=1.5)    
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/Figures/ΔMSE_Networks.pdf') 
plt.show() 

#%% Graph analysis (Multi-layer modularity (Q) and number of communities)
import pandas as pd
import string

run = 2 # rset1LR -> 1; rest1RL -> 2; rest2LR -> 3; rest2RL -> 4

Q = []; num_cmty = []; num_cmty_ns = []
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_0.6,0.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_0.6,0.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_0.6,0.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_0.6,-1.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_0.6,-1.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_0.6,-1.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_0.6,-2.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_0.6,-2.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_0.6,-2.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_0.6,-3.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_0.6,-3.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_0.6,-3.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_0.6,-4.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_0.6,-4.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_0.6,-4.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_0.8,0.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_0.8,0.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_0.8,0.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_0.8,-1.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_0.8,-1.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_0.8,-1.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_0.8,-2.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_0.8,-2.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_0.8,-2.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_0.8,-3.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_0.8,-3.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_0.8,-3.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_0.8,-4.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_0.8,-4.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_0.8,-4.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.0,0.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.0,0.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.0,0.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.0,-1.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.0,-1.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.0,-1.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.0,-2.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.0,-2.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.0,-2.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.0,-3.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.0,-3.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.0,-3.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.0,-4.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.0,-4.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.0,-4.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.2,0.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.2,0.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.2,0.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.2,-1.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.2,-1.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.2,-1.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.2,-2.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.2,-2.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.2,-2.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.2,-3.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.2,-3.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.2,-3.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.2,-4.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.2,-4.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.2,-4.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.4,0.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.4,0.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.4,0.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.4,-1.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.4,-1.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.4,-1.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.4,-2.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.4,-2.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.4,-2.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.4,-3.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.4,-3.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.4,-3.0.mat', squeeze_me=True)['num_cmty_ns'])
Q.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_1.4,-4.0.mat', squeeze_me=True)['Q']); num_cmty.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_1.4,-4.0.mat', squeeze_me=True)['num_cmty']); num_cmty_ns.append(sio.loadmat('/Volumes/Elements/Modularity/variables/num_cmty_ns_1.4,-4.0.mat', squeeze_me=True)['num_cmty_ns'])
Q = np.asarray(Q); num_cmty = np.asarray(num_cmty); num_cmty_ns = np.asarray(num_cmty_ns)

datasetA1 = Q[:,:,run-1].T
datasetA2 = Q[:,:,run+3].T
datasetB1 = np.log(num_cmty[:,:,run-1].T)
datasetB2 = np.log(num_cmty[:,:,run+3].T)
datasetC1 = np.log(num_cmty_ns[:,:,run-1].T)
datasetC2 = np.log(num_cmty_ns[:,:,run+3].T)

col_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']
ticks = ['0', '-1', '-2', '-3', '-4', '0', '-1', '-2', '-3', '-4', '0', '-1', '-2', '-3', '-4', '0', '-1', '-2', '-3', '-4', '0', '-1', '-2', '-3', '-4']

dfA1 = pd.DataFrame(datasetA1, columns=col_name)
dfA2 = pd.DataFrame(datasetA2, columns=col_name)
dfB1 = pd.DataFrame(datasetB1, columns=col_name)
dfB2 = pd.DataFrame(datasetB2, columns=col_name)
dfC1 = pd.DataFrame(datasetC1, columns=col_name)
dfC2 = pd.DataFrame(datasetC2, columns=col_name)

names = []
valsA1, xsA1, valsA2, xsA2 = [],[], [],[]
valsB1, xsB1, valsB2, xsB2 = [],[], [],[]
valsC1, xsC1, valsC2, xsC2 = [],[], [],[]

for i, col in enumerate(dfA1.columns):
    valsA1.append(dfA1[col].values)
    valsA2.append(dfA2[col].values)
    valsB1.append(dfB1[col].values)
    valsB2.append(dfB2[col].values)
    valsC1.append(dfC1[col].values)
    valsC2.append(dfC2[col].values)
    names.append(col)
    # Add some random "jitter" to the data points
    xsA1.append(np.random.normal(i*3-0.5, 0.07, dfA1[col].values.shape[0]))
    xsA2.append(np.random.normal(i*3+0.5, 0.07, dfA2[col].values.shape[0]))
    xsB1.append(np.random.normal(i*3-0.5, 0.07, dfB1[col].values.shape[0]))
    xsB2.append(np.random.normal(i*3+0.5, 0.07, dfB2[col].values.shape[0]))
    xsC1.append(np.random.normal(i*3-0.5, 0.07, dfC1[col].values.shape[0]))
    xsC2.append(np.random.normal(i*3+0.5, 0.07, dfC2[col].values.shape[0]))
    
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(15, 7), dpi=300)
#fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(15, 15), dpi=300)

bpA1 = ax1.boxplot(valsA1, labels=names, positions=np.array(range(len(datasetA1[0])))*3-0.5, sym='', widths=0.7)
bpA2 = ax1.boxplot(valsA2, labels=names, positions=np.array(range(len(datasetA2[0])))*3+0.5, sym='', widths=0.7)
bpB1 = ax2.boxplot(valsB1, labels=names, positions=np.array(range(len(datasetB1[0])))*3-0.5, sym='', widths=0.7)
bpB2 = ax2.boxplot(valsB2, labels=names, positions=np.array(range(len(datasetB2[0])))*3+0.5, sym='', widths=0.7)
bpC1 = ax3.boxplot(valsC1, labels=names, positions=np.array(range(len(datasetC1[0])))*3-0.5, sym='', widths=0.7)
bpC2 = ax3.boxplot(valsC2, labels=names, positions=np.array(range(len(datasetC2[0])))*3+0.5, sym='', widths=0.7)
# Optional: change the color of 'boxes', 'whiskers', 'caps', 'medians', and 'fliers'
plt.setp(bpA1['medians'], color='r') # or color='#D7191C' ...
plt.setp(bpA2['medians'], linewidth=1, linestyle='-', color='r')
plt.setp(bpB1['medians'], color='r')
plt.setp(bpB2['medians'], linewidth=1, linestyle='-', color='r')
plt.setp(bpC1['medians'], color='r')
plt.setp(bpC2['medians'], linewidth=1, linestyle='-', color='r')

palette = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'tan', 'orchid', 'cyan', 'gold', 'crimson', 'r', 'g', 'b', 'y', 'm', 'c', 'k', 'tan', 'orchid', 'cyan', 'gold', 'crimson', 'r']

for xA1, xA2, valA1, valA2, color in zip(xsA1, xsA2, valsA1, valsA2, palette):
    ax1.scatter(xA1, valA1, s=5, alpha=0.4, color='#FFD700') # plt.plot(xA1, valA1, 'r.', alpha=0.4)
    ax1.scatter(xA2, valA2, s=5, alpha=0.4, color='#7F00FF')
    
for xB1, xB2, valB1, valB2, color in zip(xsB1, xsB2, valsB1, valsB2, palette):
    ax2.scatter(xB1, valB1, s=5, alpha=0.4, color='#FFD700')
    ax2.scatter(xB2, valB2, s=5, alpha=0.4, color='#7F00FF')   
    
for xC1, xC2, valC1, valC2, color in zip(xsC1, xsC2, valsC1, valsC2, palette):
    ax3.scatter(xC1, valC1, s=5, alpha=0.4, color='#FFD700')
    ax3.scatter(xC2, valC2, s=5, alpha=0.4, color='#7F00FF') 

# Use the pyplot interface to customize any subplot...
# First subplot
plt.sca(ax1)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Multi-layer modularity", fontweight='normal', fontsize=16)
plt.xlabel(r'Inter-subject coupling, $log_{10}(\omega)$', fontweight='normal', fontsize=16)
plt.plot([], c='#FFD700', label='MSM-All', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='#7F00FF', label='CHA', marker='o', linestyle='None', markersize=8)
plt.axvline(x=13.5, color='k', linestyle='--', linewidth=1)
plt.axvline(x=28.5, color='k', linestyle='--', linewidth=1)
plt.axvline(x=43.5, color='k', linestyle='--', linewidth=1)
plt.axvline(x=58.5, color='k', linestyle='--', linewidth=1)
plt.text(6, .65, '$\gamma=0.6$', ha='center', va='bottom', color='k', size=12)  
plt.text(21, .65, '$\gamma=0.8$', ha='center', va='bottom', color='k', size=12)  
plt.text(36, .65, '$\gamma=1.0$', ha='center', va='bottom', color='k', size=12)  
plt.text(51, .65, '$\gamma=1.2$', ha='center', va='bottom', color='k', size=12)  
plt.text(66, .65, '$\gamma=1.4$', ha='center', va='bottom', color='k', size=12)  
 
# Second subplot
plt.sca(ax2)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Communities (ln)", fontweight='normal', fontsize=16)
plt.xlabel(r'Inter-subject coupling, $log_{10}(\omega)$', fontweight='normal', fontsize=16)
plt.plot([], c='#FFD700', label='MSM-All', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='#7F00FF', label='CHA', marker='o', linestyle='None', markersize=8)
plt.axvline(x=13.5, color='k', linestyle='--', linewidth=1)
plt.axvline(x=28.5, color='k', linestyle='--', linewidth=1)
plt.axvline(x=43.5, color='k', linestyle='--', linewidth=1)
plt.axvline(x=58.5, color='k', linestyle='--', linewidth=1)
plt.text(6, 6.5, '$\gamma=0.6$', ha='center', va='bottom', color='k', size=12)  
plt.text(21, 6.5, '$\gamma=0.8$', ha='center', va='bottom', color='k', size=12)  
plt.text(36, 6.5, '$\gamma=1.0$', ha='center', va='bottom', color='k', size=12)  
plt.text(51, 6.5, '$\gamma=1.2$', ha='center', va='bottom', color='k', size=12)  
plt.text(66, 6.5, '$\gamma=1.4$', ha='center', va='bottom', color='k', size=12)  

# Third subplot
"""plt.sca(ax3)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Communities (non-singleton)", fontweight='normal', fontsize=16)
plt.xlabel(r'Inter-subject coupling, $log_{10}(\omega)$', fontweight='normal', fontsize=16)
plt.plot([], c='#FFD700', label='MSM-All', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='#7F00FF', label='CHA', marker='o', linestyle='None', markersize=8)
"""

# Unified legend  
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size':14})

# Annotate Subplots in a Figure with A, B, C 
"""for n, ax in enumerate((ax1, ax2)):
    ax.text(-0.05, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=18, weight='bold')
"""

sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)

# Adjust the layout of the plot
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/Figures/modularity_change.pdf') 
plt.show() 

#%% Graph analysis (ci and z-score)

import os # os.path as op
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import brainconn
import networkx as nx
from nilearn.connectome import ConnectivityMeasure
from brainconn import degree, centrality, clustering, core, distance, modularity, utils
from netneurotools import stats as nnstats
import nilearn.plotting as plotting
import hcp_utils as hcp
import scipy.io as sio

# calculating correlations
correlation_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional
os.chdir('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries_regional/mmp/') # /mmp/ or /yeo17/
#os.chdir('/dcl01/smart/data/fvfarahani/searchlight/timeseries_regional/')
#os.chdir('/users/ffarahan/')
atls = 'mmp' # mmp or yeo17
num_roi = 360 # 360, 17
num_par = 55 # 25
n_sbj = 30
num_measure = 1 # zsc

std_par = [[[] for i in range(num_par)] for i in range(8)]
std_zsc = [[[] for i in range(num_par)] for i in range(8)]

# Loading questionnaires's indices (for correlation analysis)
IQ = []
file_in = open('/Volumes/Elements/Hyperalignment/HCP/IQ.txt', 'r')
for z in file_in.read().split('\n'):
    IQ.append(float(z))
questionnaire = {'IQ': np.array(IQ)}
questionnaire_list = ['IQ']
# Set up a correlation vector (measure * Q_index) corresponding to a value for each region in the parcellation.
index = 'IQ' # Questionnaire index {AM, ESS, PW}
n_perm = 500

mse_reg, r2_reg = [], []
for i in range(num_measure):
    mse_reg.append(np.zeros([num_roi, 6, num_par])) # 6 distinct pred_sets
    r2_reg.append(np.zeros([num_roi, 6, num_par]))

"""ci = []
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.0,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.0,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.0,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.0,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.0,-4.0.mat', squeeze_me=True)['ci']);
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.5,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.5,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.5,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.5,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.5,-4.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.0,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.0,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.0,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.0,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.0,-4.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.5,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.5,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.5,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.5,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.5,-4.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_2.0,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_2.0,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_2.0,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_2.0,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_2.0,-4.0.mat', squeeze_me=True)['ci']); 
ci = np.asarray(ci);"""

ci = []
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.5,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.5,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.5,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.5,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.5,-4.0.mat', squeeze_me=True)['ci']);
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.6,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.6,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.6,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.6,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.6,-4.0.mat', squeeze_me=True)['ci']);
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.7,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.7,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.7,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.7,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.7,-4.0.mat', squeeze_me=True)['ci']);
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.8,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.8,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.8,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.8,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.8,-4.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.9,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.9,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.9,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.9,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_0.9,-4.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.0,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.0,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.0,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.0,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.0,-4.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.1,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.1,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.1,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.1,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.1,-4.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.2,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.2,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.2,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.2,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.2,-4.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.3,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.3,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.3,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.3,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.3,-4.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.4,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.4,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.4,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.4,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.4,-4.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.5,0.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.5,-1.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.5,-2.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.5,-3.0.mat', squeeze_me=True)['ci']); 
ci.append(sio.loadmat('/Volumes/Elements/Modularity/variables/ci_1.5,-4.0.mat', squeeze_me=True)['ci']); 
ci = np.asarray(ci);

for roi in range(1,num_roi+1): # example, 149 PFm
#for roi in range(1,18): # region/network of interet
    
    r = roi-1; # must start from zero
    
    ts_rest1LR = np.load('ts_rest1LR_' + atls + '_' + str(roi) + '.npy')
    ts_rest1RL = np.load('ts_rest1RL_' + atls + '_' + str(roi) + '.npy')
    ts_rest2LR = np.load('ts_rest2LR_' + atls + '_' + str(roi) + '.npy')
    ts_rest2RL = np.load('ts_rest2RL_' + atls + '_' + str(roi) + '.npy')
    ts_aligned_rest1LR = np.load('ts_aligned_rest1LR_' + atls + '_' + str(roi) + '.npy')
    ts_aligned_rest1RL = np.load('ts_aligned_rest1RL_' + atls + '_' + str(roi) + '.npy')
    ts_aligned_rest2LR = np.load('ts_aligned_rest2LR_' + atls + '_' + str(roi) + '.npy')
    ts_aligned_rest2RL = np.load('ts_aligned_rest2RL_' + atls + '_' + str(roi) + '.npy')
    
    corr_rest1LR = abs(correlation_measure.fit_transform(ts_rest1LR))
    corr_rest1RL = abs(correlation_measure.fit_transform(ts_rest1RL))
    corr_rest2LR = abs(correlation_measure.fit_transform(ts_rest2LR))
    corr_rest2RL = abs(correlation_measure.fit_transform(ts_rest2RL))
    corr_aligned_rest1LR = abs(correlation_measure.fit_transform(ts_aligned_rest1LR))
    corr_aligned_rest1RL = abs(correlation_measure.fit_transform(ts_aligned_rest1RL))
    corr_aligned_rest2LR = abs(correlation_measure.fit_transform(ts_aligned_rest2LR))
    corr_aligned_rest2RL = abs(correlation_measure.fit_transform(ts_aligned_rest2RL))    

    # remove the self-connections (zero diagonal)
    adj_wei = [[] for i in range(8)] # 8 sets (list of lists; wrong way -> adj_wei = [[]] * 8)
    for k in range(n_sbj): 
        adj_wei[0].append(corr_rest1LR[k] - np.eye(corr_rest1LR[k].shape[0]))
        adj_wei[1].append(corr_rest1RL[k] - np.eye(corr_rest1RL[k].shape[0]))
        adj_wei[2].append(corr_rest2LR[k] - np.eye(corr_rest2LR[k].shape[0]))
        adj_wei[3].append(corr_rest2RL[k] - np.eye(corr_rest2RL[k].shape[0]))
        adj_wei[4].append(corr_aligned_rest1LR[k] - np.eye(corr_aligned_rest1LR[k].shape[0]))
        adj_wei[5].append(corr_aligned_rest1RL[k] - np.eye(corr_aligned_rest1RL[k].shape[0]))
        adj_wei[6].append(corr_aligned_rest2LR[k] - np.eye(corr_aligned_rest2LR[k].shape[0]))
        adj_wei[7].append(corr_aligned_rest2RL[k] - np.eye(corr_aligned_rest2RL[k].shape[0]))
    
    # binarize      
    adj_bin = [[] for i in range(8)] # 8 sets 
    for i in range(8):
        for k in range(n_sbj):
            bin_matrix = brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei[i][k], 0.3, copy=True))
            adj_bin[i].append(bin_matrix)
    
    num_vertex = np.shape(adj_bin[0][0])[0]
        
    # define local measures        
    #par = [[[] for i in range(num_par)] for i in range(8)] # 8 sets, 25 parameter sets
    zsc = [[[] for i in range(num_par)] for i in range(8)] # 8 sets, 25 parameter sets

    # compute local measures
    for i in range(8):
        for j in range(num_par):
            for k in range(n_sbj): 
                #par[i][j].append(centrality.participation_coef_sign(adj_wei[i][k], ci[j,r][i,k,:])[0])
                zsc[i][j].append(centrality.module_degree_zscore(adj_wei[i][k], ci[j,r][i,k,:], flag=0)) # 0: undirected graph
        
    # standard deviations of all nodes(vertices)/brains
    for i in range(8):
        for j in range(num_par):
            # local measures
            #std_par[i][j].append(np.std(par[i][j], axis=0))
            std_zsc[i][j].append(np.std(zsc[i][j], axis=0))
    
    # Regression models
    # https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html#sphx-glr-auto-examples-cross-decomposition-plot-pcr-vs-pls-py
    from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    local_measure = [zsc] # [par, zsc]
    pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
    for i in range(len(local_measure)):
        for s in range(np.shape(pred_set)[0]):
            for j in range(num_par):
                # Training/testing sets and target variable
                X_train, y_train = local_measure[i][pred_set[s][0]][j], questionnaire[index]
                X_test, y_test = local_measure[i][pred_set[s][1]][j], questionnaire[index]            
                # Create linear regression object
                reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization           
                # Train the model using the training sets
                reg.fit(X_train, y_train)            
                # Make predictions using the testing set
                y_pred_reg = reg.predict(X_test)
                
                # The mean squared error
                mse_reg[i][r,s,j] = mean_squared_error(y_test, y_pred_reg)
                
                # The coefficient of determination: 1 is perfect prediction
                r2_reg[i][r,s,j] = r2_score(y_test, y_pred_reg)
        
    print(roi)

# Raincloud plots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt # pip install ptitprince
# https://d212y8ha88k086.cloudfront.net/manuscripts/16574/2509d3d1-e074-4b6a-86d4-497f4cb0895c_15191_-_rogier_kievit.pdf?doi=10.12688/wellcomeopenres.15191.1&numberOfBrowsableCollections=8&numberOfBrowsableInstitutionalCollections=0&numberOfBrowsableGateways=14

# zsc
# grouping based on test sets
"""for i in range(num_measure):
    for j in range(num_par):
        f, ax = plt.subplots(figsize=(7, 5), dpi=300)
        data = np.reshape(mse_reg[i][:,:,j].T, (6*360,)) # concatenate MSE array of a given measure
        df = pd.DataFrame(data=data, columns=["MSE"]) # index=rows
        group = np.concatenate((np.repeat(['Test 1'],360,axis=0), np.repeat(['Test 2'],360,axis=0), np.repeat(['Test 3'],360,axis=0)) * 2, axis=0)
        df['Prediction set'] = group  
        alignment = np.repeat(['MSMAll', 'CHA'], 3*360, axis=0) # 3 prediction sets
        df['Alignment'] = alignment  
        df.loc[df['MSE']>150,'MSE'] = df['MSE'].median() # replace values greater than specific value with median
        
        pt.RainCloud(x='Prediction set', y='MSE', hue='Alignment', data=df, 
              palette=['#FFD700','#7F00FF'], width_viol=.7, width_box=.25,
              jitter=1, move=0, orient='h', alpha=.75, dodge=True,
              scale='area', cut=2, bw=.2, offset=None, ax=ax, # scale='width' or 'area'
              point_size=2, edgecolor='black', linewidth=1, pointplot=False) """

# grouping based on log(omega)
"""test_set = 1 # 1, 2, 3    
for g in range(5): # gamma = 0:0.5:2 or 0.6:0.2:1.4   
    gamma = g*.2 + 0.6
    f, ax = plt.subplots(figsize=(7, 4), dpi=300)
    data = np.reshape(mse_reg[i][:,[test_set-1,test_set+2],g*5+1:(g+1)*5].T, (360*2*4,)) # concatenate MSE array of a given measure 2:MSMAll, CHA; 4: omega range
    #data = np.reshape(mse_reg[i][:,[0,3],:][:,:,[4,9,14,19,24]].T, (360*2*5,))
    df = pd.DataFrame(data=data, columns=["MSE"]) # index=rows
    group = np.concatenate((np.repeat(['-1'],720,axis=0), np.repeat(['-2'],720,axis=0), np.repeat(['-3'],720,axis=0), np.repeat(['-4'],720,axis=0)) * 1, axis=0)
    df[r'Inter-subject coupling, $log_{10}(\omega)$'] = group  
    alignment = np.concatenate((np.repeat(['MSMAll'],360,axis=0), np.repeat(['CHA'],360,axis=0)) * 4, axis=0) # 4: omega range
    df['Alignment'] = alignment  
    df.loc[df['MSE']>100,'MSE'] = df['MSE'].median() # replace values greater than specific value with median
    
    pt.RainCloud(x=r'Inter-subject coupling, $log_{10}(\omega)$', y='MSE', hue='Alignment', data=df, 
          palette=['#FFD700','#7F00FF'], width_viol=.7, width_box=.25,
          jitter=1, move=0, orient='h', alpha=.75, dodge=True,
          scale='area', cut=2, bw=.2, offset=None, ax=ax, # scale='width' or 'area'
          point_size=2, edgecolor='black', linewidth=1, pointplot=False) 
    plt.xlim(0, 100)
    plt.title(r'Structural resolution, $\gamma={}$'.format(gamma))"""
    
    
# grouping based on gamma
test_set = 1 # 1, 2, 3    
for o in range(5): # log(omega) = 0:-0.1:-0.4  
    log_omega = -1*o
    f, ax = plt.subplots(figsize=(3, 10), dpi=300)
    data = np.reshape(mse_reg[i][:,[test_set-1,test_set+2],:][:,:,[o,o+5,o+10,o+15,o+20,o+25,o+30,o+35,o+40,o+45,o+50]].T, (360*2*11,))
    #data = np.reshape(mse_reg[i][:,[0,3],:][:,:,[4,9,14,19,24]].T, (360*2*5,))
    df = pd.DataFrame(data=data, columns=["MSE"]) # index=rows
    group = np.concatenate((np.repeat(['0.5'],720,axis=0), np.repeat(['0.6'],720,axis=0), np.repeat(['0.7'],720,axis=0), np.repeat(['0.8'],720,axis=0),
                            np.repeat(['0.9'],720,axis=0), np.repeat(['1.0'],720,axis=0), np.repeat(['1.1'],720,axis=0), np.repeat(['1.2'],720,axis=0),
                            np.repeat(['1.3'],720,axis=0), np.repeat(['1.4'],720,axis=0), np.repeat(['1.5'],720,axis=0)) * 1, axis=0)
    df[r'Structural resolution, $\gamma$'] = group  
    alignment = np.concatenate((np.repeat(['MSMAll'],360,axis=0), np.repeat(['CHA'],360,axis=0)) * 11, axis=0) # 11: gamma range
    df['Alignment'] = alignment  
    df.loc[df['MSE']>100,'MSE'] = df['MSE'].median() # replace values greater than specific value with median
    
    pt.RainCloud(x=r'Structural resolution, $\gamma$', y='MSE', hue='Alignment', data=df, 
          palette=['#FFD700','#7F00FF'], width_viol=.7, width_box=.25,
          jitter=1, move=0, orient='h', alpha=.75, dodge=True,
          scale='area', cut=2, bw=.2, offset=None, ax=ax, # scale='width' or 'area'
          point_size=1.5, edgecolor='black', linewidth=1, pointplot=False) 
    plt.xlim(0, 100)
    plt.title(r'$log10(\omega)={}$'.format(log_omega))    


#%% regression analysis based on individual Q patterns
import scipy.io as sio
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

IQ = []
file_in = open('/Volumes/Elements/Hyperalignment/HCP/IQ.txt', 'r')
#file_in = open('/users/ffarahan/IQ.txt', 'r') # temporal
for z in file_in.read().split('\n'):
    IQ.append(float(z))
#questionnaire = {'AM': np.array(am), 'ESS': np.array(ess), 'PW': np.array(pw)}
questionnaire = {'IQ': np.array(IQ)}
questionnaire_list = ['IQ']
index = 'IQ' # Questionnaire index {AM, ESS, PW}

Q_ind = []
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.5,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.5,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.5,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.5,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.5,-4.0.mat', squeeze_me=True)['Q_ind']);
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.6,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.6,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.6,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.6,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.6,-4.0.mat', squeeze_me=True)['Q_ind']);
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.7,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.7,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.7,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.7,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.7,-4.0.mat', squeeze_me=True)['Q_ind']);
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.8,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.8,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.8,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.8,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.8,-4.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.9,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.9,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.9,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.9,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_0.9,-4.0.mat', squeeze_me=True)['Q_ind']);
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.0,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.0,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.0,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.0,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.0,-4.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.1,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.1,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.1,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.1,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.1,-4.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.2,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.2,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.2,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.2,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.2,-4.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.3,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.3,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.3,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.3,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.3,-4.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.4,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.4,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.4,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.4,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.4,-4.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.5,0.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.5,-1.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.5,-2.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.5,-3.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind.append(sio.loadmat('/Volumes/Elements/Modularity/variables/Q_ind_1.5,-4.0.mat', squeeze_me=True)['Q_ind']); 
Q_ind = np.asarray(Q_ind);

pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
num_par, num_set = np.shape(Q_ind)[0], np.shape(pred_set)[0]
mse_q_reg, r2_q_reg = np.zeros((num_par, num_set)), np.zeros((num_par, num_set))
for p in range(num_par):
    for s in range(num_set):
        # Training/testing sets and target variable
        X_train, y_train = (Q_ind[p,:,pred_set[s][0],:]).T, questionnaire[index]
        X_test, y_test = (Q_ind[p,:,pred_set[s][1],:]).T, questionnaire[index]        
        # Create linear regression object
        reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization        
        # Train the model using the training sets
        reg.fit(X_train, y_train)        
        # Make predictions using the testing set
        y_pred_reg = reg.predict(X_test)       
        # The mean squared error
        mse_q_reg[p,s] = mean_squared_error(y_test, y_pred_reg)       
        # The coefficient of determination: 1 is perfect prediction
        r2_q_reg[p,s] = r2_score(y_test, y_pred_reg)
        
mse_q_reg = np.reshape(mse_q_reg, (5, 11, 6), order='F')

mean_A1 = np.mean(mse_q_reg[:,:,0], axis=0)
std_A1 = np.std(mse_q_reg[:,:,0], axis=0)
mean_A2 = np.mean(mse_q_reg[:,:,3], axis=0)
std_A2 = np.std(mse_q_reg[:,:,3], axis=0)

mean_B1 = np.mean(mse_q_reg[:,:,1], axis=0)
std_B1 = np.std(mse_q_reg[:,:,1], axis=0)
mean_B2 = np.mean(mse_q_reg[:,:,4], axis=0)
std_B2 = np.std(mse_q_reg[:,:,4], axis=0)

mean_C1 = np.mean(mse_q_reg[:,:,2], axis=0)
std_C1 = np.std(mse_q_reg[:,:,2], axis=0)
mean_C2 = np.mean(mse_q_reg[:,:,5], axis=0)
std_C2 = np.std(mse_q_reg[:,:,5], axis=0)

x = np.arange(0.5,1.6,0.1)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(18, 4))
g1 = 'MSM-All'; g2 = 'CHA'; c1 = '#FFD700'; c2 = '#7F00FF'

plt.sca(ax1)
ebA1 = ax1.plot(x, mean_A1, '-ko', label=g1, markerfacecolor=c1)
ax1.fill_between(x, mean_A1 - std_A1, mean_A1 + std_A1, color=c1, alpha=0.3)
ebA2 = ax1.plot(x, mean_A2, '-ko', label=g2, markerfacecolor=c2)
ax1.fill_between(x, mean_A2 - std_A2, mean_A2 + std_A2, color=c2, alpha=0.3)
plt.title("Test 1 [REST1_RL]", fontweight='normal', fontsize=18)
plt.xlabel(r'Structural resolution, $\gamma$', fontsize=16)
plt.ylabel("MSE", fontweight='normal', fontsize=16)

plt.sca(ax2)
ebB1 = ax2.plot(x, mean_B1, '-ko', label=g1, markerfacecolor=c1)
ax2.fill_between(x, mean_B1 - std_B1, mean_B1 + std_B1, color=c1, alpha=0.3)
ebB2 = ax2.plot(x, mean_B2, '-ko', label=g2, markerfacecolor=c2)
ax2.fill_between(x, mean_B2 - std_B2, mean_B2 + std_B2, color=c2, alpha=0.3)
plt.title("Test 2 [REST2_LR]", fontweight='normal', fontsize=18)
plt.xlabel(r'Structural resolution, $\gamma$', fontsize=16)
plt.ylabel("MSE", fontweight='normal', fontsize=16)

plt.sca(ax3)
ebC1 = ax3.plot(x, mean_C1, '-ko', label=g1, markerfacecolor=c1)
ax3.fill_between(x, mean_C1 - std_C1, mean_C1 + std_C1, color=c1, alpha=0.3)
ebC2 = ax3.plot(x, mean_C2, '-ko', label=g2, markerfacecolor=c2)
ax3.fill_between(x, mean_C2 - std_C2, mean_C2 + std_C2, color=c2, alpha=0.3)
plt.title("Test 3 [REST2_RL]", fontweight='normal', fontsize=18)
plt.xlabel(r'Structural resolution, $\gamma$', fontsize=16)
plt.ylabel("MSE", fontweight='normal', fontsize=16)

#plt.legend(prop={'size':16}, ncol=7, frameon=False, bbox_to_anchor=(.48, -.06), loc='upper center')
plt.legend()

sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)

# Adjust the layout of the plot
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/Figures/individual_Q.pdf') 
plt.show()

#%% Regression (tensor entropy) --> NOT promising results

# https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html#sphx-glr-auto-examples-cross-decomposition-plot-pcr-vs-pls-py
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading questionnaires's indices (for correlation analysis)
IQ = []
file_in = open('/Volumes/Elements/Hyperalignment/HCP/IQ.txt', 'r')
for z in file_in.read().split('\n'):
    IQ.append(float(z))
questionnaire = {'IQ': np.array(IQ)}
questionnaire_list = ['IQ']
# Set up a correlation vector (measure * Q_index) corresponding to a value for each region in the parcellation.
index = 'IQ' # Questionnaire index {AM, ESS, PW}
n_perm = 500
num_roi = 360 # 360, 17
num_par = 25
n_sbj = 30

mse_entropy = np.zeros([num_par, num_roi, 6]) # 6 distinct pred_sets
r2_entropy = np.zeros([num_par, num_roi, 6])

entropy_tensor = []
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_0.0,0.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_0.0,-1.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_0.0,-2.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_0.0,-3.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_0.0,-4.0.mat', squeeze_me=True)['entropy_tensor']);
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_0.5,0.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_0.5,-1.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_0.5,-2.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_0.5,-3.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_0.5,-4.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_1.0,0.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_1.0,-1.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_1.0,-2.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_1.0,-3.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_1.0,-4.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_1.5,0.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_1.5,-1.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_1.5,-2.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_1.5,-3.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_1.5,-4.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_2.0,0.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_2.0,-1.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_2.0,-2.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_2.0,-3.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor.append(sio.loadmat('/Volumes/Elements/Modularity/variables/entropy_tensor_2.0,-4.0.mat', squeeze_me=True)['entropy_tensor']); 
entropy_tensor = np.asarray(entropy_tensor);

axis = 1 # 0(1) or 2

pred_set = np.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]])
for j in range(num_par):
    for r in range(num_roi):
        for s in range(np.shape(pred_set)[0]):    
            # Training/testing sets and target variable
            X_train, y_train = np.mean(entropy_tensor[j,r,pred_set[s][0]], axis=axis), questionnaire[index]
            X_test, y_test = np.mean(entropy_tensor[j,r,pred_set[s][1]], axis=axis), questionnaire[index]            
            # Create linear regression object
            reg = Ridge(alpha=1.0) # Linear least squares with l2 regularization           
            # Train the model using the training sets
            reg.fit(X_train, y_train)            
            # Make predictions using the testing set
            y_pred_reg = reg.predict(X_test)
            
            # The mean squared error
            mse_entropy[j,r,s] = mean_squared_error(y_test, y_pred_reg)
            
            # The coefficient of determination: 1 is perfect prediction
            r2_entropy[j,r,s] = r2_score(y_test, y_pred_reg)

    print(j)


# Raincloud plots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt # pip install ptitprince
# https://d212y8ha88k086.cloudfront.net/manuscripts/16574/2509d3d1-e074-4b6a-86d4-497f4cb0895c_15191_-_rogier_kievit.pdf?doi=10.12688/wellcomeopenres.15191.1&numberOfBrowsableCollections=8&numberOfBrowsableInstitutionalCollections=0&numberOfBrowsableGateways=14

for j in range(num_par):
    f, ax = plt.subplots(figsize=(7, 5), dpi=300)
    data = np.reshape(mse_entropy[j,:,:].T, (6*360,)) # concatenate MSE array of a given measure
    df = pd.DataFrame(data=data, columns=["MSE"]) # index=rows
    group = np.concatenate((np.repeat(['Test 1'],360,axis=0), np.repeat(['Test 2'],360,axis=0), np.repeat(['Test 3'],360,axis=0)) * 2, axis=0)
    df['Prediction set'] = group  
    alignment = np.repeat(['MSMAll', 'CHA'], 3*360, axis=0) # 3 prediction sets
    df['Alignment'] = alignment  
    df.loc[df['MSE']>150,'MSE'] = df['MSE'].median() # replace values greater than specific value with median
    
    pt.RainCloud(x='Prediction set', y='MSE', hue='Alignment', data=df, 
          palette=['#FFD700','#7F00FF'], width_viol=.7, width_box=.25,
          jitter=1, move=0, orient='h', alpha=.75, dodge=True,
          scale='area', cut=2, bw=.2, offset=None, ax=ax, # scale='width' or 'area'
          point_size=2, edgecolor='black', linewidth=1, pointplot=False) 




#%%
yeo17 = hcp.yeo17.map_all==2
yeo17lr = hcp.make_lr_parcellation(hcp.yeo17.ids==1)

aaa = hcp.parcellate(dss_aligned[0].S, yeo17lr) 

bbb = hcp.left_cortex_data(hcp.unparcellate(dss_aligned[0].S[16], hcp.yeo17))

for i in range(91282):
    ccc = hcp.unparcellate(dss_aligned[0].S[:,i], hcp.yeo17)

ddd = hcp.parcellate(ccc, yeo17lr) 

mmp_lr = hcp.make_lr_parcellation(hcp.mmp)


#%%
# load an specific individual surface and use it for visualization
# the group average surfaces are much smoother than the ones for individual subjects.
mesh_sub = hcp.load_surfaces(example_filename='/Users/Farzad/Desktop/Hyperalignment/HCP/100206.R.inflated_MSMAll.32k_fs_LR.surf.gii')
mesh_avg = hcp.load_surfaces(example_filename='/Users/Farzad/Desktop/Hyperalignment/HCP/S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii')

# let’s look at the same data as previously, but now on the inflated single subject surface:
plotting.view_surf(mesh_sub.inflated, hcp.cortex_data(Xn[29]), 
    threshold=1.5, bg_map=mesh_sub.sulc)

#%% Connected components
# decompose the region (where the condition is satisfied) into connected components

# E.g. insert a condition which is always true:
    # we would get just the two cortical hemispheres as the connected components
n_components, sizes, rois = hcp.cortical_components(Xn[29]>-1000.0)
n_components, sizes    

# a more realistic example:
n_components, sizes, rois = hcp.cortical_components(Xn[29]>1.0, cutoff=36)
n_components, sizes
# the largest connected component of size 237 could be displayed by
view = plotting.view_surf(hcp.mesh.inflated, 
    hcp.cortex_data(hcp.mask(Xn[29], rois==1)), 
    threshold=1.0, bg_map=hcp.mesh.sulc)
view.open_in_browser()

#%% Display the connectivity profiles
# -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.imshow(X_train[0][:,10000:10010])

plt.colorbar()
plt.show()

#%% Plot Aligned Voxel Time course 

import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from nltools.mask import create_sphere, expand_mask
from nltools.data import Brain_Data, Adjacency
from nltools.stats import align
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nilearn.plotting import plot_stat_map
import datalad.api as dl
import warnings

voxel_index = np.argmax(sim_aligned1)

voxel_unaligned = pd.DataFrame([ds.samples[:, voxel_index] for ds in dss_test1]).T
voxel_aligned = pd.DataFrame([ds.samples[:, voxel_index] for ds in dss_aligned1]).T

f, a = plt.subplots(nrows=2, figsize=(20, 5), sharex=True)
a[0].plot(voxel_unaligned, linestyle='-', alpha=.2)
a[0].plot(np.mean(voxel_unaligned, axis=1), linestyle='-', color='navy')
a[0].set_ylabel('Unaligned Voxel', fontsize=16)
a[0].yaxis.set_ticks([])

a[1].plot(voxel_aligned, linestyle='-', alpha=.2)
a[1].plot(np.mean(voxel_aligned, axis=1), linestyle='-', color='navy')
a[1].set_ylabel('Aligned Voxel', fontsize=16)
a[1].yaxis.set_ticks([])

plt.xlabel('Voxel Time Course (TRs)', fontsize=16)
a[0].set_title(f"Unaligned Voxel ISC: r={Adjacency(voxel_unaligned.corr(), matrix_type='similarity').mean():.02}", fontsize=18)
a[1].set_title(f"Aligned Voxel ISC: r={Adjacency(voxel_aligned.corr(), matrix_type='similarity').mean():.02}", fontsize=18)



