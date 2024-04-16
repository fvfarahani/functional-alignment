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

#%% Part 1: ROI Hyperalignment (Generate New Timeseries)
# #SBATCH --mem=50G

# # Define the ROIs and other parameters
# #roi = 11 
# roi = int(os.getenv("SLURM_ARRAY_TASK_ID")) # roi = int(os.getenv("SGE_TASK_ID"))
# roi_idx = np.where(hcp.mmp.map_all == roi)[0]
# n_sbj = 200
# subjects = ['100206', '100307', '100408', '100610', '101006', '101107', '101309', '101915', '102008', '102311', '102513', '102816', '103111', '103414', '103515', '103818', '104012', '104416', '105014', '105115', '105216', '105620', '105923', '106016', '106319', '106521', '107018', '107321', '107422', '107725', '108121', '108222', '108323', '108525', '108828', '109123', '109325', '109830', '110007', '110411', '110613', '111009', '111312', '111413', '111716', '112112', '112314', '112516', '112920', '113215', '113619', '113922', '114217', '114318', '114419', '114621', '114823', '114924', '115017', '115219', '115320', '115825', '116524', '116726', '117122', '117324', '117930', '118124', '118528', '118730', '118932', '119126', '120212', '120515', '120717', '121416', '121618', '121921', '122317', '122620', '122822', '123117', '123420', '123521', '123824', '123925', '124220', '124422', '124624', '124826', '125525', '126325', '126628', '127327', '127630', '127933', '128026', '128127', '128632', '128935', '129028', '129129', '129331', '129634', '130013', '130316', '130417', '130619', '130821', '130922', '131217', '131419', '131722', '131823', '131924', '132017', '132118', '133019', '133827', '133928', '134021', '134223', '134324', '134425', '134728', '134829', '135225', '135528', '135730', '135932', '136227', '136732', '136833', '137027', '137128', '137229', '137633', '137936', '138231', '138534', '138837', '139233', '139637', '139839', '140319', '140824', '140925', '141119', '141826', '142828', '143426', '144125', '144428', '144832', '145127', '146129', '146331', '146432', '146533', '146937', '147030', '147737', '148032', '148133', '148335', '148840', '148941', '149236', '149337', '149539', '149741', '149842', '150625', '150726', '150928', '151223', '151425', '151526', '151627', '151728', '151829', '152831', '153025', '153227', '153429', '153631', '153833', '154229', '179245', '179346', '180129', '180432', '180735', '180836', '180937', '181131', '181232', '181636', '182032', '182436', '182739', '182840', '183034', '185139', '185341', '185442', '185846', '185947', '186141', '186444', '187143', '187547', '187850', '188347', '188448', '188751', '189349', '189450', '190031', '191033', '191336', '191437', '191942', '192035', '192136', '192540', '192641', '192843', '193239', '194140', '194645', '194746', '194847', '195041', '195647', '195849', '195950', '196144', '196346', '196750', '197348', '197550', '198249', '198350', '198451', '198653', '198855', '199150', '199251', '199453', '199655', '199958', '200008', '200614', '200917', '201111', '201414', '201818', '202113', '202719', '203418', '204016', '204319', '204420', '204622', '205725', '206222', '207123', '208024', '208125', '208226', '208327', '209127', '209228', '209329']
# sessions = ['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL']

# cp_path = '/dcs05/ciprian/smart/farahani/SL-CHA/connectomes'
# cp_rest1LR = np.load(f'{cp_path}/cp_REST1_LR_200sbj.npy')[:, :, roi_idx] # .astype(np.float16)

# ds_train = []
# for k in range(n_sbj):
#     ds = Dataset(cp_rest1LR[k])   
#     ds.fa['node_indices'] = np.arange(ds.shape[1], dtype=int)
#     zscore(ds, chunks_attr=None)
#     ds_train.append(ds)

# # create common template space with training data   
# hyper = Hyperalignment()
# mappers = hyper(ds_train)  

# # Loop over sessions
# for session_name in sessions:
#     # Load and preprocess time series data for the current session
#     ts = []
#     ts_path = '/dcs05/ciprian/smart/farahani/SL-CHA/ts'
#     for subj in subjects[:n_sbj]:
#         ts.append(np.load(f'{ts_path}/{session_name}_MSM/{session_name}_MSM_{subj}.npy')[:, roi_idx])
#     print(session_name)
    
#     # ds = Dataset(ts)    
    
#     # Apply hyperalignment to time series data
#     ts_aligned = [mapper.forward(ds) for ds, mapper in zip(ts, mappers)]
#     _ = [zscore(ds, chunks_attr=None) for ds in ts_aligned]
        
#     # Save the aligned time series data
#     save_path = '/dcs05/ciprian/smart/farahani/SL-CHA/ts_roi'
#     save_filename = os.path.join(save_path, f'{session_name}_RHA_{roi}.npy')
#     np.save(save_filename, ts_aligned)

#%% Part 1': ROI Hyperalignment (Generate New CPs)
#SBATCH --mem=50G

# # Define the ROIs and other parameters
# #roi = 11 
# roi = int(os.getenv("SLURM_ARRAY_TASK_ID")) # roi = int(os.getenv("SGE_TASK_ID"))
# roi_idx = np.where(hcp.mmp.map_all == roi)[0]
# n_sbj = 200
# subjects = ['100206', '100307', '100408', '100610', '101006', '101107', '101309', '101915', '102008', '102311', '102513', '102816', '103111', '103414', '103515', '103818', '104012', '104416', '105014', '105115', '105216', '105620', '105923', '106016', '106319', '106521', '107018', '107321', '107422', '107725', '108121', '108222', '108323', '108525', '108828', '109123', '109325', '109830', '110007', '110411', '110613', '111009', '111312', '111413', '111716', '112112', '112314', '112516', '112920', '113215', '113619', '113922', '114217', '114318', '114419', '114621', '114823', '114924', '115017', '115219', '115320', '115825', '116524', '116726', '117122', '117324', '117930', '118124', '118528', '118730', '118932', '119126', '120212', '120515', '120717', '121416', '121618', '121921', '122317', '122620', '122822', '123117', '123420', '123521', '123824', '123925', '124220', '124422', '124624', '124826', '125525', '126325', '126628', '127327', '127630', '127933', '128026', '128127', '128632', '128935', '129028', '129129', '129331', '129634', '130013', '130316', '130417', '130619', '130821', '130922', '131217', '131419', '131722', '131823', '131924', '132017', '132118', '133019', '133827', '133928', '134021', '134223', '134324', '134425', '134728', '134829', '135225', '135528', '135730', '135932', '136227', '136732', '136833', '137027', '137128', '137229', '137633', '137936', '138231', '138534', '138837', '139233', '139637', '139839', '140319', '140824', '140925', '141119', '141826', '142828', '143426', '144125', '144428', '144832', '145127', '146129', '146331', '146432', '146533', '146937', '147030', '147737', '148032', '148133', '148335', '148840', '148941', '149236', '149337', '149539', '149741', '149842', '150625', '150726', '150928', '151223', '151425', '151526', '151627', '151728', '151829', '152831', '153025', '153227', '153429', '153631', '153833', '154229', '179245', '179346', '180129', '180432', '180735', '180836', '180937', '181131', '181232', '181636', '182032', '182436', '182739', '182840', '183034', '185139', '185341', '185442', '185846', '185947', '186141', '186444', '187143', '187547', '187850', '188347', '188448', '188751', '189349', '189450', '190031', '191033', '191336', '191437', '191942', '192035', '192136', '192540', '192641', '192843', '193239', '194140', '194645', '194746', '194847', '195041', '195647', '195849', '195950', '196144', '196346', '196750', '197348', '197550', '198249', '198350', '198451', '198653', '198855', '199150', '199251', '199453', '199655', '199958', '200008', '200614', '200917', '201111', '201414', '201818', '202113', '202719', '203418', '204016', '204319', '204420', '204622', '205725', '206222', '207123', '208024', '208125', '208226', '208327', '209127', '209228', '209329']
# sessions = ['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL']

# cp_path = '/dcs05/ciprian/smart/farahani/SL-CHA/connectomes'
# cp_rest1LR = np.load(f'{cp_path}/cp_REST1_LR_200sbj.npy')[:, :, roi_idx] # .astype(np.float16)

# ds_train = []
# for k in range(n_sbj):
#     ds = Dataset(cp_rest1LR[k])   
#     ds.fa['node_indices'] = np.arange(ds.shape[1], dtype=int)
#     zscore(ds, chunks_attr=None)
#     ds_train.append(ds)

# # create common template space with training data   
# hyper = Hyperalignment()
# mappers = hyper(ds_train)  

# # Loop over sessions
# for session_name in sessions:
#     # Load connectivity profiles (CP) data for the current session    
#     cp = np.load(f'{cp_path}/cp_{session_name}_200sbj.npy')[:, :, roi_idx]
#     print(session_name)
    
#     # Apply hyperalignment to CPs
#     cp_aligned = [mapper.forward(ds) for ds, mapper in zip(cp, mappers)]
#     _ = [zscore(ds, chunks_attr=None) for ds in cp_aligned]
    
#     # Save the aligned CPs
#     save_path = '/dcs05/ciprian/smart/farahani/SL-CHA/cp_roi'
#     save_filename = os.path.join(save_path, f'CP_{session_name}_{roi}.npy')
#     np.save(save_filename, cp_aligned)

# Not enough storage for these regions, do manually:
# ls *REST1_LR* | wc -l --> 356 {8, 181, 189, 311}
# ls *REST1_RL* | wc -l --> 354 {1, 8, 181, 188, 189, 311}
# ls *REST2_LR* | wc -l --> 350 {1, 8, 9, 181, 188, 189, 309, 311, 318, 331}
# ls *REST2_RL* | wc -l --> 349 {1, 8, 9, 51, 181, 188, 189, 309, 311, 318, 331}

#%% Part 2: Combine timeseries of regions
# #SBATCH --mem=5G

# k = int(os.getenv("SLURM_ARRAY_TASK_ID")) # subject number
# subjects = ['100206', '100307', '100408', '100610', '101006', '101107', '101309', '101915', '102008', '102311', '102513', '102816', '103111', '103414', '103515', '103818', '104012', '104416', '105014', '105115', '105216', '105620', '105923', '106016', '106319', '106521', '107018', '107321', '107422', '107725', '108121', '108222', '108323', '108525', '108828', '109123', '109325', '109830', '110007', '110411', '110613', '111009', '111312', '111413', '111716', '112112', '112314', '112516', '112920', '113215', '113619', '113922', '114217', '114318', '114419', '114621', '114823', '114924', '115017', '115219', '115320', '115825', '116524', '116726', '117122', '117324', '117930', '118124', '118528', '118730', '118932', '119126', '120212', '120515', '120717', '121416', '121618', '121921', '122317', '122620', '122822', '123117', '123420', '123521', '123824', '123925', '124220', '124422', '124624', '124826', '125525', '126325', '126628', '127327', '127630', '127933', '128026', '128127', '128632', '128935', '129028', '129129', '129331', '129634', '130013', '130316', '130417', '130619', '130821', '130922', '131217', '131419', '131722', '131823', '131924', '132017', '132118', '133019', '133827', '133928', '134021', '134223', '134324', '134425', '134728', '134829', '135225', '135528', '135730', '135932', '136227', '136732', '136833', '137027', '137128', '137229', '137633', '137936', '138231', '138534', '138837', '139233', '139637', '139839', '140319', '140824', '140925', '141119', '141826', '142828', '143426', '144125', '144428', '144832', '145127', '146129', '146331', '146432', '146533', '146937', '147030', '147737', '148032', '148133', '148335', '148840', '148941', '149236', '149337', '149539', '149741', '149842', '150625', '150726', '150928', '151223', '151425', '151526', '151627', '151728', '151829', '152831', '153025', '153227', '153429', '153631', '153833', '154229', '179245', '179346', '180129', '180432', '180735', '180836', '180937', '181131', '181232', '181636', '182032', '182436', '182739', '182840', '183034', '185139', '185341', '185442', '185846', '185947', '186141', '186444', '187143', '187547', '187850', '188347', '188448', '188751', '189349', '189450', '190031', '191033', '191336', '191437', '191942', '192035', '192136', '192540', '192641', '192843', '193239', '194140', '194645', '194746', '194847', '195041', '195647', '195849', '195950', '196144', '196346', '196750', '197348', '197550', '198249', '198350', '198451', '198653', '198855', '199150', '199251', '199453', '199655', '199958', '200008', '200614', '200917', '201111', '201414', '201818', '202113', '202719', '203418', '204016', '204319', '204420', '204622', '205725', '206222', '207123', '208024', '208125', '208226', '208327', '209127', '209228', '209329']
# sessions = ['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL']

# subj = subjects[k - 1]

# ts_path = '/dcs05/ciprian/smart/farahani/SL-CHA/ts'
# ts_roi_path = '/dcs05/ciprian/smart/farahani/SL-CHA/ts_roi'

# for session_name in sessions:
#     ts = np.load(f'{ts_path}/{session_name}_MSM/{session_name}_MSM_{subj}.npy')
    
#     for roi in range(1, 360 + 1):
#         roi_idx = np.where(hcp.mmp.map_all == roi)[0]
#         roi_data = np.load(f'{ts_roi_path}/{session_name}_RHA_{roi}.npy')[k - 1]
#         ts[:, roi_idx] = roi_data
#         print(roi)
        
#     # Save the aligned time series data
#     save_filename = os.path.join(ts_path, f'{session_name}_RHA', f'{session_name}_RHA_{subj}.npy')
#     np.save(save_filename, ts)
#     print(f'Time series data saved for Session: {session_name}, Subject: {subj}')

#%% Part 3: Calculate Inter-Subject Correlations (ISCs) for each ROI
# #SBATCH --mem=5G

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from mvpa2.mappers.zscore import zscore
from mvpa2.datasets.base import Dataset

roi = int(os.getenv("SLURM_ARRAY_TASK_ID"))
#roi = 2 
data_path = '/dcs05/ciprian/smart/farahani/SL-CHA/cp_roi/'

# Define the sessions
sessions = ['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL']

# Initialize lists to store sim_node and sim_subject for all sessions
sim_node_list = []
sim_subject_list = []

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
    sim = np.zeros((n_features, ))
    for i in range(n_features):
        data = np.array([ds.samples[:, i] for ds in dss])
        dist = pdist(data, metric)
        # "Correlation distance" is not the same as the correlation coefficient. 
        # A "distance" between two equal points is supposed to be 0.
        sim[i] = 1 - dist.mean()
    return sim

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
    sim = np.zeros((n_features, ))
    for i in range(n_features):
        data = np.array([ds.samples[:, i] for ds in dss])
        dist = cdist(data[k:k+1], data, metric)
        sim[i] = 1 - dist.mean()
    return sim.mean()


for session in sessions:
    # Loading the CPs for the current session
    cp = np.load(f'{data_path}CP_{session}_{roi}.npy')
    dss = []
    for k in range(len(cp)):
        ds = Dataset(cp[k])
        zscore(ds, chunks_attr=None) # normalize features (vertices)
        dss.append(ds)
    
    # Compute similarity metrics (node)
    sim_node = compute_average_similarity_node(dss)
    
    # Compute similarity metrics (subject)
    n_sbj = len(dss)
    sim_subject = np.zeros((n_sbj, ))
    for k in range(n_sbj):
        sim_subject[k] = compute_average_similarity_subject(dss, k)
        print(k)
    
    # Append sim_node and sim_subject to the lists
    sim_node_list.append(sim_node)
    sim_subject_list.append(sim_subject)

# Create a DataFrame from sim_node_list and sim_subject_list
sim_node_df = pd.DataFrame(np.array(sim_node_list).T, columns=sessions)
sim_subject_df = pd.DataFrame(np.array(sim_subject_list).T, columns=sessions)

# Save the files
save_path = '/dcs05/ciprian/smart/farahani/SL-CHA/ISC/RHA/'
sim_node_df.to_csv(f'{save_path}sim_node_{roi}.csv', sep='\t', index=False)
sim_subject_df.to_csv(f'{save_path}sim_subject_{roi}.csv', sep='\t', index=False)

#%% Combine ISCs across Regions (both sim_node and sim_subject)

import numpy as np
import pandas as pd
import hcp_utils as hcp

# Define the directory path
path = '/Volumes/Elements/Hyperalignment/HCP/200sbj/ISC/RHA/'

# Define the sessions
sessions = ['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL']

# SIM NODE & SIM SUBJECT
sim_node = np.zeros((59412, len(sessions))) # An empty array to store the sim_node
weighted_sum = np.zeros((200, len(sessions))) # An empty array to store the weighted sum

# Iterate over each session
for session_idx, session in enumerate(sessions):
    # Iterate over each ROI
    for roi in range(1, 360 + 1):
        
        roi_idx = np.where(hcp.mmp.map_all == roi)[0]
        roi_len = len(np.where(hcp.mmp.map_all == roi)[0])
        
        # SIM NODE
        roi_data_node = pd.read_csv(f'{path}/sim_node_{roi}.csv', sep='\t')[session].values
        sim_node[roi_idx, session_idx] = roi_data_node
        
        # SIM SUBJECT
        roi_data_subject = pd.read_csv(f'{path}/sim_subject_{roi}.csv', sep='\t')[session].values
        # Multiply the array by the voxel length to get the weighted array
        weighted_array = roi_data_subject * roi_len
        # Add the weighted array to the appropriate column of the weighted sum array
        weighted_sum[:, session_idx] += weighted_array

# Get the weighted average across all voxels
sim_subject = weighted_sum / 59412


## Loading MSM ISCs for Comparison Purpose
path_msm = '/Volumes/Elements/Hyperalignment/HCP/200sbj/ISC/'

# Load data
df_node_l = pd.read_csv(f'{path_msm}sim_node_200sbj_10r_L.inflated.csv', sep='\t', index_col=0)
df_subject_l = pd.read_csv(f'{path_msm}sim_subject_200sbj_10r_L.inflated.csv', sep='\t', index_col=0)
df_node_r = pd.read_csv(f'{path_msm}sim_node_200sbj_10r_R.inflated.csv', sep='\t', index_col=0)
df_subject_r = pd.read_csv(f'{path_msm}sim_subject_200sbj_10r_R.inflated.csv', sep='\t', index_col=0)

# Merge data
df_node = pd.concat([df_node_l, df_node_r], ignore_index=True)
df_subject = ((df_subject_l * 29696) + (df_subject_r * 29716)) / 59412


# Extract variables
var1 = df_node['rest2LR'].values  # MSM
var2 = sim_node[:, 2] # RHA
var3 = df_subject['rest2LR'].values  # MSM
var4 = sim_subject[:, 2] # RHA

#%% ISC Visualizations
import matplotlib.pyplot as plt

# 1.1) Average ISCs in each surface node 
plt.figure(figsize=(7, 7))
plt.scatter(var1, var2, s=25, alpha=0.2, edgecolors='none', c='k', label='Data Points')
plt.scatter(var1.mean(), var2.mean(), s=150, marker='o', color='r', edgecolors='k', label='Mean Point')
plt.xlim([-0.05, 0.8]) 
plt.ylim([-0.05, 0.8]) 
plt.xlabel('MSM', size=22)
plt.ylabel('rCHA', size=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot([-1, 1], [-1, 1], 'k--')
plt.text(var1.mean()-0.08, var2.mean()+0.03, '({:.2f}, {:.2f})'.format(var1.mean(), var2.mean()), fontsize=18, color='white', fontweight='bold')
#plt.title('Average Pairwise Correlation', size=18)
plt.tight_layout()
plt.savefig('ISC-rCHA-Vertex.png', dpi=300, bbox_inches='tight')
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
sns.distplot(var1, color="#FFD700", label="MSM", **kwargs)
sns.distplot(var2, color="#4CBB17", label="RHA", **kwargs)
plt.axvline(x=np.mean(list(var1)), linestyle='--', color='k', linewidth=2)
plt.axvline(x=np.mean(list(var2)), linestyle='--', color='k', linewidth=2)
plt.text(np.mean(list(var1)), plt.ylim()[1]*0.8, f"Mean: {np.mean(list(var1)):.2f}", va='top', ha='center', color='k', fontsize=18)
plt.text(np.mean(list(var2)), plt.ylim()[1]*0.7, f"Mean: {np.mean(list(var2)):.2f}", va='top', ha='center', color='k', fontsize=18)
plt.xlim(-0.05, 0.9)
plt.legend(prop={'size':20})
plt.savefig('distribution_ISC_rCHA.png', dpi=300, bbox_inches='tight')

# 1.3) Scatter plot of individual ISCs before and after CHA with linear fit
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import linregress
# Calculate Pearson correlation coefficient
corr_coef, _ = pearsonr(var3, var4)
# Create scatter plot with linear regression line
sns.set_style('white')
plt.figure(figsize=(7, 7))
ax = sns.regplot(x=var3, y=var4, scatter_kws={"s": 100, "alpha": 0.6, "edgecolors": 'none', "color": 'k'}, line_kws={"color": 'r'})
# Add text with correlation coefficient to plot
ax.text(0.05, 0.95, f'r = {corr_coef:.2f}', transform=ax.transAxes, fontsize=20, verticalalignment='top')
# Set axis labels and tick sizes
ax.set_xlabel('MSM', size=22)
ax.set_ylabel('rCHA', size=22)
# Set xtick and ytick labels with one decimal place
ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# Set the same number of ticks for both axes
ax.xaxis.set_major_locator(plt.MaxNLocator(9))
ax.yaxis.set_major_locator(plt.MaxNLocator(9))
# Set tick label sizes
ax.tick_params(labelsize=14)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
plt.plot([.25, .43], [.25, .43], 'k--')
# Set square aspect ratio
plt.axis('square')
# Save plot
plt.savefig('scatter_ISC_rCHA.png', dpi=300, bbox_inches='tight')
plt.show()

# 1.4) Visualize results on cortical brain map (ONLY LEF OR RIGHT)
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

#%% Plot ISCs using Workbench

import os
import nibabel as nib

session = 'REST2_LR' # var2

# Load the CIFTI2 file
mmp_path = '/Volumes/Elements/Hyperalignment/HCP/workbench/MMP/'
nifti_file = mmp_path + '/S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'
img = nib.load(nifti_file)

# Get the data array
data = img.get_fdata()

# Extract the session data: create a copy of the data array and replace it
data_RHA = var2.reshape(1, -1) # RHA : REST2_LR

# Create new CIFTI2 images
img_RHA = nib.Cifti2Image(data_RHA, img.header)

# Save the modified CIFTI2 file
output_file_RHA = os.path.join(mmp_path, 'output', f'ISC_RHA_{session}.dscalar.nii')
nib.save(img_RHA, output_file_RHA)

#%%
# #%% #######################################################################
# # Step 2 (GPU): parcel aligned timeseries 
# # #########################################################################   

# ts_parcellated_aligned_train = []
# ts_parcellated_aligned_test1 = []

# for k in range(n_sbj):
#     train = np.zeros((1200,target_nmbr), dtype="float16")
#     for roi in range(target_nmbr):
#         train[:,roi] = np.mean(np.load('/dcl01/smart/data/fvfarahani/connectivity_profiles_360mmp/ts_aligned_train_' + str(roi+1) + '.npy')[k], axis=1)
#     ts_parcellated_aligned_train.append(train) 
#     print(k)
    
# for k in range(n_sbj):
#     test1 = np.zeros((1200,target_nmbr), dtype="float16")
#     for roi in range(target_nmbr):
#         test1[:,roi] = np.mean(np.load('/dcl01/smart/data/fvfarahani/connectivity_profiles_360mmp/ts_aligned_test1_' + str(roi+1) + '.npy')[k], axis=1)
#     ts_parcellated_aligned_test1.append(test1) 
#     print(k)    
    
# np.save('ts_parcellated_aligned_train', ts_parcellated_aligned_train) 
# np.save('ts_parcellated_aligned_test1', ts_parcellated_aligned_test1)     