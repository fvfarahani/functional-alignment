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

#%% Part 1: ROI Hyperalignment
# #SBATCH --mem=50G

# # Define the ROIs and other parameters
# #roi = 11 
# roi = int(os.getenv("SLURM_ARRAY_TASK_ID")) # roi = int(os.getenv("SGE_TASK_ID"))
# roi_idx = np.where(hcp.mmp.map_all == roi)[0]
# n_sbj = 200
# subjects = ['100206', '100307', '100408', '100610', '101006', '101107', '101309', '101915', '102008', '102311', '102513', '102816', '103111', '103414', '103515', '103818', '104012', '104416', '105014', '105115', '105216', '105620', '105923', '106016', '106319', '106521', '107018', '107321', '107422', '107725', '108121', '108222', '108323', '108525', '108828', '109123', '109325', '109830', '110007', '110411', '110613', '111009', '111312', '111413', '111716', '112112', '112314', '112516', '112920', '113215', '113619', '113922', '114217', '114318', '114419', '114621', '114823', '114924', '115017', '115219', '115320', '115825', '116524', '116726', '117122', '117324', '117930', '118124', '118528', '118730', '118932', '119126', '120212', '120515', '120717', '121416', '121618', '121921', '122317', '122620', '122822', '123117', '123420', '123521', '123824', '123925', '124220', '124422', '124624', '124826', '125525', '126325', '126628', '127327', '127630', '127933', '128026', '128127', '128632', '128935', '129028', '129129', '129331', '129634', '130013', '130316', '130417', '130619', '130821', '130922', '131217', '131419', '131722', '131823', '131924', '132017', '132118', '133019', '133827', '133928', '134021', '134223', '134324', '134425', '134728', '134829', '135225', '135528', '135730', '135932', '136227', '136732', '136833', '137027', '137128', '137229', '137633', '137936', '138231', '138534', '138837', '139233', '139637', '139839', '140319', '140824', '140925', '141119', '141826', '142828', '143426', '144125', '144428', '144832', '145127', '146129', '146331', '146432', '146533', '146937', '147030', '147737', '148032', '148133', '148335', '148840', '148941', '149236', '149337', '149539', '149741', '149842', '150625', '150726', '150928', '151223', '151425', '151526', '151627', '151728', '151829', '152831', '153025', '153227', '153429', '153631', '153833', '154229', '179245', '179346', '180129', '180432', '180735', '180836', '180937', '181131', '181232', '181636', '182032', '182436', '182739', '182840', '183034', '185139', '185341', '185442', '185846', '185947', '186141', '186444', '187143', '187547', '187850', '188347', '188448', '188751', '189349', '189450', '190031', '191033', '191336', '191437', '191942', '192035', '192136', '192540', '192641', '192843', '193239', '194140', '194645', '194746', '194847', '195041', '195647', '195849', '195950', '196144', '196346', '196750', '197348', '197550', '198249', '198350', '198451', '198653', '198855', '199150', '199251', '199453', '199655', '199958', '200008', '200614', '200917', '201111', '201414', '201818', '202113', '202719', '203418', '204016', '204319', '204420', '204622', '205725', '206222', '207123', '208024', '208125', '208226', '208327', '209127', '209228', '209329']
# sessions = ['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL']

# cp_path = '/dcs05/ciprian/smart/farahani/SL-CHA/connectomes'
# cp_rest1LR = np.load(f'{cp_path}/cp_rest1LR_200sbj.npy')[:, :, roi_idx] # .astype(np.float16)

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

#%% Part 2: Combine timeseries of regions
#SBATCH --mem=5G

k = int(os.getenv("SLURM_ARRAY_TASK_ID")) # subject number
subjects = ['100206', '100307', '100408', '100610', '101006', '101107', '101309', '101915', '102008', '102311', '102513', '102816', '103111', '103414', '103515', '103818', '104012', '104416', '105014', '105115', '105216', '105620', '105923', '106016', '106319', '106521', '107018', '107321', '107422', '107725', '108121', '108222', '108323', '108525', '108828', '109123', '109325', '109830', '110007', '110411', '110613', '111009', '111312', '111413', '111716', '112112', '112314', '112516', '112920', '113215', '113619', '113922', '114217', '114318', '114419', '114621', '114823', '114924', '115017', '115219', '115320', '115825', '116524', '116726', '117122', '117324', '117930', '118124', '118528', '118730', '118932', '119126', '120212', '120515', '120717', '121416', '121618', '121921', '122317', '122620', '122822', '123117', '123420', '123521', '123824', '123925', '124220', '124422', '124624', '124826', '125525', '126325', '126628', '127327', '127630', '127933', '128026', '128127', '128632', '128935', '129028', '129129', '129331', '129634', '130013', '130316', '130417', '130619', '130821', '130922', '131217', '131419', '131722', '131823', '131924', '132017', '132118', '133019', '133827', '133928', '134021', '134223', '134324', '134425', '134728', '134829', '135225', '135528', '135730', '135932', '136227', '136732', '136833', '137027', '137128', '137229', '137633', '137936', '138231', '138534', '138837', '139233', '139637', '139839', '140319', '140824', '140925', '141119', '141826', '142828', '143426', '144125', '144428', '144832', '145127', '146129', '146331', '146432', '146533', '146937', '147030', '147737', '148032', '148133', '148335', '148840', '148941', '149236', '149337', '149539', '149741', '149842', '150625', '150726', '150928', '151223', '151425', '151526', '151627', '151728', '151829', '152831', '153025', '153227', '153429', '153631', '153833', '154229', '179245', '179346', '180129', '180432', '180735', '180836', '180937', '181131', '181232', '181636', '182032', '182436', '182739', '182840', '183034', '185139', '185341', '185442', '185846', '185947', '186141', '186444', '187143', '187547', '187850', '188347', '188448', '188751', '189349', '189450', '190031', '191033', '191336', '191437', '191942', '192035', '192136', '192540', '192641', '192843', '193239', '194140', '194645', '194746', '194847', '195041', '195647', '195849', '195950', '196144', '196346', '196750', '197348', '197550', '198249', '198350', '198451', '198653', '198855', '199150', '199251', '199453', '199655', '199958', '200008', '200614', '200917', '201111', '201414', '201818', '202113', '202719', '203418', '204016', '204319', '204420', '204622', '205725', '206222', '207123', '208024', '208125', '208226', '208327', '209127', '209228', '209329']
sessions = ['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL']

subj = subjects[k - 1]

ts_path = '/dcs05/ciprian/smart/farahani/SL-CHA/ts'
ts_roi_path = '/dcs05/ciprian/smart/farahani/SL-CHA/ts_roi'

for session_name in sessions:
    ts = np.load(f'{ts_path}/{session_name}_MSM/{session_name}_MSM_{subj}.npy')
    
    for roi in range(1, 360 + 1):
        roi_idx = np.where(hcp.mmp.map_all == roi)[0]
        roi_data = np.load(f'{ts_roi_path}/{session_name}_RHA_{roi}.npy')[k - 1]
        ts[:, roi_idx] = roi_data
        print(roi)
        
    # Save the aligned time series data
    save_filename = os.path.join(ts_path, f'{session_name}_RHA', f'{session_name}_RHA_{subj}.npy')
    np.save(save_filename, ts)
    print(f'Time series data saved for Session: {session_name}, Subject: {subj}')

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

# #%% #######################################################################
# # Step 3 (batch): Calculating conectivity profils based on aligned timeseries
# # #########################################################################

# cp_aligned_train = []
# cp_aligned_test1 = []

# #roi = 90
# roi = int(os.getenv("SGE_TASK_ID"))

# roi_idx = np.where(atlas.map_all == roi)[0]

# ts_aligned_train = np.load('/dcl01/smart/data/fvfarahani/connectivity_profiles_360mmp/ts_aligned_train_' + str(roi) + '.npy')
# ts_parcellated_aligned_train = np.load('/dcl01/smart/data/fvfarahani/connectivity_profiles_360mmp/ts_parcellated_aligned_train.npy')

# ts_aligned_test1 = np.load('/dcl01/smart/data/fvfarahani/connectivity_profiles_360mmp/ts_aligned_test1_' + str(roi) + '.npy')
# ts_parcellated_aligned_test1 = np.load('/dcl01/smart/data/fvfarahani/connectivity_profiles_360mmp/ts_parcellated_aligned_test1.npy')

# for k in range(n_sbj):   
#     Xc = np.zeros((target_nmbr,len(roi_idx)), dtype="float16")
#     for i in range(len(roi_idx)): 
#         a = ts_aligned_train[k][:,i]
#         for t in range(target_nmbr):   
#             b = ts_parcellated_aligned_train[k][:,t]
#             Xc[t,i] = np.corrcoef(a, b)[0, 1]
#     cp_aligned_train.append(Xc)  
#     print(k)  


# for k in range(n_sbj):   
#     Xc = np.zeros((target_nmbr,len(roi_idx)), dtype="float16")
#     for i in range(len(roi_idx)): 
#         a = ts_aligned_test1[k][:,i]
#         for t in range(target_nmbr):   
#             b = ts_parcellated_aligned_test1[k][:,t]
#             Xc[t,i] = np.corrcoef(a, b)[0, 1]
#     cp_aligned_test1.append(Xc)  
#     print(k)      
            
# np.save('cp_aligned_train_' + str(roi), cp_aligned_train) 
# np.save('cp_aligned_test1_' + str(roi), cp_aligned_test1) 


# #%% #######################################################################
# # Step 4: Inter-subject correlations
# # #########################################################################    

# def compute_average_similarity(dss, metric='correlation'):
    
#     n_features = dss[0].shape[1]
#     sim = np.zeros((n_features, ))
#     for i in range(n_features):
#         data = np.array([ds[:, i] for ds in dss])
#         dist = pdist(data, metric)
#         sim[i] = 1 - dist.mean()
#     return sim


# roi = 108

# cp_unaligned_train = np.load('/dcl01/smart/data/fvfarahani/connectivity_profiles_360mmp/cp_unaligned_train_' + str(roi) + '.npy')
# cp_unaligned_test1 = np.load('/dcl01/smart/data/fvfarahani/connectivity_profiles_360mmp/cp_unaligned_test1_' + str(roi) + '.npy')
# cp_aligned_train = np.load('/dcl01/smart/data/fvfarahani/connectivity_profiles_360mmp/cp_aligned_train_' + str(roi) + '.npy')
# cp_aligned_test1 = np.load('/dcl01/smart/data/fvfarahani/connectivity_profiles_360mmp/cp_aligned_test1_' + str(roi) + '.npy')

# sim_unaligned_train = compute_average_similarity(cp_unaligned_train)
# sim_unaligned_test1 = compute_average_similarity(cp_unaligned_test1)
# sim_aligned_train = compute_average_similarity(cp_aligned_train)
# sim_aligned_test1 = compute_average_similarity(cp_aligned_test1)

# np.mean(sim_aligned_train)/np.mean(sim_unaligned_train)
# np.mean(sim_aligned_test1)/np.mean(sim_unaligned_test1)   

# """