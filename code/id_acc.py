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





