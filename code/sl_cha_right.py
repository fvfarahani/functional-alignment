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

#%% Prepare CPs for PyMVPA (Left or Right?): fMRI data points with mapped nodes (indices) to surface file, 
# no auxilliary array, no masking in HA is recommended since we already defined the node indices
# Datasets should have feature attribute `node_indices` containing spatial coordinates of all features

#% loading precalculated connectivity profiles
os.chdir('/dcs05/ciprian/smart/farahani/SL-CHA/connectomes/')

cp_rest1LR = np.load('cp_rest1LR_200sbj.npy')  

dss_rest1LR = []

# mapping data points to surface file
hemi = 'right' # 'left' or 'right'

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


for k in range(len(cp_rest1LR)):
      
    ds = Dataset(cp_rest1LR[k][:, hcp_struct])
    ds.fa['node_indices'] = node_indices # set feature attributes
    zscore(ds, chunks_attr=None) # normalize features (vertices) to have unit variance (GLM parameters estimates for each voxel at this point).
    dss_rest1LR.append(ds)

# Each run has 360(718) target regions and 29696 (or 29716) features per subject.
print(dss_rest1LR[0].shape)

del cp_rest1LR

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
    nproc = 1, # Number of processes to use. Change "Docker - Preferences - Advanced - CPUs" accordingly.
) 

# mask_node_ids = list(node_indices),
# exclude_from_model = list(nan_indices), # almost similar results with "mask_node_ids"
# roi_ids=np.unique(ds3.fa.node_indices),
# "roi_ids" and "mask_node_ids"
# combine_neighbormappers = True, # no differences in my case

# Create common template space with training data
os.chdir('/dcs05/ciprian/smart/farahani/SL-CHA/mappers/')
mappers = hyper(dss_rest1LR)

stop = timeit.default_timer()
print('Run time of the SearchlightHyperalignment:', stop - start)

h5save('mappers_'+ str(n_sbj) + 'sbj_' + str(sl_radius) + 'r_' + name + '.hdf5.gz', mappers, compression=9)

#mappers = h5load('mappers.hdf5.gz') # load pre-computed mappers





 