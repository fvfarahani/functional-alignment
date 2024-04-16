#%% Allegiance --> fine scale
import numpy as np
import os
import scipy.io as sio
import hcp_utils as hcp
# pip install teneto
# https://teneto.readthedocs.io/en/latest/tutorial.html
# https://teneto.readthedocs.io/en/latest/tutorial/networkmeasures.html
from teneto import communitymeasures

idx = int(os.getenv("SLURM_ARRAY_TASK_ID"))
#idx = 1

atlas = hcp.yeo17 # {‘mmp’, ‘ca_parcels’, ‘ca_network’, ‘yeo7’, ‘yeo17’}
#path = '/Volumes/Elements/Hyperalignment/HCP/200sbj/modularity/matlab_output/'
path = '/dcs05/ciprian/smart/farahani/SL-CHA/modularity/matlab_output/'

for net in range(idx,idx+1):
    roi_idx = np.where(atlas.map_all[:59412] == net)[0] # only cortex  
    
    # import community assignment of sessions
    communities = sio.loadmat(f'{path}S_net{net}.mat', squeeze_me=True)['S']
    
    num_set = communities.shape[0]
    
    # create static communities (networks' labels)
    static_communities = hcp.ca_parcels.map_all[roi_idx] # ca_parcels == mmp
    
    allegiance, flexibility, integration, recruitment = [], [], [], [] # promiscuity =[]
    
    for s in range(num_set):
        allegiance.append(communitymeasures.allegiance(communities[s]))  
        flexibility.append(communitymeasures.flexibility(communities[s]))
        integration.append(communitymeasures.integration(communities[s], static_communities))
        recruitment.append(communitymeasures.recruitment(communities[s], static_communities))
        #promiscuity.append(communitymeasures.promiscuity(communities[s])) # 0 entails only 1 community. 1 entails all communities
    
    print("Network: {} --> Length: {}".format(net, len(roi_idx)))
    
    #os.chdir('/Volumes/Elements/Hyperalignment/HCP/200sbj/modularity/allegiance/')
    os.chdir('/dcs05/ciprian/smart/farahani/SL-CHA/modularity/allegiance/')
    
    np.save(f'allegiance_{net}', allegiance)
    np.save(f'flexibility_{net}', flexibility)
    np.save(f'integration_{net}', integration)
    np.save(f'recruitment_{net}', recruitment)
    