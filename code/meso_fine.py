#%% Meso-fine: create FC .mat files (Yeo17 Networks)
import os
import numpy as np
import nibabel as nib
import hcp_utils as hcp
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
from scipy.io import savemat 
from brainconn import utils

# qsub -cwd -t 2:360 graph.sh --> only for fine calculations
net = int(os.getenv("SGE_TASK_ID")) # Network number in 17-Yeo Atlas
# net = 1:17

computer = 'JHPCE' # 'JHPCE' or 'local'

# Setting the main path for subjetcs' ts/fc data
if computer == 'JHPCE':
    main_path = '/dcs05/ciprian/smart/farahani/SL-CHA/'
elif computer == 'local':
    main_path = '/Volumes/Elements/Hyperalignment/HCP/200sbj/'

# List of sessions and subjects
sessions = ['REST1_LR_MSM', 'REST1_RL_MSM', 'REST2_LR_MSM', 'REST2_RL_MSM',
            'REST1_LR_CHA', 'REST1_RL_CHA', 'REST2_LR_CHA', 'REST2_RL_CHA']

subjects = ['100206', '100307', '100408', '100610', '101006', '101107', '101309', '101915', '102008', '102311', '102513', '102816', '103111', '103414', '103515', '103818', '104012', '104416', '105014', '105115', '105216', '105620', '105923', '106016', '106319', '106521', '107018', '107321', '107422', '107725', '108121', '108222', '108323', '108525', '108828', '109123', '109325', '109830', '110007', '110411', '110613', '111009', '111312', '111413', '111716', '112112', '112314', '112516', '112920', '113215', '113619', '113922', '114217', '114318', '114419', '114621', '114823', '114924', '115017', '115219', '115320', '115825', '116524', '116726', '117122', '117324', '117930', '118124', '118528', '118730', '118932', '119126', '120212', '120515', '120717', '121416', '121618', '121921', '122317', '122620', '122822', '123117', '123420', '123521', '123824', '123925', '124220', '124422', '124624', '124826', '125525', '126325', '126628', '127327', '127630', '127933', '128026', '128127', '128632', '128935', '129028', '129129', '129331', '129634', '130013', '130316', '130417', '130619', '130821', '130922', '131217', '131419', '131722', '131823', '131924', '132017', '132118', '133019', '133827', '133928', '134021', '134223', '134324', '134425', '134728', '134829', '135225', '135528', '135730', '135932', '136227', '136732', '136833', '137027', '137128', '137229', '137633', '137936', '138231', '138534', '138837', '139233', '139637', '139839', '140319', '140824', '140925', '141119', '141826', '142828', '143426', '144125', '144428', '144832', '145127', '146129', '146331', '146432', '146533', '146937', '147030', '147737', '148032', '148133', '148335', '148840', '148941', '149236', '149337', '149539', '149741', '149842', '150625', '150726', '150928', '151223', '151425', '151526', '151627', '151728', '151829', '152831', '153025', '153227', '153429', '153631', '153833', '154229', '179245', '179346', '180129', '180432', '180735', '180836', '180937', '181131', '181232', '181636', '182032', '182436', '182739', '182840', '183034', '185139', '185341', '185442', '185846', '185947', '186141', '186444', '187143', '187547', '187850', '188347', '188448', '188751', '189349', '189450', '190031', '191033', '191336', '191437', '191942', '192035', '192136', '192540', '192641', '192843', '193239', '194140', '194645', '194746', '194847', '195041', '195647', '195849', '195950', '196144', '196346', '196750', '197348', '197550', '198249', '198350', '198451', '198653', '198855', '199150', '199251', '199453', '199655', '199958', '200008', '200614', '200917', '201111', '201414', '201818', '202113', '202719', '203418', '204016', '204319', '204420', '204622', '205725', '206222', '207123', '208024', '208125', '208226', '208327', '209127', '209228', '209329']
n_subjects = 200

net_idx = np.where(hcp.yeo17.map_all[:59412] == net)[0]

for session in sessions:
        
    ts_all = []
    for subject in subjects[:n_subjects]:
        ts_path = os.path.join(main_path, 'dtseries', session, f"{session}_{subject}.dtseries.nii")
        img = nib.load(ts_path)
        ts = np.array(img.get_fdata().astype(np.float32))[:, net_idx]
        ts_all.append(ts)
        
    fc = correlation_measure.fit_transform(ts_all)
    
    # Saving FCs
    fc_path = os.path.join(main_path, 'fc_yeo17', f"net{net}", session)
    os.makedirs(fc_path, exist_ok=True)
    
    # Binarize the FC matrix 
    # Matrix too large to save with Matlab 5 format --> subjects one-by-one
    threshold = 0.3
    #binarized_fc = np.where(fc >= threshold, 1, 0).astype(np.int32)
    for k, subject in enumerate(subjects[:n_subjects]): 
        binarized_fc = utils.binarize(utils.threshold_proportional(fc[k], threshold, copy=True))
        binarized_fc = np.float32(binarized_fc)
        savemat(os.path.join(fc_path, f'{session}_{subject}.mat'), {'sbj_' + session: binarized_fc}, do_compression=True) 
    
    print(session)