#%% #######################################################################
# *****                         Graph Analysis                        *****
# #########################################################################
import os
import pickle
import numpy as np
import networkx as nx
import community
import hcp_utils as hcp
from nilearn.connectome import ConnectivityMeasure

is_coarse = False  # set to True for coarse calculations, False for fine calculations

# List of sessions
sessions = ['REST1_LR_MSM', 'REST1_RL_MSM', 'REST2_LR_MSM', 'REST2_RL_MSM',
            'REST1_LR_RHA', 'REST1_RL_RHA', 'REST2_LR_RHA', 'REST2_RL_RHA',
            'REST1_LR_CHA', 'REST1_RL_CHA', 'REST2_LR_CHA', 'REST2_RL_CHA']

session_name = sessions[0]

subjects = ['100206', '100307', '100408', '100610', '101006', '101107', '101309', '101915', '102008', '102311', '102513', '102816', '103111', '103414', '103515', '103818', '104012', '104416', '105014', '105115', '105216', '105620', '105923', '106016', '106319', '106521', '107018', '107321', '107422', '107725', '108121', '108222', '108323', '108525', '108828', '109123', '109325', '109830', '110007', '110411', '110613', '111009', '111312', '111413', '111716', '112112', '112314', '112516', '112920', '113215', '113619', '113922', '114217', '114318', '114419', '114621', '114823', '114924', '115017', '115219', '115320', '115825', '116524', '116726', '117122', '117324', '117930', '118124', '118528', '118730', '118932', '119126', '120212', '120515', '120717', '121416', '121618', '121921', '122317', '122620', '122822', '123117', '123420', '123521', '123824', '123925', '124220', '124422', '124624', '124826', '125525', '126325', '126628', '127327', '127630', '127933', '128026', '128127', '128632', '128935', '129028', '129129', '129331', '129634', '130013', '130316', '130417', '130619', '130821', '130922', '131217', '131419', '131722', '131823', '131924', '132017', '132118', '133019', '133827', '133928', '134021', '134223', '134324', '134425', '134728', '134829', '135225', '135528', '135730', '135932', '136227', '136732', '136833', '137027', '137128', '137229', '137633', '137936', '138231', '138534', '138837', '139233', '139637', '139839', '140319', '140824', '140925', '141119', '141826', '142828', '143426', '144125', '144428', '144832', '145127', '146129', '146331', '146432', '146533', '146937', '147030', '147737', '148032', '148133', '148335', '148840', '148941', '149236', '149337', '149539', '149741', '149842', '150625', '150726', '150928', '151223', '151425', '151526', '151627', '151728', '151829', '152831', '153025', '153227', '153429', '153631', '153833', '154229', '179245', '179346', '180129', '180432', '180735', '180836', '180937', '181131', '181232', '181636', '182032', '182436', '182739', '182840', '183034', '185139', '185341', '185442', '185846', '185947', '186141', '186444', '187143', '187547', '187850', '188347', '188448', '188751', '189349', '189450', '190031', '191033', '191336', '191437', '191942', '192035', '192136', '192540', '192641', '192843', '193239', '194140', '194645', '194746', '194847', '195041', '195647', '195849', '195950', '196144', '196346', '196750', '197348', '197550', '198249', '198350', '198451', '198653', '198855', '199150', '199251', '199453', '199655', '199958', '200008', '200614', '200917', '201111', '201414', '201818', '202113', '202719', '203418', '204016', '204319', '204420', '204622', '205725', '206222', '207123', '208024', '208125', '208226', '208327', '209127', '209228', '209329']
n_subjects = 200

if not is_coarse:
    # Define the ROI index for fine calculations
    roi = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    roi_idx = np.where(hcp.mmp.map_all[:59412] == roi)[0] # only cortex

# Define the connectivity measure
correlation_measure = ConnectivityMeasure(kind='correlation') 
# kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional

# Directory containing the data files (time-series)
data_dir = '/dcs05/ciprian/smart/farahani/SL-CHA/ts'

# Define the output directory and filename 
if is_coarse:
    output_dir = '/dcs05/ciprian/smart/farahani/SL-CHA/graph_measures/coarse/'
    output_filename = f'graph_measures_{session_name}.pickle'
else:
    output_dir = '/dcs05/ciprian/smart/farahani/SL-CHA/graph_measures/fine/'
    output_filename = f'graph_measures_roi{roi}_{session_name}.pickle'

output_path = os.path.join(output_dir, output_filename)


def extract_graph_measures(session_name, n_subjects, density):
    
    def calculate_path_length(G):
        # get the connected components
        components = list(nx.connected_components(G))
        # compute the average shortest path length for each non-isolated component
        component_lengths = []
        for component in components:
            # filter out isolated nodes
            subgraph = G.subgraph(component)
            non_isolated_nodes = [node for node in subgraph if len(list(subgraph.neighbors(node))) > 0]
            if len(non_isolated_nodes) > 1:
                subgraph = subgraph.subgraph(non_isolated_nodes)
                component_lengths.append(nx.average_shortest_path_length(subgraph))
        # Calculate the average path length over all non-isolated components
        if len(component_lengths) > 0:
            path_length = np.mean(component_lengths)
        else:
            path_length = np.nan
        return path_length
    
    def calculate_small_worldness(G, cc, pl, n_rand=10):
        rand_cc = []
        rand_pl = []
        for i in range(n_rand):
            RG = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
            rand_cc.append(nx.average_clustering(RG))
            rand_pl.append(nx.average_shortest_path_length(RG))
        rcc = np.mean(rand_cc)
        rpl = np.mean(rand_pl)
        return (cc / rcc) / (pl / rpl)
    
    # Initialize lists to store graph measures for all subjects
    graph_measures = {'path_length': [], 'global_clustering': [], 'global_efficiency': [], 'assortativity': [],
                      'modularity': [], 'small_worldness': [], 'degree': [], 'eigenvector_centrality': [], 
                      'closeness_centrality': [], 'pagerank_centrality': [], 'local_clustering': [],  'k_coreness': []}
    
    for subject in subjects[:n_subjects]:
        # Load the time series for the given subject and session
        ts_path = os.path.join(data_dir, session_name, f'{session_name}_{subject}.npy')
        ts = np.load(ts_path)
        if is_coarse:
            # parcelate time-series
            ts_p = hcp.parcellate(ts, hcp.mmp)[:,:360] # coarse matrix
        else:
            # extract time-series of the selected ROI
            ts_p = ts[:, roi_idx] # fine matrix
        # Calculate the correlation matrix for the parcelated time series
        corr = correlation_measure.fit_transform([ts_p])[0]
        # Binarize the correlation matrix based on density
        corr_flat = np.triu(corr, k=1).flatten() # Get the upper triangle elements of the correlation matrix
        corr_flat = corr_flat[corr_flat != 0] # Remove zeros
        corr_flat_sorted = np.sort(np.abs(corr_flat))[::-1] # Sort by absolute value in descending order
        num_edges = int(density * len(corr_flat)) # Calculate the number of edges to keep based on the density
        threshold = corr_flat_sorted[num_edges] # Get the threshold value based on the density
        corr_binary = np.where(np.abs(corr - np.diag(np.diag(corr))) >= threshold, 1, 0)
        
        # Create an undirected graph from the binary correlation matrix
        G = nx.from_numpy_array(corr_binary, create_using=nx.Graph())
        
        # Calculate global graph measures
        path_length = calculate_path_length(G) # path_length = nx.average_shortest_path_length(G)
        global_clustering = nx.average_clustering(G)
        global_efficiency = nx.global_efficiency(G)
        #local_efficiency = nx.local_efficiency(G) # time-consuming
        assortativity = nx.degree_assortativity_coefficient(G)
        modularity = community.modularity(community.best_partition(G), G) # compute the partition using the Louvain algorithm
        small_worldness = calculate_small_worldness(G, global_clustering, path_length)
        
        # Calculate local graph measures
        degree = dict(G.degree())
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
        closeness_centrality = nx.closeness_centrality(G)
        pagerank_centrality = nx.pagerank(G)
        local_clustering = nx.clustering(G)
        k_coreness = nx.core_number(G)
        
        # Store graph measures for this participant
        graph_measures['path_length'].append(path_length)
        graph_measures['global_clustering'].append(global_clustering)
        graph_measures['global_efficiency'].append(global_efficiency)
        graph_measures['assortativity'].append(assortativity)
        graph_measures['modularity'].append(modularity)
        graph_measures['small_worldness'].append(small_worldness)
        graph_measures['degree'].append(list(degree.values()))
        graph_measures['eigenvector_centrality'].append(list(eigenvector_centrality.values()))
        graph_measures['closeness_centrality'].append(list(closeness_centrality.values()))
        graph_measures['pagerank_centrality'].append(list(pagerank_centrality.values()))
        graph_measures['local_clustering'].append(list(local_clustering.values()))
        graph_measures['k_coreness'].append(list(k_coreness.values()))
        
        print(f'Subject {subject} processed.')
        
    # Serialize the graph_measures dictionary to a file in the output directory
    with open(output_path, 'wb') as f:
        pickle.dump(graph_measures, f)
    
    return graph_measures

# Extract graph measures for a given session
graph_measures = extract_graph_measures(session_name=session_name, n_subjects=n_subjects, density=0.3)

# =============================================================================
# # Load the serialized dictionary from the file
# with open('graph_measures.pickle', 'rb') as f:
#     graph_measures = pickle.load(f)  
# =============================================================================
