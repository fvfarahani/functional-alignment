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

# calculating correlations
correlation_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional
os.chdir('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries_regional/mmp/') # /mmp/ or /yeo17/
#os.chdir('/dcl01/smart/data/fvfarahani/searchlight/timeseries_regional/')
#os.chdir('/users/ffarahan/')
atls = 'mmp' # mmp or yeo17
num_roi = 360 # 360, 17

num_measure = 2
num_loc_measure = 1
num_glb_measure = 1

# global measures 
ass = [[[] for i in range(8)] for i in range(num_roi)] # assortativity

std_stg_l = [[] for i in range(8)]

std_ass = [[] for i in range(8)]

n_sbj = 30
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






path_coarse = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries/'
path_fine = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/timeseries_regional/mmp/'
runs = ['rest1LR_', 'rest1RL_', 'rest2LR_', 'rest2RL_',
            'aligned_rest1LR_', 'aligned_rest1RL_', 'aligned_rest2LR_', 'aligned_rest2RL_']
n_run = len(runs)
        
ts_rest1LR_c = np.load(path_coarse + 'ts_' + runs[0] + '30sbj.npy')
ts_rest1RL_c = np.load(path_coarse + 'ts_' + runs[1] + '30sbj.npy')
ts_rest2LR_c = np.load(path_coarse + 'ts_' + runs[2] + '30sbj.npy')
ts_rest2RL_c = np.load(path_coarse + 'ts_' + runs[3] + '30sbj.npy')
ts_aligned_rest1LR_c = np.load(path_coarse + 'ts_' + runs[4] + '30sbj.npy')
ts_aligned_rest1RL_c = np.load(path_coarse + 'ts_' + runs[5] + '30sbj.npy')
ts_aligned_rest2LR_c = np.load(path_coarse + 'ts_' + runs[6] + '30sbj.npy')
ts_aligned_rest2RL_c = np.load(path_coarse + 'ts_' + runs[7] + '30sbj.npy')

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
    
    corr_rest1LR = correlation_measure.fit_transform(ts_rest1LR)
    corr_rest1RL = correlation_measure.fit_transform(ts_rest1RL)
    corr_rest2LR = correlation_measure.fit_transform(ts_rest2LR)
    corr_rest2RL = correlation_measure.fit_transform(ts_rest2RL)
    corr_aligned_rest1LR = correlation_measure.fit_transform(ts_aligned_rest1LR)
    corr_aligned_rest1RL = correlation_measure.fit_transform(ts_aligned_rest1RL)
    corr_aligned_rest2LR = correlation_measure.fit_transform(ts_aligned_rest2LR)
    corr_aligned_rest2RL = correlation_measure.fit_transform(ts_aligned_rest2RL)  

    #corr_rest1LR = correlation_measure.fit_transform(np.dstack((ts_rest1LR, ts_rest1LR_c)))
    #corr_rest1RL = correlation_measure.fit_transform(np.dstack((ts_rest1RL, ts_rest1LR_c)))
    #corr_rest2LR = correlation_measure.fit_transform(np.dstack((ts_rest2LR, ts_rest2RL_c)))
    #corr_rest2RL = correlation_measure.fit_transform(np.dstack((ts_rest2RL, ts_rest2RL_c)))
    #corr_aligned_rest1LR = correlation_measure.fit_transform(np.dstack((ts_aligned_rest1LR, ts_aligned_rest1LR_c)))
    #corr_aligned_rest1RL = correlation_measure.fit_transform(np.dstack((ts_aligned_rest1RL, ts_aligned_rest1LR_c)))
    #corr_aligned_rest2LR = correlation_measure.fit_transform(np.dstack((ts_aligned_rest2LR, ts_aligned_rest2RL_c)))
    #corr_aligned_rest2RL = correlation_measure.fit_transform(np.dstack((ts_aligned_rest2RL, ts_aligned_rest2RL_c))) 
    
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
    stg_l = [[] for i in range(8)]
    
    # compute global measures
    for i in range(8):
        for k in range(n_sbj): 
            ass[r][i].append(core.assortativity_wei(adj_wei[i][k], flag=0)) # 0: undirected graph
                                    
    # compute local measures
    for i in range(8):
        for k in range(n_sbj): 
            #stg_l[i].append(degree.strengths_und(adj_wei[i][k]))
            stg_l[i].append(degree.degrees_und(adj_bin[i][k]))
            
    stg_l = np.array(stg_l) 
        
    #from networkx.algorithms.smallworld import sigma
    #G = nx.from_numpy_matrix(adj_wei[0][1])     
    #sigma(G, niter=10, nrand=5, seed=None)
       
    # standard deviations of all nodes(vertices)/brains
    for i in range(8):
        # local measures
        std_stg_l[i].append(np.std(stg_l[i], axis=0))
        # global measures
        std_ass[i].append(np.std(ass[r][i], axis=0))
    
    # Regression models
    # https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html#sphx-glr-auto-examples-cross-decomposition-plot-pcr-vs-pls-py
    from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score
    
    local_measure = [stg_l] #eloc_w
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
        
    print(roi)
    


ass = np.array(ass)

std_stg_l = [np.hstack(std_stg_l[i]).squeeze() for i in range(8)]
std_ass = [np.hstack(std_ass[i]).squeeze() for i in range(8)]

var = [std_stg_l, std_ass]
var_name = ['Strengths', 'Assortativity'] 
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
i = 0 # measure: 0, 1, 2, 3, 4
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

# regression analysis based on global patterns
import scipy.io as sio
global_measure = [ass] # [lam, eff, clc, ass, mod]
num = 1
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
metric = np.repeat(['Assortativity'], 6, axis=0) # 5 measures
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
plt.show()    
    
    
    


# Raincloud plots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt # pip install ptitprince
# https://d212y8ha88k086.cloudfront.net/manuscripts/16574/2509d3d1-e074-4b6a-86d4-497f4cb0895c_15191_-_rogier_kievit.pdf?doi=10.12688/wellcomeopenres.15191.1&numberOfBrowsableCollections=8&numberOfBrowsableInstitutionalCollections=0&numberOfBrowsableGateways=14

# deg_l, stg_l, eig_l, clc_l, eff_l, par_l, zsc_l
var_name = ['Strengths'] 
mse_loc = [mse_reg[index] for index in [0]] # Create list of chosen list items
sns.set_style("white")
f, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=True, figsize=(13, 8), dpi=300)
for i, ax in enumerate(axes.flatten()):
    i = 0
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
          point_size=1.5, edgecolor='black', linewidth=1, pointplot=False) 
        
    sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)
    ax.set_title(var_name[i], fontsize=16) # title of plot
    #ax.set_xlabel('xlabel', fontsize = 14) # xlabel
    ax.set_ylabel('Test Set', fontsize = 14) # ylabel
    ax.get_legend().remove()    

#plt.legend(prop={'size':16}, frameon=False, bbox_to_anchor=(.48, -.06), loc='upper center')

# Adjust the layout of the plot (so titles and xlabels don't overlap?)
plt.tight_layout()
plt.show()     
    
    
    
index = np.zeros((360,))
for roi in range(1,361):
    r = roi-1
    index_parcel = np.where(hcp.ca_parcels.map_all==roi)[0][0] # first one is enough
    index[r] = hcp.ca_network.map_all[index_parcel]

sns.set(style="white")
f, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(13, 8), dpi=300)
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
    ax.set_title(var_name[i], fontsize=16) # title of plot
    ax.set(xlabel=None)
    ax.set_xticklabels(["Primary Visual", "Secondary Visual", "Somatomotor", "Cingulo-Opercular",
                        "Dorsal Attention", "Language", "Frontoparietal", "Auditory",
                        "Default", "Posterior Multimodal", "Ventral Multimodal", "Orbito-Affective"], rotation = 90)
    ax.axhline(y=df['ΔMSE'].mean(), color='r', linestyle='--', linewidth=1.5)
    
plt.tight_layout()
plt.show()     
    
    
    
    
    
    