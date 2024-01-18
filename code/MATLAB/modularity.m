%{
clear
clc

% iterated_GenLouvain (fMRI data)
roi = 1;
data = load(append('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/corr_regional/mmp_', int2str(roi), '/corr_rest1LR.mat'));
data_aligned = load(append('/Volumes/Elements/Hyperalignment/HCP/results_30sbj/corr_regional/mmp_', int2str(roi), '/corr_aligned_rest1LR.mat'));
corr_rest1LR = permute(data.corr_rest1LR, [2 3 1]);
corr_aligned_rest1LR = permute(data_aligned.corr_aligned_rest1LR, [2 3 1]);
A = squeeze(num2cell(corr_rest1LR,[1 2]));
A_aligned = squeeze(num2cell(corr_aligned_rest1LR,[1 2]));
N=length(A{1}); % number of nodes
T=length(A); % number of layers
counter = 1; info = zeros(1,10);
for gamma = 0:0.5:2 % gamma = 0:0.01:2
    for log_omega = 0:-1:-4 % log_omega = 0:-0.01:-4
        omega = power(10,log_omega);
        [B,twom] = multicat(A,gamma,omega);
        [B_aligned,twom_aligned] = multicat(A_aligned,gamma,omega);
        PP = @(S)postprocess_categorical_multilayer(S,T);
        [S,Q] = iterated_genlouvain(B,10000,0,1,'moverandw',[], PP); % 4th entry(randord): 0[move] or 1[moverand]
        [S_aligned,Q_aligned] = iterated_genlouvain(B_aligned,10000,0,1,'moverandw',[], PP);
        Q = Q/twom;
        Q_aligned = Q_aligned/twom_aligned;
        S = reshape(S,N,T);
        S_aligned = reshape(S_aligned,N,T);
        
        % entropy
        K = max(S,[],'all'); % number of communities
        K_aligned = max(S_aligned,[],'all'); 
        p = zeros(N,K); p_aligned = zeros(N,K_aligned);
        h = zeros(N,1); h_aligned = zeros(N,1);
        % for anatomical alignemnet
        for n = 1:N
            for k = 1:K
                p(n,k) = sum(S(n,:)==k)/T;
            end
            np = p(n,:);
            np = np(np>0); % only non-zero entries [log(0) = -Inf]
            h(n) = -sum(np.*log2(np));
        end
        % for hyperalignemnet
        for n = 1:N
            for k = 1:K_aligned
                p_aligned(n,k) = sum(S_aligned(n,:)==k)/T;
            end
            np_aligned = p_aligned(n,:);
            np_aligned = np_aligned(np_aligned>0); % only non-zero entries [log(0) = -Inf]
            h_aligned(n) = -sum(np_aligned.*log2(np_aligned));
        end
        
        % non-singltone communities
        bin = unique(S); bin_aligned = unique(S);
        freq = [bin, histc(S(:),bin)]; freq_aligned = [bin_aligned, histc(S_aligned(:),bin_aligned)];
        idx = find(freq(:,2)==1); idx_aligned = find(freq_aligned(:,2)==1);
        freq(idx,:) = []; freq_aligned(idx_aligned,:) = [];
        
        info(counter,:) = [gamma omega Q Q_aligned K K_aligned...
            size(freq,1) size(freq_aligned,1) mean(h) mean(h_aligned)];
        counter = counter+1;
    end
    fprintf('gamma = %f\n', gamma);
end

%}

%% 200 subjects (meso-coarse)
clear
clc
        
gamma = 1.0; % gamma = 0.5:0.1:1.5
log_omega = -1; % log_omega = 0:-1:-4
omega = power(10,log_omega);        

corr_g1 = permute(cell2mat(struct2cell(load('/Volumes/Elements/Hyperalignment/HCP/200sbj/ptseries_fc/REST1_LR_MSM.mat'))), [2 3 1]);
corr_g2 = permute(cell2mat(struct2cell(load('/Volumes/Elements/Hyperalignment/HCP/200sbj/ptseries_fc/REST1_LR_CHA.mat'))), [2 3 1]);

% model settings 
A_g1 = squeeze(num2cell(corr_g1,[1 2])); % adjacency/connectivity matrix
A_g2 = squeeze(num2cell(corr_g2,[1 2]));         
N = length(A_g1{1}); % number of nodes (same for both groups)
T_g1 = size(corr_g1,3); % number of layers (subjects)
T_g2 = size(corr_g2,3);

% multi-layer modularity (Q) and partiotions
% Group 1
[B_g1,twom_g1] = multicat(A_g1,gamma,omega);
PP_g1 = @(S_g1)postprocess_categorical_multilayer(S_g1,T_g1);
[S_g1,Q_g1] = iterated_genlouvain(B_g1,10000,0,1,'moverandw',[], PP_g1); % 4th entry(randord): 0[move] or 1[moverand] | 5th: move, moverand, or moverandw
Q_g1 = Q_g1/twom_g1;
S_g1 = reshape(S_g1,N,T_g1);
C_g1 = mode(S_g1,2); % consensus
K_g1 = max(S_g1,[],'all'); % number of communities
% Group 2
[B_g2,twom_g2] = multicat(A_g2,gamma,omega);
PP_g2 = @(S_g2)postprocess_categorical_multilayer(S_g2,T_g2);
[S_g2,Q_g2] = iterated_genlouvain(B_g2,10000,0,1,'moverandw',[], PP_g2); % 4th entry(randord): 0[move] or 1[moverand] | 5th: move, moverand, or moverandw
Q_g2 = Q_g2/twom_g2;
S_g2 = reshape(S_g2,N,T_g2);
C_g2 = mode(S_g2,2); % consensus
K_g2 = max(S_g2,[],'all'); % number of communities

cd '/Volumes/Elements/Hyperalignment/HCP/200sbj/modularity/';           
filename = sprintf('S_MSM_%.1f,%.1f.mat', gamma, log_omega); save(filename,'S_g1');
filename = sprintf('S_CHA_%.1f,%.1f.mat', gamma, log_omega); save(filename,'S_g2');

%% 200 Subjects (meso-fine)
% qrsh -l mem_free=100G,h_vmem=100G,h_fsize=100G --> net: {othres}
% qrsh -l mem_free=200G,h_vmem=200G,h_fsize=100G --> net: {3}
% module load matlab
% matlab

clear, clc

net = 1;

% 'JHPCE' or 'local'
computer = 'JHPCE';

% Add the GenLouvain directory and all its subfolders to MATLAB's search path
if strcmp(computer, 'JHPCE')
    addpath(genpath('/dcs05/ciprian/smart/farahani/SL-CHA/code/MATLAB/GenLouvain'));
end

gamma = 1.0; % gamma = 0.5:0.1:1.5
log_omega = -1; % log_omega = 0:-1:-4
omega = power(10,log_omega);

% Select two sessions (MSM vs. CHA)
sessions = {'REST1_LR_MSM', 'REST1_LR_CHA'};

% Define the network-vertex# map
network_map = containers.Map(1:17, [3875, 3352, 5980, 4836, 3485, 3370, 4871, 3273, 2464, 1976, 1307, 3476, 3479, 2042, 1489, 4843, 4548]);

subjects = {'100206', '100307', '100408', '100610', '101006', '101107', '101309', '101915', '102008', '102311', '102513', '102816', '103111', '103414', '103515', '103818', '104012', '104416', '105014', '105115', '105216', '105620', '105923', '106016', '106319', '106521', '107018', '107321', '107422', '107725', '108121', '108222', '108323', '108525', '108828', '109123', '109325', '109830', '110007', '110411', '110613', '111009', '111312', '111413', '111716', '112112', '112314', '112516', '112920', '113215', '113619', '113922', '114217', '114318', '114419', '114621', '114823', '114924', '115017', '115219', '115320', '115825', '116524', '116726', '117122', '117324', '117930', '118124', '118528', '118730', '118932', '119126', '120212', '120515', '120717', '121416', '121618', '121921', '122317', '122620', '122822', '123117', '123420', '123521', '123824', '123925', '124220', '124422', '124624', '124826', '125525', '126325', '126628', '127327', '127630', '127933', '128026', '128127', '128632', '128935', '129028', '129129', '129331', '129634', '130013', '130316', '130417', '130619', '130821', '130922', '131217', '131419', '131722', '131823', '131924', '132017', '132118', '133019', '133827', '133928', '134021', '134223', '134324', '134425', '134728', '134829', '135225', '135528', '135730', '135932', '136227', '136732', '136833', '137027', '137128', '137229', '137633', '137936', '138231', '138534', '138837', '139233', '139637', '139839', '140319', '140824', '140925', '141119', '141826', '142828', '143426', '144125', '144428', '144832', '145127', '146129', '146331', '146432', '146533', '146937', '147030', '147737', '148032', '148133', '148335', '148840', '148941', '149236', '149337', '149539', '149741', '149842', '150625', '150726', '150928', '151223', '151425', '151526', '151627', '151728', '151829', '152831', '153025', '153227', '153429', '153631', '153833', '154229', '179245', '179346', '180129', '180432', '180735', '180836', '180937', '181131', '181232', '181636', '182032', '182436', '182739', '182840', '183034', '185139', '185341', '185442', '185846', '185947', '186141', '186444', '187143', '187547', '187850', '188347', '188448', '188751', '189349', '189450', '190031', '191033', '191336', '191437', '191942', '192035', '192136', '192540', '192641', '192843', '193239', '194140', '194645', '194746', '194847', '195041', '195647', '195849', '195950', '196144', '196346', '196750', '197348', '197550', '198249', '198350', '198451', '198653', '198855', '199150', '199251', '199453', '199655', '199958', '200008', '200614', '200917', '201111', '201414', '201818', '202113', '202719', '203418', '204016', '204319', '204420', '204622', '205725', '206222', '207123', '208024', '208125', '208226', '208327', '209127', '209228', '209329'};
num_sbj = 50;
subjects = subjects(1:num_sbj);

% Variable settings
Q = zeros(1, numel(sessions)); % Initialize Q array
S = cell(1, numel(sessions)); % Initialize S cell array
C = cell(1, numel(sessions)); % Initialize C cell array
K = zeros(1, numel(sessions)); % Initialize K array

N = network_map(net); % number of nodes (same for both groups)
T = num_sbj; % number of layers (subjects)

for s = 1:numel(sessions)
    % Setting adjacency/connectivity matrices
    A = cell(num_sbj, 1); % Initialize the cell array
    for i = 1:num_sbj
        subject_id = subjects{i};
        if strcmp(computer, 'local')
            file_path = sprintf('/Volumes/Elements/Hyperalignment/HCP/200sbj/fc_yeo17/net%d/%s/%s_%s.mat', net, sessions{s}, sessions{s}, subject_id);
        elseif strcmp(computer, 'JHPCE')
            file_path = sprintf('/dcs05/ciprian/smart/farahani/SL-CHA/fc_yeo17/net%d/%s/%s_%s.mat', net, sessions{s}, sessions{s}, subject_id);
        end
        subject_data = load(file_path, ['sbj_' sessions{s}]);
        A{i} = subject_data.(['sbj_' sessions{s}]); % Assign the subject's data to the cell
    end
    % Multi-layer modularity (Q)
    [B, twom] = multicat(A, gamma, omega);
    PP = @(S)postprocess_categorical_multilayer(S, T);
    [S{1,s}, Q_raw] = iterated_genlouvain(B, 10000, 0, 1, 'moverandw', [], PP); % 4th entry(randord): 0[move] or 1[moverand] | 5th: move, moverand, or moverandw
    Q(1,s) = Q_raw / twom;
    S{1,s} = reshape(S{1,s}, N, T);
    C{1,s} = mode(S{1,s}, 2); % consensus
    K(1,s) = max(S{1,s}, [], 'all'); % number of communities

    fprintf('Session = %s\n', sessions{s});

end

if strcmp(computer, 'local')
    cd '/Volumes/Elements/Hyperalignment/HCP/200sbj/modularity/matlab_output/';
elseif strcmp(computer, 'JHPCE')
    cd '/dcs05/ciprian/smart/farahani/SL-CHA/modularity/matlab_output/';
end

filename = sprintf('Q_net%d', net); save(filename,'Q');
filename = sprintf('S_net%d', net); save(filename,'S');
filename = sprintf('C_net%d', net); save(filename,'C');
filename = sprintf('K_net%d', net); save(filename,'K');

%% 30 subjects
clear
clc

num_roi = 360; % 360
num_sub = 30;
num_set = 4*2; % [rest1LR, rest1RL, rest2LR, rest2RL] * 2
path = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/corr_regional/mmp/mmp_';

for gamma = 1.1:0.1:1.3 % gamma = 0.8:0.1:1.5 or 0:0.01:2
    for log_omega = -1:-1:-2 % log_omega = 0:-1:-4 or 0:-0.01:-4

    omega = power(10,log_omega);
    Q = zeros(num_roi,num_set);
    %Q_ind = zeros(num_roi,num_set,num_sub);
    S_a = cell(num_roi,1);
    %entropy_tensor = cell(num_roi,num_set);
    %entropy = cell(num_roi,1);
    consensus = cell(num_roi,1);
    %num_cmty = zeros(num_roi,num_set); % number of communities
    %num_cmty_ns = zeros(num_roi,num_set); % number of communities (non-singleton)
    A = cell(num_sub,num_set);

        for roi = 1:num_roi
            corr_rest1LR = abs(permute(load(append(path, int2str(roi), '/bin_rest1LR.mat')).bin_rest1LR, [2 3 1]));
            corr_rest1RL = abs(permute(load(append(path, int2str(roi), '/bin_rest1RL.mat')).bin_rest1RL, [2 3 1]));
            corr_rest2LR = abs(permute(load(append(path, int2str(roi), '/bin_rest2LR.mat')).bin_rest2LR, [2 3 1]));
            corr_rest2RL = abs(permute(load(append(path, int2str(roi), '/bin_rest2RL.mat')).bin_rest2RL, [2 3 1]));    
            corr_aligned_rest1LR = abs(permute(load(append(path, int2str(roi), '/bin_aligned_rest1LR.mat')).bin_aligned_rest1LR, [2 3 1]));
            corr_aligned_rest1RL = abs(permute(load(append(path, int2str(roi), '/bin_aligned_rest1RL.mat')).bin_aligned_rest1RL, [2 3 1]));
            corr_aligned_rest2LR = abs(permute(load(append(path, int2str(roi), '/bin_aligned_rest2LR.mat')).bin_aligned_rest2LR, [2 3 1]));
            corr_aligned_rest2RL = abs(permute(load(append(path, int2str(roi), '/bin_aligned_rest2RL.mat')).bin_aligned_rest2RL, [2 3 1]));

            A(:,1) = squeeze(num2cell(corr_rest1LR,[1 2]));
            A(:,2) = squeeze(num2cell(corr_rest1RL,[1 2]));
            A(:,3) = squeeze(num2cell(corr_rest2LR,[1 2]));
            A(:,4) = squeeze(num2cell(corr_rest2RL,[1 2]));
            A(:,5) = squeeze(num2cell(corr_aligned_rest1LR,[1 2]));
            A(:,6) = squeeze(num2cell(corr_aligned_rest1RL,[1 2]));
            A(:,7) = squeeze(num2cell(corr_aligned_rest2LR,[1 2]));
            A(:,8) = squeeze(num2cell(corr_aligned_rest2RL,[1 2]));

            N = length(A{1}); % number of nodes
            T = num_sub; % number of layers (subjects)
            %h = zeros(N,num_set); % entropy
            m = zeros(N,num_set); % consensus partition

            for set = 1:num_set

                % multi-layer modularity (Q) and partiotions
                [B,twom] = multicat(A(:,set),gamma,omega);
                PP = @(S)postprocess_categorical_multilayer(S,T);
                [S,Q(roi,set)] = iterated_genlouvain(B,10000,0,1,'moverandw',[], PP); % 4th entry(randord): 0[move] or 1[moverand]
                Q(roi,set) = Q(roi,set)/twom;
                S = reshape(S,N,T);
                S_a{roi}(set,:,:) = S';
                m(:,set) = mode(S,2);
                
                %{
                % entropy
                K = max(S,[],'all'); % number of communities
                num_cmty(roi,set) = K;
                p = zeros(N,K);
                for n = 1:N
                    for k = 1:K
                        p(n,k) = sum(S(n,:)==k)/T;
                    end
                    np = p(n,:);
                    np = np(np>0); % only non-zero entries [log(0) = -Inf]
                    h(n,set) = -sum(np.*log2(np));
                end 

                % entropy tensor
                for i = 1:T
                    for j = 1:T
                        entropy_tensor{roi,set}(i,j,:) = S(:,i)~=S(:,j);
                    end
                end

                % non-singltone communities
                bin = unique(S); % number of communities
                freq = [bin, histc(S(:),bin)];
                % remove singltone communities
                idx = find(freq(:,2)==1);
                freq(idx,:) = [];
                num_cmty_ns(roi,set) = length(freq);

                % compute single-layer modularity (Q individuals) based on the extracted partitions 
                for sub = 1:num_sub
                    [S_a{roi}(set,sub,:), Q_ind(roi,set,sub)] = community_louvain(cell2mat(A(sub,set)), gamma, S(:,sub), 'modularity'); % negative_sym
                end
                %}
                
                fprintf('Set = %f\n', set);

            end

            %entropy{roi} = h;
            consensus{roi} = m;

            fprintf('ROI = %f\n', roi);
        end
        
        cd '/Volumes/Elements/Modularity/var_mmp/';
                
        filename = sprintf('Q_%.1f,%.1f.mat', gamma, log_omega); save(filename,'Q');
        %filename = sprintf('num_cmty_%.1f,%.1f.mat', gamma, log_omega); save(filename,'num_cmty');
        %filename = sprintf('num_cmty_ns_%.1f,%.1f.mat', gamma, log_omega); save(filename,'num_cmty_ns');
        filename = sprintf('S_a_%.1f,%.1f.mat', gamma, log_omega); save(filename,'S_a');
        %filename = sprintf('Q_ind_%.1f,%.1f.mat', gamma, log_omega); save(filename,'Q_ind');
        %filename = sprintf('entropy_%.1f,%.1f.mat', gamma, log_omega); save(filename,'entropy');
        %filename = sprintf('entropy_tensor_%.1f,%.1f.mat', gamma, log_omega); save(filename,'entropy_tensor');
        filename = sprintf('consensus_%.1f,%.1f.mat', gamma, log_omega); save(filename,'consensus');
        
        fprintf('gamma = %f, log_omega = %f\n', gamma, log_omega);
        
    end
    
end

%% temporary
clear
clc

num_sub = 30;
runs = ["rest1LR"; "rest1RL"; "rest2LR"; "rest2RL"; "aligned_rest1LR"; "aligned_rest1RL"; "aligned_rest2LR"; "aligned_rest2RL"];
path = '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/corr_regional/yeo17/yeo17_';

gamma = 1.0; % gamma = 0.5:0.1:1.5
log_omega = -1; % log_omega = 0:-1:-4
omega = power(10,log_omega);

for roi = 15:15
    
    S_a = []; % labels
    Q_a = []; % multi-layer modularity
    K_a = []; % number of modules(components)
    C_a = []; % consensus

    for run = 1:length(runs)
        % 4 pieces of binary matrix
        bin_ul = permute(cell2mat(struct2cell(load(append(path, int2str(roi), '/bin_', runs(run), '_ul.mat')))), [2 3 1]);
        bin_ur = permute(cell2mat(struct2cell(load(append(path, int2str(roi), '/bin_', runs(run), '_ur.mat')))), [2 3 1]);
        bin_ll = permute(cell2mat(struct2cell(load(append(path, int2str(roi), '/bin_', runs(run), '_ll.mat')))), [2 3 1]);
        bin_lr = permute(cell2mat(struct2cell(load(append(path, int2str(roi), '/bin_', runs(run), '_lr.mat')))), [2 3 1]);
        % concatenate the halves separately
        upper_half = cat(2,bin_ul,bin_ur);
        lower_half = cat(2,bin_ll,bin_lr);
        % reconstruct the whole binary matrix (adjacency matrix)
        bin_mat = cat(1,upper_half,lower_half);

        % model settings
        A = squeeze(num2cell(bin_mat,[1 2])); % adjacency matrix
        N = length(A{1}); % number of nodes
        T = num_sub; % number of layers (subjects)

        % multi-layer modularity (Q) and partiotions
        [B,twom] = multicat(A,gamma,omega);
        PP = @(S)postprocess_categorical_multilayer(S,T);
        [S,Q] = iterated_genlouvain(B,10000,0,1,'moverandw',[], PP); % 4th entry(randord): 0[move] or 1[moverand] | 5th: move, moverand, or moverandw
        Q = Q/twom; % modularity
        Q_a = cat(1, Q_a, Q); % modularity (all runs)
        S = reshape(S,N,T); % community labels
        S_a = cat(3, S_a, S); % community labels (all runs)
        K = max(S,[],'all'); % number of communities
        K_a = cat(1, K_a, K); % number of communities (all runs)
        C = mode(S,2); % consensus (on labels across subjects)
        C_a = cat(2, C_a, C); % consensus (all runs)

        fprintf('Run = %f\n', run);
    end

    S_a = permute(S_a, [3,1,2]); % change axes locations
    C_a = permute(C_a, [2,1]); % change axes locations

    cd '/Volumes/Elements/Hyperalignment/HCP/results_30sbj/corr_regional/yeo17/modularity_var/';           
    filename = sprintf('Q_a_net%.d,%.1f,%.1f.mat', roi, gamma, log_omega); save(filename,'Q_a');
    filename = sprintf('S_a_net%.d,%.1f,%.1f.mat', roi, gamma, log_omega); save(filename,'S_a');
    filename = sprintf('K_a_net%.d,%.1f,%.1f.mat', roi, gamma, log_omega); save(filename,'K_a');
    filename = sprintf('C_a_net%.d,%.1f,%.1f.mat', roi, gamma, log_omega); save(filename,'C_a');
    
    fprintf('ROI = %f\n', roi);

end

%%
% consensus partition
m(:,1) = mode(S,2); m(:,2) = mode(S_aligned,2); 

% load matrix info
%info = struct2cell(load('/Volumes/Elements/Modularity/info.mat'));
%info = cat(1,info{:});

% scatterplot (gamma vs. omega vs. communities vs. entropy)
Gamma = info(:,1);
Omega = log10(info(:,2));
t = tiledlayout(2,3); % Requires R2019b or later
% Top left plot(tile)
nexttile
Community = info(:,3);
scatter(Gamma,Omega,[],Community,'filled');
colorbar;
title('Modularity (MSM-All)');
xlabel('Structural resolution, \gamma');
ylabel('Inter-subject coupling, log_{10}(\omega)'); %x^{3}_{4}
% Top middle plot(tile)
nexttile
Community = info(:,5);
scatter(Gamma,Omega,[],Community,'filled');
colorbar;
title('Communities (MSM-All)');
xlabel('Structural resolution, \gamma');
ylabel('Inter-subject coupling, log_{10}(\omega)'); %x^{3}_{4}
% Top right plot(tile)
nexttile
Community = info(:,9);
scatter(Gamma,Omega,[],Community,'filled');
colorbar;
title('Entropy (MSM-All)');
xlabel('Structural resolution, \gamma');
ylabel('Inter-subject coupling, log_{10}(\omega)');
% Bottom left plot(tile)
nexttile
Community = info(:,4);
scatter(Gamma,Omega,[],Community,'filled');
colorbar;
title('Modularity (HA)');
xlabel('Structural resolution, \gamma');
ylabel('Inter-subject coupling, log_{10}(\omega)');
% Bottom middle plot(tile)
nexttile
Community = info(:,6);
scatter(Gamma,Omega,[],Community,'filled');
colorbar;
title('Communities (HA)');
xlabel('Structural resolution, \gamma');
ylabel('Inter-subject coupling, log_{10}(\omega)');
% Bottom right plot(tile)
nexttile
Community = info(:,10);
scatter(Gamma,Omega,[],Community,'filled');
colorbar;
title('Entropy (HA)');
xlabel('Structural resolution, \gamma');
ylabel('Inter-subject coupling, log_{10}(\omega)');
% Reduce the spacing around the perimeter of the layout and around each tile
t.Padding = 'compact';
t.TileSpacing = 'compact';


