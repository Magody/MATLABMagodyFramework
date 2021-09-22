%% Clean up
clc;
clear all; %#ok<CLALL>
close all;
seed_rng = 44;

%% Libs
path_to_framework = "/home/magody/programming/MATLAB/deep_learning_from_scratch/magody_framework";
addpath(genpath(path_to_framework));

%% Init general parameters
generate_rng(seed_rng)
verbose_level = 10;

matrix_mask = [
    [-1, 2, 1, 1, 1]; 
    [-2, 1, 1, -1, 1]; 
    [-1, 1, 1, -2, 1]; 
    [1, 1, 0, -1, 1]; 
    [-2, -1, -1, -1, 2]
];
matrix = matrix_mask;
terminals = [[1, 2]; [2, 1]; [5, 1]; [3, 4]; [5, 5]];
position = [4, 3];
matrix(position(1), position(2)) = 0;

shape_input = [size(matrix_mask), 1];
state_length = prod(shape_input);

initial_state = matrix;


m = length(matrix);
n = length(matrix(1, :));

rewards = containers.Map({'collision', 'positive', 'negative', 'visited'}, ...
    {-2, 1, -1, -2});

context = containers.Map( ...
        {'matrix_mask', 'initial_matrix', 'initial_position', ...
            'initial_state', 'm', 'n', ...
            'terminals', 'shape_input', ...
            'rewards'}, ...
        {matrix_mask, matrix, position, ...
        initial_state, m, n, ...
        terminals, shape_input, rewards});
    
%% Init Q neural network and its hyper parameters

generate_rng(seed_rng)
total_episodes = 100;
epochs = 1; % epochs inside each NN
learning_rate = 0.001;
batch_size = 32;
gamma = 0.99;
epsilon = 1;
decay_rate_alpha = 0.01;
experience_replay_reserved_space = 30;
gameReplayStrategy = 1;
loss_type = "mse";

% Pooling("mean"), ...
sequential_conv_network = Sequential({
    Convolutional([2, 2], 3, 0, 1, shape_input), ...
    Activation("tanh"), ....
    Reshape(), ...
});

input_dense = sequential_conv_network.shape_output;

sequential_network = Sequential({
    Dense(16, input_dense), ...
    Activation("relu"), ...
    Dense(16), ...
    Activation("relu"), ...
    Dense(4), ...
});

nnConfig = NNConfig(epochs, learning_rate, batch_size, loss_type);
nnConfig.decay_rate_alpha = decay_rate_alpha;

qLearningConfig = QLearningConfig(gamma, epsilon, gameReplayStrategy, experience_replay_reserved_space, total_episodes);
    
q_neural_network = QNeuralNetwork(sequential_conv_network, sequential_network, nnConfig, qLearningConfig, @executeEpisodeCNNGridWorld);    


%% Train
t_begin = tic;
history_episodes_train = q_neural_network.runEpisodes(@getRewardCNNGridWorld, false, context, verbose_level-1);
t_end = toc(t_begin);
fprintf("Elapsed time: %.4f [minutes]\n", t_end/60);

mean_reward_train = mean(history_episodes_train('rewards'));
fprintf("Mean reward train: %.4f\n", mean_reward_train);

figure(1);
subplot(1,2,1)
plot(history_episodes_train('rewards'));

linear_update_costs = [];
update_costs_by_episode = history_episodes_train('update_costs');

for i=1:length(update_costs_by_episode)
    costs = update_costs_by_episode{i};
    linear_update_costs = [linear_update_costs; costs(:)]; 
end
subplot(1,2,2)
plot(linear_update_costs);

%% Test

history_episodes_test = q_neural_network.runEpisodes(@getRewardCNNGridWorld, true, context, verbose_level-1);
mean_reward = mean(history_episodes_test('rewards'));
fprintf("Mean reward: %.4f\n", mean_reward);

%% TSNE, dimension reduction. Table of features and plots

valid_replay = getCellsNotEmpty(gqnn.gameReplay);

len_gamereplay = length(gqnn.gameReplay);

dataX = [];
dataY = [];

input_dim = length(gqnn.q_neural_network.shape_input);

index_valid = 0;

for numExample=1:len_gamereplay
    
    reward_er = valid_replay{numExample}.reward;
    
    if reward_er < 0
        continue;
    end
    
    index_valid = index_valid + 1;
    
    s = valid_replay{numExample}.state;    
    action_er = valid_replay{numExample}.action;

    if input_dim == 1
        dataX(:, index_valid) = s;
    elseif input_dim == 2
        dataX(:, :, index_valid) = s;
    elseif input_dim == 3
        dataX(:, :, :, index_valid) = s;
    end

    dataY(index_valid, :) = sparse_one_hot_encoding(action_er, 4);

end


fprintf("Generating table of features\n");
len_data_train = size(dataX, 4);
features_matrix = zeros([len_data_train, gqnn.q_neural_network.conv_network{end}.shape_output]);
classes = {};

class_names = ["up", "right", "down", "left"];



for index_data=1:len_data_train
    
    x = dataX(:, :, :, index_data);
    [~, y] = max(dataY(index_data, :));  % argmax
    features = gqnn.q_neural_network.forwardConv(x);
    
    
    classes{index_data, 1} = char(class_names(y));
    features_matrix(index_data, :) = features';
            
end

%% TSNE plots
% features_matrix = table2array(features_table);

options = containers.Map();
options('limit_samples') = 10000;
options('plot_point_size') = 20;
options('include3D') = false;
options('save') = true;

options('dir') = 'figures/';
options('algorithms') =  [ ...
    struct('distance', 'euclidean','plot', struct('title', 'Euclidean')), ...
    % struct('distance', 'chebychev','plot', struct('title', 'Chebychev')), ...
];

history_tsne = generateTSNE(features_matrix, classes, options, verbose_level-1);
k = keys(history_tsne);
v = values(history_tsne);
for i=1:length(k)
    fprintf("Results for algorithm %s: ", k{i});
    disp(v{i});
end
